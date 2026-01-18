"""FastAPI backend that mirrors the CLI demo flows in scripts/demo/interact.py."""
from __future__ import annotations

import argparse
import base64
import binascii
import logging
import os
import shlex
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mmengine.config import Config
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import internal modules from the F-LMM project
from scripts.demo.interact import (  # noqa: E402
    SessionState,
    clear_history,
    handle_ground,
    handle_load,
    history_turns,
    load_model,
    parse_args as cli_parse_args,
    perform_ground_custom,
    pipeline_default_ask,
)

from scripts.demo.web.backend.task_queue.db import connect as tq_connect, init_db as tq_init_db  # noqa: E402
from scripts.demo.web.backend.task_queue.queue import enqueue_task, get_task  # noqa: E402
from scripts.demo.web.backend.task_queue.paths import SessionPaths  # noqa: E402
from scripts.demo.web.backend.prompt_overrides import apply_prompt_overrides  # noqa: E402

APP_VERSION = "0.1.0"
LOGGER = logging.getLogger("flmm.web")


# --- Request Models ---

class APIModel(BaseModel):
    """Base Pydantic model with strict field checking."""

    model_config = ConfigDict(extra="forbid")


class SessionIdentRequest(APIModel):
    session_id: str = Field(..., description="Session identifier")


class SessionCreateRequest(APIModel):
    image_path: Optional[str] = Field(
        None, description="Optional path of the image to load immediately."
    )
    image_base64: Optional[str] = Field(
        None, description="Optional base64 image payload to preload."
    )

    @model_validator(mode="after")
    def validate_payload(self):
        if self.image_path and self.image_base64:
            raise ValueError("Provide either image_path or image_base64, not both.")
        return self


class LoadImageRequest(SessionIdentRequest):
    image_path: Optional[str] = Field(
        None, description="Absolute/relative image path accessible to the backend."
    )
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image content."
    )

    @model_validator(mode="after")
    def validate_payload(self):
        if self.image_path and self.image_base64:
            raise ValueError("Provide either image_path or image_base64, not both.")
        if not self.image_path and not self.image_base64:
            raise ValueError("image_path or image_base64 is required.")
        return self


class AskRequest(SessionIdentRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "session_id": "session_abc123",
                    "question": "What is in the image?",
                    "enable_roi": False,
                }
            ]
        },
    )
    mode: Literal["default"] = Field(
        "default", description="Ask mode (default only)."
    )
    question: str = Field("", description="User question.")
    index: Optional[int] = Field(
        None, description="ROI or CoT index when mode != default."
    )
    reset_history: bool = Field(
        False, description="Clear history before asking (default-only)."
    )
    enable_roi: bool = Field(
        True, description="Enable ROI re-answer (visual_cot_resample)."
    )

    @model_validator(mode="after")
    def validate_payload(self):
        if self.mode == "default" and not self.question.strip():
            raise ValueError("Question must not be empty in default mode.")
        return self


class GroundPhrase(APIModel):
    text: Optional[str] = Field(
        None, description="Phrase text to match in the latest answer."
    )
    char_span: Optional[List[int]] = Field(
        None, description="Character span [start, end) in answer text."
    )
    token_span: Optional[List[int]] = Field(
        None, description="Token span [start, end) in answer tokens."
    )
    occurrence: Optional[int] = Field(
        None, description="Occurrence index when text appears multiple times (0-based)."
    )


class GroundRequest(SessionIdentRequest):
    indices: Optional[List[int]] = Field(
        None, description="Phrase indices to ground (legacy)."
    )
    phrases: Optional[List[GroundPhrase]] = Field(
        None,
        description="Custom phrases with optional spans for grounding.",
    )

    @model_validator(mode="after")
    def validate_payload(self):
        has_indices = bool(self.indices) and isinstance(self.indices, list)
        has_phrases = bool(self.phrases) and isinstance(self.phrases, list)
        if not has_indices and not has_phrases:
            raise ValueError("GROUND payload requires 'indices' or 'phrases'.")
        return self


class ClearRequest(SessionIdentRequest):
    pass


class SessionResetRequest(SessionIdentRequest):
    pass


class TaskEnqueueRequest(APIModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "type": "ASK",
                    "session_id": "session_abc123",
                    "payload": {
                        "question": "What is in the image?",
                        "image_path": "results/sessions/session_abc123/images/upload.png",
                        "enable_roi": False,
                    },
                }
            ]
        },
    )
    type: Literal['ASK', 'GROUND', 'ATTN_I2T', 'ATTN_T2I'] = Field(..., description='Task type')
    session_id: str = Field(..., description='Session identifier')
    payload: Dict[str, Any] = Field(default_factory=dict, description='Task input payload')
    turn_idx: Optional[int] = Field(None, description='Optional turn index for dedupe')
    turn_uid: Optional[str] = Field(None, description='Optional turn UID')


# --- Response Models ---

TaskType = Literal["ASK", "GROUND", "ATTN_I2T", "ATTN_T2I"]
TaskStatus = Literal["PENDING", "RUNNING", "DONE", "FAILED"]


class HistoryItem(BaseModel):
    role: str = Field(..., description="Message role (user/assistant/system).")
    text: str = Field(..., description="Message text.")


class ImageInfo(BaseModel):
    name: Optional[str] = Field(None, description="Image filename if available.")
    path: Optional[str] = Field(None, description="Local path for the image.")
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    url: Optional[str] = Field(None, description="Public URL for the image.")


class SessionPayload(BaseModel):
    session_id: str = Field(..., description="Session identifier.")
    session_dir: str = Field(..., description="Relative session directory.")
    session_dir_url: str = Field(..., description="Public URL for the session directory.")
    history: List[HistoryItem] = Field(default_factory=list, description="Conversation history.")
    history_turns: int = Field(..., description="Number of turns in history.")
    image: Optional[ImageInfo] = Field(None, description="Current image info.")
    created_at: str = Field(..., description="Session creation time (ISO-8601).")
    last_active: str = Field(..., description="Last active time (ISO-8601).")


class PhrasePayload(BaseModel):
    index: int = Field(..., description="Phrase index.")
    text: str = Field(..., description="Phrase text.")
    char_span: List[int] = Field(default_factory=list, description="Character span of the phrase.")
    token_span: List[int] = Field(default_factory=list, description="Token span of the phrase.")


class GroundRecord(BaseModel):
    index: int = Field(..., description="Record index.")
    phrase: str = Field(..., description="Phrase text.")
    overlay_url: Optional[str] = Field(None, description="Overlay visualization URL.")
    mask_url: Optional[str] = Field(None, description="Mask visualization URL.")
    roi_url: Optional[str] = Field(None, description="ROI image URL.")
    char_span: List[int] = Field(default_factory=list, description="Character span of the phrase.")
    token_span: List[int] = Field(default_factory=list, description="Token span of the phrase.")
    bbox: List[float] = Field(default_factory=list, description="Bounding box [x1,y1,x2,y2].")


class GroundPayload(BaseModel):
    records: List[GroundRecord] = Field(default_factory=list, description="Grounding records.")
    turn_dir: str = Field(..., description="Relative turn directory.")
    turn_url: str = Field(..., description="Public URL for the turn directory.")
    ground_dir: str = Field(..., description="Relative grounding directory.")
    ground_url: str = Field(..., description="Public URL for the grounding directory.")
    round_dir: str = Field(..., description="Backward-compatible alias for ground_dir.")
    round_url: str = Field(..., description="Backward-compatible alias for ground_url.")
    history: List[HistoryItem] = Field(default_factory=list, description="Updated history.")


class VerificationPayload(BaseModel):
    original_answer: Optional[str] = Field(None, description="Answer before ROI verification.")
    used: Optional[bool] = Field(None, description="Whether ROI verification was used.")
    roi_answer: Optional[str] = Field(None, description="Answer after ROI verification.")
    roi_bbox: Optional[List[float]] = Field(None, description="ROI bounding box.")
    roi_prompt: Optional[str] = Field(None, description="Prompt used for ROI re-answering.")
    roi_prompt_raw: Optional[str] = Field(None, description="Unprocessed ROI prompt.")
    error: Optional[str] = Field(None, description="Verification error if any.")
    records: Optional[List[GroundRecord]] = Field(None, description="Grounding records used.")
    turn_dir: Optional[str] = Field(None, description="Relative turn directory.")
    turn_url: Optional[str] = Field(None, description="Public URL for the turn directory.")
    ground_dir: Optional[str] = Field(None, description="Relative grounding directory.")
    ground_url: Optional[str] = Field(None, description="Public URL for the grounding directory.")
    round_dir: Optional[str] = Field(None, description="Backward-compatible alias for ground_dir.")
    round_url: Optional[str] = Field(None, description="Backward-compatible alias for ground_url.")

    model_config = ConfigDict(extra="allow")


class AskPayload(BaseModel):
    mode: str = Field(..., description="Ask mode.")
    raw_answer: Optional[str] = Field(None, description="Answer before ROI re-answer.")
    answer: str = Field(..., description="Final answer.")
    phrases: List[PhrasePayload] = Field(default_factory=list, description="Extracted phrases.")
    verification: Optional[VerificationPayload] = Field(None, description="Verification details.")
    history: List[HistoryItem] = Field(default_factory=list, description="Updated history.")
    history_turns: int = Field(..., description="Number of turns in history.")


class ClearPayload(BaseModel):
    history: List[HistoryItem] = Field(default_factory=list, description="Cleared history.")
    history_turns: int = Field(..., description="History turn count after clear.")


class HealthPayload(BaseModel):
    version: str = Field(..., description="Backend version.")
    sessions: int = Field(..., description="Active session count.")


class TaskRecord(BaseModel):
    id: int = Field(..., description="Task identifier.")
    type: TaskType = Field(..., description="Task type.")
    status: TaskStatus = Field(..., description="Task status.")
    session_id: str = Field(..., description="Session identifier.")
    turn_idx: Optional[int] = Field(None, description="Associated turn index.")
    turn_uid: Optional[str] = Field(None, description="Associated turn UID.")
    input_json: Optional[Any] = Field(None, description="Task input payload.")
    output_json: Optional[Any] = Field(None, description="Task output payload.")
    error: Optional[str] = Field(None, description="Error message if failed.")
    worker_id: Optional[str] = Field(None, description="Worker handling the task.")
    created_at: str = Field(..., description="Creation timestamp (ISO-8601).")
    updated_at: str = Field(..., description="Last update timestamp (ISO-8601).")


class TaskEnqueuePayload(BaseModel):
    task_id: int = Field(..., description="Enqueued task ID.")


class TaskPollPayload(BaseModel):
    task: TaskRecord = Field(..., description="Task record.")


class ResponseBase(BaseModel):
    status: Literal["ok", "error"] = Field("ok", description="Request status.")
    message: str = Field("", description="Human-readable message.")


class HealthResponse(ResponseBase):
    data: HealthPayload = Field(..., description="Health payload.")


class SessionResponse(ResponseBase):
    data: SessionPayload = Field(..., description="Session payload.")


class AskResponse(ResponseBase):
    data: AskPayload = Field(..., description="Ask payload.")


class GroundResponse(ResponseBase):
    data: GroundPayload = Field(..., description="Grounding payload.")


class ClearResponse(ResponseBase):
    data: ClearPayload = Field(..., description="Clear payload.")


class TaskEnqueueResponse(ResponseBase):
    data: TaskEnqueuePayload = Field(..., description="Task enqueue payload.")


class TaskPollResponse(ResponseBase):
    data: TaskPollPayload = Field(..., description="Task polling payload.")


class DeleteSessionResponse(ResponseBase):
    data: Dict[str, Any] = Field(default_factory=dict, description="Empty payload.")

@dataclass
class SessionEntry:
    """Represents a single user session with its own state and lock."""
    session_id: str
    state: SessionState
    created_at: datetime
    last_active: datetime
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touch(self):
        """Update the last active timestamp."""
        self.last_active = datetime.utcnow()


class SessionStore:
    """
    In-memory session registry with TTL (Time-To-Live) + cleanup thread.
    Manages multiple user sessions concurrently.
    """

    def __init__(
        self,
        model: Any,
        args: argparse.Namespace,
        result_root: Path,
        ttl: timedelta,
        cleanup_interval: int,
    ):
        self._model = model
        self._args = args
        self._result_root = result_root
        self._ttl = ttl
        self._cleanup_interval = max(cleanup_interval, 60)
        self._sessions: Dict[str, SessionEntry] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sweeper = threading.Thread(
            target=self._cleanup_loop, name="flmm-session-cleaner", daemon=True
        )
        self._sweeper.start()

    def shutdown(self):
        self._stop_event.set()
        self._sweeper.join(timeout=2)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create_session(self) -> SessionEntry:
        state = SessionState(model=self._model, args=self._args, result_root=self._result_root)
        entry = SessionEntry(
            session_id=str(uuid.uuid4()),
            state=state,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
        )
        with entry.lock:
            self._reset_workspace(entry)
            entry.state.reset_answer()
        with self._lock:
            self._sessions[entry.session_id] = entry
        return entry

    def get(self, session_id: str) -> SessionEntry:
        with self._lock:
            entry = self._sessions.get(session_id)
        if not entry:
            raise KeyError(f"Session '{session_id}' not found.")
        entry.touch()
        return entry

    def reset(self, session_id: str) -> SessionEntry:
        entry = self.get(session_id)
        with entry.lock:
            self._reset_state(entry.state)
            self._reset_workspace(entry)
        entry.touch()
        return entry

    def delete(self, session_id: str) -> SessionEntry:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if not entry:
            raise KeyError(f"Session '{session_id}' not found.")
        with entry.lock:
            self._remove_dir(entry.state.session_dir)
        return entry

    def drop_expired(self) -> int:
        now = datetime.utcnow()
        removed: List[SessionEntry] = []
        with self._lock:
            for sid, entry in list(self._sessions.items()):
                if now - entry.last_active <= self._ttl:
                    continue
                if not entry.lock.acquire(blocking=False):
                    continue
                try:
                    removed.append(self._sessions.pop(sid))
                finally:
                    entry.lock.release()
        for entry in removed:
            self._remove_dir(entry.state.session_dir)
        if removed:
            LOGGER.info("Cleaned %d expired session(s).", len(removed))
        return len(removed)

    def _cleanup_loop(self):
        while not self._stop_event.wait(self._cleanup_interval):
            try:
                self.drop_expired()
            except Exception as exc:
                LOGGER.warning("Session cleanup failed: %s", exc)

    def _reset_state(self, state: SessionState):
        state.reset_answer()
        state.current_image = None
        state.current_image_path = None
        state.last_records = []
        state.ground_id = 0
        state.attn_counters = {"i2t": 0, "t2i": 0}
        state.turn_idx = 0

    def _reset_workspace(self, entry: SessionEntry):
        entry.state.session_id = entry.session_id
        entry.state.session_paths = SessionPaths(self._result_root, entry.session_id)
        session_root = entry.state.session_paths.session_root
        # Remove any previous workspace to ensure a clean slate
        self._remove_dir(getattr(entry.state, "session_dir", session_root))
        self._remove_dir(session_root)
        session_root.mkdir(parents=True, exist_ok=True)
        entry.state.session_paths.turns_dir.mkdir(parents=True, exist_ok=True)
        entry.state.session_paths.images_dir.mkdir(parents=True, exist_ok=True)
        entry.state.session_dir = session_root
        entry.state.turn_idx = 0
        entry.state.ground_id = 0
        entry.state.attn_counters = {"i2t": 0, "t2i": 0}

    def _remove_dir(self, path: Path):
        try:
            if path and path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            LOGGER.warning("Failed to remove %s: %s", path, exc)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%s, fallback=%d", name, raw, default)
        return default


def ensure_mount_path(value: str) -> str:
    path = value or "/results"
    path = "/" + path.lstrip("/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path


def parse_cors_origins(raw: str) -> List[str]:
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def verify_topk() -> int:
    return max(env_int("FLMM_WEB_VERIFY_TOPK", 1), 1)


def configure_logging():
    level_name = os.getenv("FLMM_WEB_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.setLevel(level)
    mm_level_name = os.getenv("FLMM_WEB_MMENGINE_LOG_LEVEL", "WARNING").upper()
    mm_level = getattr(logging, mm_level_name, logging.WARNING)
    logging.getLogger("mmengine").setLevel(mm_level)
    LOGGER.info("Logging configured. app=%s mmengine=%s", level_name, mm_level_name)


def build_args_from_env() -> argparse.Namespace:
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["flmm-web-backend"]
        args = cli_parse_args()
    finally:
        sys.argv = argv_backup
    args.config = os.getenv("FLMM_WEB_CONFIG", args.config)
    args.checkpoint = os.getenv("FLMM_WEB_CHECKPOINT", args.checkpoint)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (REPO_ROOT / ckpt_path).resolve()
        args.checkpoint = str(ckpt_path)
        LOGGER.info("Using checkpoint: %s (abs: %s)", os.getenv("FLMM_WEB_CHECKPOINT"), args.checkpoint)
    else:
        LOGGER.info("Using checkpoint: <from config>")
    args.device = os.getenv("FLMM_WEB_DEVICE", args.device)
    args.device_map = os.getenv("FLMM_WEB_DEVICE_MAP", args.device_map)
    args.device_max_memory = os.getenv(
        "FLMM_WEB_DEVICE_MAX_MEMORY", args.device_max_memory
    )
    args.results_dir = os.getenv("FLMM_WEB_RESULTS_DIR", args.results_dir)
    args.max_new_tokens = env_int("FLMM_WEB_MAX_NEW_TOKENS", args.max_new_tokens)
    args.phrase_max_tokens = env_int(
        "FLMM_WEB_PHRASE_MAX_TOKENS", args.phrase_max_tokens
    )
    args.max_phrases = env_int("FLMM_WEB_MAX_PHRASES", args.max_phrases)
    args.max_history_turns = env_int(
        "FLMM_WEB_MAX_HISTORY_TURNS", args.max_history_turns
    )
    args.inspect_prompt = os.getenv("FLMM_WEB_INSPECT_PROMPT", args.inspect_prompt)
    args.extra_prompt = os.getenv("FLMM_WEB_EXTRA_PROMPT", args.extra_prompt or "")
    args.prompt_file = os.getenv("FLMM_WEB_PROMPT_FILE")
    args.no_sam = env_bool("FLMM_WEB_NO_SAM", args.no_sam)
    args.image = None
    args.no_model = env_bool("FLMM_WEB_NO_MODEL", False)
    return args


class BackendState:
    """
    Global backend state holding the model, configuration, and session store.
    Initialized once when the FastAPI app starts.
    """

    def __init__(self):
        configure_logging()
        args = build_args_from_env()
        cfg = None
        model = None
        if args.no_model:
            LOGGER.info("FLMM_WEB_NO_MODEL=1 -> backend will not load the model; use /tasks with worker for inference.")
        else:
            cfg_path = Path(args.config)
            if not cfg_path.is_absolute():
                cfg_path = (REPO_ROOT / cfg_path).resolve()
            LOGGER.info("Loading config: %s", cfg_path)
            cfg = Config.fromfile(cfg_path)
            LOGGER.info("Loading model...")
            if args.checkpoint:
                LOGGER.info("Checkpoint override resolved path: %s (exists=%s)", args.checkpoint, Path(args.checkpoint).exists())
            apply_prompt_overrides(cfg, args, args.prompt_file)
            # Load the model using the provided config and args
            model = load_model(cfg, args)
        self.args = args
        self.cfg = cfg
        self.model = model
        self.results_dir = Path(args.results_dir).expanduser().resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_mount = ensure_mount_path(os.getenv("FLMM_WEB_RESULTS_MOUNT", "/results"))
        task_db_env = os.getenv("FLMM_WEB_TASK_DB")
        self.task_db_path = Path(task_db_env).expanduser().resolve() if task_db_env else (self.results_dir / "task_queue.db")
        self.task_conn = tq_connect(self.task_db_path)
        tq_init_db(self.task_conn)
        
        # Session management settings
        ttl_minutes = env_int("FLMM_WEB_SESSION_TTL_MINUTES", 60)
        cleanup_seconds = env_int("FLMM_WEB_SESSION_SWEEP_SECONDS", 300)
        self.session_store = SessionStore(
            model=model,
            args=args,
            result_root=self.results_dir,
            ttl=timedelta(minutes=max(ttl_minutes, 1)),
            cleanup_interval=cleanup_seconds,
        )
        # Lock to ensure thread-safe model inference
        self.model_lock = threading.Lock()
        LOGGER.info(
            "Backend ready: ttl=%dmin results=%s max_new_tokens=%s max_history_turns=%s",
            ttl_minutes,
            self.results_dir,
            args.max_new_tokens,
            args.max_history_turns,
        )

    def result_url(self, path: Optional[Path]) -> Optional[str]:
        """Convert a local result path to a URL accessible by the frontend."""
        if path is None:
            return None
        try:
            rel = path.resolve().relative_to(self.results_dir)
        except ValueError:
            return path.as_posix()
        rel_str = rel.as_posix().lstrip("/")
        return f"{self.results_mount}/{rel_str}" if rel_str else self.results_mount

    def relative_result_path(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            rel = path.resolve().relative_to(self.results_dir)
            return rel.as_posix()
        except ValueError:
            return path.as_posix()


def build_response(data: Optional[Dict[str, Any]] = None, message: str = "", status: str = "ok"):
    return {"status": status, "data": data or {}, "message": message}


def error_response(message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content=build_response(status="error", message=message),
    )


EXAMPLE_SESSION_ID = "8b2f3c8a-3a35-4c5a-9d71-2c8a3b0f1234"
EXAMPLE_SESSION_DIR = f"sessions/{EXAMPLE_SESSION_ID}"
EXAMPLE_SESSION_URL = f"/results/{EXAMPLE_SESSION_DIR}"
EXAMPLE_IMAGE_PATH = f"{EXAMPLE_SESSION_DIR}/images/upload_1705400000000.png"
EXAMPLE_IMAGE_URL = f"/results/{EXAMPLE_IMAGE_PATH}"
EXAMPLE_TURN_DIR = f"{EXAMPLE_SESSION_DIR}/turns/turn_0000"
EXAMPLE_TURN_URL = f"/results/{EXAMPLE_TURN_DIR}"
EXAMPLE_GROUND_DIR = f"{EXAMPLE_TURN_DIR}/ground/ground_0000"
EXAMPLE_GROUND_URL = f"/results/{EXAMPLE_GROUND_DIR}"

EXAMPLE_HISTORY = [
    {"role": "user", "text": "What is in the image?"},
    {"role": "assistant", "text": "A red car is parked in a lot."},
]

EXAMPLE_SESSION_PAYLOAD = {
    "session_id": EXAMPLE_SESSION_ID,
    "session_dir": EXAMPLE_SESSION_DIR,
    "session_dir_url": EXAMPLE_SESSION_URL,
    "history": EXAMPLE_HISTORY,
    "history_turns": 1,
    "image": {
        "name": "upload_1705400000000.png",
        "path": EXAMPLE_IMAGE_PATH,
        "width": 1024,
        "height": 768,
        "url": EXAMPLE_IMAGE_URL,
    },
    "created_at": "2026-01-16T12:34:56.000Z",
    "last_active": "2026-01-16T12:35:10.000Z",
}

EXAMPLE_PHRASES = [
    {"index": 0, "text": "red car", "char_span": [2, 9], "token_span": [1, 2]},
    {"index": 1, "text": "parking lot", "char_span": [22, 33], "token_span": [5, 6]},
]

EXAMPLE_GROUND_RECORDS = [
    {
        "index": 0,
        "phrase": "red car",
        "overlay_url": f"{EXAMPLE_GROUND_URL}/overlay_0000.png",
        "mask_url": f"{EXAMPLE_GROUND_URL}/mask_0000.png",
        "roi_url": f"{EXAMPLE_GROUND_URL}/roi_0000.png",
        "char_span": [2, 9],
        "token_span": [1, 2],
        "bbox": [120, 80, 460, 320],
    }
]

EXAMPLE_GROUND_PAYLOAD = {
    "records": EXAMPLE_GROUND_RECORDS,
    "turn_dir": EXAMPLE_TURN_DIR,
    "turn_url": EXAMPLE_TURN_URL,
    "ground_dir": EXAMPLE_GROUND_DIR,
    "ground_url": EXAMPLE_GROUND_URL,
    "round_dir": EXAMPLE_GROUND_DIR,
    "round_url": EXAMPLE_GROUND_URL,
}

EXAMPLE_VERIFICATION = {
    "original_answer": "A red car is parked in a lot.",
    "used": True,
    "roi_answer": "The car is red.",
    "roi_bbox": [120, 80, 460, 320],
    "roi_prompt": "Focus on the car and answer the question.",
    "roi_prompt_raw": "Focus on the car and answer the question.",
    **EXAMPLE_GROUND_PAYLOAD,
}

EXAMPLE_ASK_PAYLOAD = {
    "mode": "default",
    "raw_answer": "A red car is parked in a lot.",
    "answer": "The car is red.",
    "phrases": EXAMPLE_PHRASES,
    "verification": EXAMPLE_VERIFICATION,
    "history": EXAMPLE_HISTORY,
    "history_turns": 1,
}

EXAMPLE_TASK = {
    "id": 42,
    "type": "ASK",
    "status": "DONE",
    "session_id": EXAMPLE_SESSION_ID,
    "turn_idx": 0,
    "turn_uid": "turn_0000",
    "input_json": {"question": "What is in the image?", "image_path": EXAMPLE_IMAGE_PATH},
    "output_json": {
        "raw_answer": "A red car is parked in a lot.",
        "answer": "The car is red.",
        "roi_answer": "The car is red.",
    },
    "error": None,
    "worker_id": "worker-1",
    "created_at": "2026-01-16T12:34:56.000Z",
    "updated_at": "2026-01-16T12:34:58.000Z",
}

EXAMPLE_HEALTH_RESPONSE = build_response(
    data={"version": APP_VERSION, "sessions": 3}
)
EXAMPLE_SESSION_RESPONSE = build_response(data=EXAMPLE_SESSION_PAYLOAD)
EXAMPLE_SESSION_RESET_RESPONSE = build_response(
    data=EXAMPLE_SESSION_PAYLOAD, message="Session reset."
)
EXAMPLE_SESSION_DELETE_RESPONSE = build_response(message="Session deleted.")
EXAMPLE_LOAD_IMAGE_RESPONSE = build_response(
    data=EXAMPLE_SESSION_PAYLOAD, message="Image loaded."
)
EXAMPLE_ASK_RESPONSE = build_response(data=EXAMPLE_ASK_PAYLOAD)
EXAMPLE_GROUND_RESPONSE = build_response(
    data={**EXAMPLE_GROUND_PAYLOAD, "history": EXAMPLE_HISTORY}
)
EXAMPLE_TASK_ENQUEUE_RESPONSE = build_response(data={"task_id": 42})
EXAMPLE_TASK_POLL_RESPONSE = build_response(data={"task": EXAMPLE_TASK})
EXAMPLE_CLEAR_RESPONSE = build_response(
    data={"history": [], "history_turns": 0}
)
EXAMPLE_ERROR_RESPONSE = build_response(
    status="error", message="Session 'missing-session-id' not found."
)
EXAMPLE_MODEL_DISABLED_ASK_RESPONSE = build_response(
    status="error",
    message="Model disabled (FLMM_WEB_NO_MODEL=1). Please enqueue ASK via /tasks.",
)
EXAMPLE_MODEL_DISABLED_GROUND_RESPONSE = build_response(
    status="error",
    message="Model disabled (FLMM_WEB_NO_MODEL=1). Please enqueue GROUND via /tasks.",
)

RESPONSES_HEALTH = {
    200: {"content": {"application/json": {"example": EXAMPLE_HEALTH_RESPONSE}}},
}
RESPONSES_SESSION_CREATE = {
    200: {"content": {"application/json": {"example": EXAMPLE_SESSION_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_SESSION_RESET = {
    200: {"content": {"application/json": {"example": EXAMPLE_SESSION_RESET_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_SESSION_DELETE = {
    200: {"content": {"application/json": {"example": EXAMPLE_SESSION_DELETE_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_LOAD_IMAGE = {
    200: {"content": {"application/json": {"example": EXAMPLE_LOAD_IMAGE_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_ASK = {
    200: {"content": {"application/json": {"example": EXAMPLE_ASK_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
    503: {"content": {"application/json": {"example": EXAMPLE_MODEL_DISABLED_ASK_RESPONSE}}},
}
RESPONSES_GROUND = {
    200: {"content": {"application/json": {"example": EXAMPLE_GROUND_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
    503: {"content": {"application/json": {"example": EXAMPLE_MODEL_DISABLED_GROUND_RESPONSE}}},
}
RESPONSES_TASK_ENQUEUE = {
    200: {"content": {"application/json": {"example": EXAMPLE_TASK_ENQUEUE_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_TASK_POLL = {
    200: {"content": {"application/json": {"example": EXAMPLE_TASK_POLL_RESPONSE}}},
    404: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}
RESPONSES_CLEAR = {
    200: {"content": {"application/json": {"example": EXAMPLE_CLEAR_RESPONSE}}},
    400: {"content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}


def _clean_answer(text: str) -> str:
    """Remove internal placeholders (e.g., <|image_pad|>) for frontend display."""
    if not text:
        return text
    cleaned = text.replace("<|image_pad|>", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def snapshot_history(session: SessionState) -> List[Dict[str, str]]:
    return [{"role": item.get("role", ""), "text": item.get("text", "")} for item in session.history]


def image_info(backend: BackendState, session: SessionState) -> Optional[Dict[str, Any]]:
    if session.current_image is None:
        return None
    width, height = session.current_image.size
    path_str = str(session.current_image_path) if session.current_image_path else None
    name = session.current_image_path.name if session.current_image_path else None
    url = backend.result_url(session.current_image_path) if session.current_image_path else None
    return {"name": name, "path": path_str, "width": width, "height": height, "url": url}


def to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python built-ins for JSON encoding."""
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    # numpy.ndarray or torch.Tensor may expose tolist()
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def session_payload(backend: BackendState, entry: SessionEntry) -> Dict[str, Any]:
    session = entry.state
    return {
        "session_id": entry.session_id,
        "session_dir": backend.relative_result_path(session.session_dir),
        "session_dir_url": backend.result_url(session.session_dir),
        "history": snapshot_history(session),
        "history_turns": history_turns(session),
        "image": image_info(backend, session),
        "created_at": entry.created_at.isoformat(),
        "last_active": entry.last_active.isoformat(),
    }


def build_phrase_payload(session: SessionState) -> List[Dict[str, Any]]:
    payload = []
    for idx, candidate in enumerate(session.phrases):
        payload.append(
            {
                "index": idx,
                "text": candidate.text,
                "char_span": candidate.char_span,
                "token_span": candidate.token_span,
            }
        )
    return payload


def ground_payload(backend: BackendState, session: SessionState) -> Dict[str, Any]:
    records = []
    for idx, record in enumerate(session.last_records):
        records.append(
            {
                "index": idx,
                "phrase": record.phrase_text,
                "overlay_url": backend.result_url(record.overlay_path),
                "mask_url": backend.result_url(record.mask_path),
                "roi_url": backend.result_url(record.roi_path),
                "char_span": to_native(record.char_span),
                "token_span": to_native(record.token_span),
                "bbox": to_native(record.bbox),
            }
        )
    ground_idx = max(session.ground_id - 1, 0)
    turn_dir = session.session_paths.turn_dir(session.turn_idx)
    ground_dir = session.session_paths.ground_dir(session.turn_idx, ground_idx)
    return {
        "records": records,
        "turn_dir": backend.relative_result_path(turn_dir),
        "turn_url": backend.result_url(turn_dir),
        "ground_dir": backend.relative_result_path(ground_dir),
        "ground_url": backend.result_url(ground_dir),
        # backward-compatible aliases
        "round_dir": backend.relative_result_path(ground_dir),
        "round_url": backend.result_url(ground_dir),
    }


def select_auto_phrases(session: SessionState, limit: int) -> List[int]:
    if not session.phrases:
        return []
    indices: List[int] = []
    for idx in range(len(session.phrases)):
        indices.append(idx)
        if len(indices) >= limit:
            break
    return indices


def latest_assistant_text(session: SessionState) -> Optional[str]:
    for item in reversed(session.history):
        if item.get("role") == "assistant":
            return item.get("text", "")
    return None


def decode_base64_image(session: SessionState, payload: str) -> Path:
    header, _, data = payload.partition(",")
    encoded = data or header
    try:
        binary = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 payload.") from exc
    ext = ".png"
    header_lower = header.lower()
    if "jpeg" in header_lower or "jpg" in header_lower:
        ext = ".jpg"
    elif "gif" in header_lower:
        ext = ".gif"
    session.session_paths.images_dir.mkdir(parents=True, exist_ok=True)
    file_path = session.session_paths.images_dir / f"upload_{int(time.time() * 1000)}{ext}"
    file_path.write_bytes(binary)
    return file_path


def resolve_image_source(session: SessionState, path: Optional[str], base64_payload: Optional[str]) -> str:
    if path:
        return path
    if base64_payload:
        stored = decode_base64_image(session, base64_payload)
        return str(stored)
    raise ValueError("image_path or image_base64 must be provided.")


def build_ask_cli(request: AskRequest) -> str:
    # legacy placeholder; only default mode is supported now
    tokens: List[str] = []
    if request.reset_history:
        tokens.append("--reset-history")
    if request.question.strip():
        tokens.append(request.question)
    return shlex.join(tokens) if tokens else ""


# --- FastAPI App Setup ---

backend = BackendState()
app = FastAPI(title="F-LMM Web Backend", version=APP_VERSION)
app.state.backend = backend

# Mount the results directory to serve static files (images, masks, etc.)
app.mount(
    backend.results_mount,
    StaticFiles(directory=str(backend.results_dir), html=False),
    name="results",
)

# Enable CORS for development (allows frontend to call the backend from different origins)
cors_env = os.getenv("FLMM_WEB_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(cors_env),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def _shutdown():
    """Cleanup session store on application shutdown."""
    backend.session_store.shutdown()
    try:
        backend.task_conn.close()
    except Exception:
        pass


@app.get("/healthz", response_model=HealthResponse, responses=RESPONSES_HEALTH)
def healthz():
    """Health check endpoint."""
    return build_response(
        data={
            "version": APP_VERSION,
            "sessions": backend.session_store.count(),
        }
    )


def _with_session(session_id: str) -> SessionEntry:
    """Helper to retrieve a session or raise an error if not found."""
    try:
        return backend.session_store.get(session_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


@app.post("/session", response_model=SessionResponse, responses=RESPONSES_SESSION_CREATE)
@app.post("/session/create", response_model=SessionResponse, responses=RESPONSES_SESSION_CREATE)
def create_session(request: SessionCreateRequest):
    """Create a new session, optionally preloading an image."""
    try:
        entry = backend.session_store.create_session()
        if request.image_path or request.image_base64:
            source = resolve_image_source(entry.state, request.image_path, request.image_base64)
            with entry.lock:
                handle_load(entry.state, source)
        return build_response(data=session_payload(backend, entry))
    except Exception as exc:
        LOGGER.exception("Failed to create session: %s", exc)
        return error_response(str(exc))


@app.post("/session/reset", response_model=SessionResponse, responses=RESPONSES_SESSION_RESET)
def reset_session(request: SessionResetRequest):
    """Reset the state of an existing session."""
    try:
        entry = backend.session_store.reset(request.session_id)
        return build_response(
            data=session_payload(backend, entry), message="Session reset."
        )
    except Exception as exc:
        LOGGER.exception("Failed to reset session: %s", exc)
        return error_response(str(exc))


@app.delete("/session/{session_id}", response_model=DeleteSessionResponse, responses=RESPONSES_SESSION_DELETE)
def delete_session(session_id: str):
    """Delete a session and its associated workspace."""
    try:
        backend.session_store.delete(session_id)
        return build_response(message="Session deleted.")
    except Exception as exc:
        LOGGER.exception("Failed to delete session: %s", exc)
        return error_response(str(exc))


@app.post("/load_image", response_model=SessionResponse, responses=RESPONSES_LOAD_IMAGE)
def load_image(request: LoadImageRequest):
    """Load an image into an existing session."""
    try:
        entry = _with_session(request.session_id)
        source = resolve_image_source(entry.state, request.image_path, request.image_base64)
        with entry.lock:
            handle_load(entry.state, source)
        return build_response(
            data=session_payload(backend, entry),
            message="Image loaded.",
        )
    except Exception as exc:
        LOGGER.exception("Load image failed: %s", exc)
        return error_response(str(exc))


@app.post("/ask", response_model=AskResponse, responses=RESPONSES_ASK)
def ask(request: AskRequest):
    """
    Handle a user question. 
    Pipeline: Generate answer -> Extract phrases -> Auto-ground -> ROI re-answer.
    """
    try:
        if backend.model is None:
            return error_response("Model disabled (FLMM_WEB_NO_MODEL=1). Please enqueue ASK via /tasks.", status_code=503)
        entry = _with_session(request.session_id)
        with entry.lock:
            data: Dict[str, Any] = {"mode": "default"}
            with backend.model_lock:
                # Run the inference pipeline
                result = pipeline_default_ask(
                    entry.state,
                    request.question,
                    reset_history=request.reset_history,
                    auto_topk=verify_topk(),
                    enable_roi=request.enable_roi,
                )
            
            # Clean the answer for display (remove internal tokens)
            raw_answer = _clean_answer(
                result.get("raw_answer") or result.get("original_answer") or ""
            )
            answer = _clean_answer(result.get("answer", "")) or raw_answer
            print("[debug]Answer :", answer)
            
            data.update(
                {
                    "raw_answer": raw_answer,
                    "answer": answer,
                    "phrases": build_phrase_payload(entry.state),
                }
            )
            
            verification = result.get("verification")
            if verification:
                # Enrich verification data with file URLs for masks/ROIs
                ground_data = ground_payload(backend, entry.state)
                verification.update(ground_data)
                
                # Clean ROI prompt if present
                roi_prompt_raw = verification.get("roi_prompt")
                if isinstance(roi_prompt_raw, str):
                    verification["roi_prompt_raw"] = roi_prompt_raw
                    verification["roi_prompt"] = _clean_answer(roi_prompt_raw)
                
                data["verification"] = verification
            
            data.update(
                {
                    "history": snapshot_history(entry.state),
                    "history_turns": history_turns(entry.state),
                }
            )
        return build_response(data=data)
    except Exception as exc:
        LOGGER.exception("Ask failed: %s", exc)
        return error_response(str(exc))


@app.post("/ground", response_model=GroundResponse, responses=RESPONSES_GROUND)
def ground(request: GroundRequest):
    """Manually trigger grounding for specific phrase indices."""
    try:
        if backend.model is None:
            return error_response("Model disabled (FLMM_WEB_NO_MODEL=1). Please enqueue GROUND via /tasks.", status_code=503)
        entry = _with_session(request.session_id)
        with entry.lock:
            with backend.model_lock:
                indices = request.indices or []
                if indices:
                    handle_ground(entry.state, indices)
                else:
                    if entry.state.last_answer is None:
                        raise ValueError("No cached answer. Run ASK first.")
                    phrases = [item.model_dump() for item in (request.phrases or [])]
                    records = perform_ground_custom(entry.state, phrases)
                    if not records:
                        raise ValueError(
                            "GROUND produced no records; check payload or previous ASK."
                        )
            data = ground_payload(backend, entry.state)
            data["history"] = snapshot_history(entry.state)
        return build_response(data=data)
    except Exception as exc:
        LOGGER.exception("Ground failed: %s", exc)
        return error_response(str(exc))


@app.post("/tasks", response_model=TaskEnqueueResponse, responses=RESPONSES_TASK_ENQUEUE)
def enqueue_task_api(request: TaskEnqueueRequest):
    """Enqueue a task for asynchronous processing."""
    try:
        entry = _with_session(request.session_id)
        payload = dict(request.payload or {})
        # Provide default turn index when absent
        turn_idx = request.turn_idx
        if turn_idx is None:
            if request.type == "ASK":
                turn_idx = history_turns(entry.state)
            else:
                turn_idx = entry.state.turn_idx
        # Ensure worker can load the current image for ASK tasks
        if request.type == "ASK" and not payload.get("image_path"):
            if entry.state.current_image_path:
                payload["image_path"] = str(entry.state.current_image_path)
        task_id = enqueue_task(
            backend.task_conn,
            request.type,
            request.session_id,
            payload,
            turn_idx=turn_idx,
            turn_uid=request.turn_uid,
        )
        return build_response(data={"task_id": task_id})
    except Exception as exc:
        LOGGER.exception("Enqueue task failed: %s", exc)
        return error_response(str(exc))


@app.get("/tasks/{task_id}", response_model=TaskPollResponse, responses=RESPONSES_TASK_POLL)
def poll_task(task_id: int):
    """Poll task status and output."""
    try:
        task = get_task(backend.task_conn, task_id)
        if not task:
            return error_response("Task not found.", status_code=404)
        return build_response(data={"task": task})
    except Exception as exc:
        LOGGER.exception("Poll task failed: %s", exc)
        return error_response(str(exc))


@app.post("/clear", response_model=ClearResponse, responses=RESPONSES_CLEAR)
def clear(request: ClearRequest):
    """Clear the conversation history for a session."""
    try:
        entry = _with_session(request.session_id)
        with entry.lock:
            clear_history(entry.state)
        return build_response(
            data={
                "history": snapshot_history(entry.state),
                "history_turns": history_turns(entry.state),
            }
        )
    except Exception as exc:
        LOGGER.exception("Clear failed: %s", exc)
        return error_response(str(exc))


# if __name__ == "__main__":
#     # import uvicorn

#     # uvicorn.run(
#     #     "scripts.demo.web.backend.main:app",
#     #     host="0.0.0.0",
#     #     port=int(os.getenv("PORT", "9000")),
#     #     reload=False,
#     # )
