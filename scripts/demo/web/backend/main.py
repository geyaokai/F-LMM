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
    pipeline_default_ask,
)

from scripts.demo.web.backend.task_queue.db import connect as tq_connect, init_db as tq_init_db  # noqa: E402
from scripts.demo.web.backend.task_queue.queue import enqueue_task, get_task  # noqa: E402
from scripts.demo.web.backend.task_queue.paths import SessionPaths  # noqa: E402

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

    @model_validator(mode="after")
    def validate_payload(self):
        if self.mode == "default" and not self.question.strip():
            raise ValueError("Question must not be empty in default mode.")
        return self


class GroundRequest(SessionIdentRequest):
    indices: List[int] = Field(..., description="Phrase indices to ground.", min_length=1)


class ClearRequest(SessionIdentRequest):
    pass


class SessionResetRequest(SessionIdentRequest):
    pass


class TaskEnqueueRequest(APIModel):
    type: Literal['ASK', 'GROUND', 'ATTN_I2T', 'ATTN_T2I'] = Field(..., description='Task type')
    session_id: str = Field(..., description='Session identifier')
    payload: Dict[str, Any] = Field(default_factory=dict, description='Task input payload')
    turn_idx: Optional[int] = Field(None, description='Optional turn index for dedupe')
    turn_uid: Optional[str] = Field(None, description='Optional turn UID')


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
    args.no_sam = env_bool("FLMM_WEB_NO_SAM", args.no_sam)
    args.image = None
    return args


class BackendState:
    """
    Global backend state holding the model, configuration, and session store.
    Initialized once when the FastAPI app starts.
    """

    def __init__(self):
        configure_logging()
        args = build_args_from_env()
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / cfg_path).resolve()
        LOGGER.info("Loading config: %s", cfg_path)
        cfg = Config.fromfile(cfg_path)
        LOGGER.info("Loading model...")
        if args.checkpoint:
            LOGGER.info("Checkpoint override resolved path: %s (exists=%s)", args.checkpoint, Path(args.checkpoint).exists())
        
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


def _clean_answer(text: str) -> str:
    """Remove internal placeholders (e.g., <|image_pad|>) for frontend display."""
    if not text:
        return text
    cleaned = text.replace("<|image_pad|>", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def snapshot_history(session: SessionState) -> List[Dict[str, str]]:
    return [{"role": item.get("role", ""), "text": item.get("text", "")} for item in session.history]


def image_info(session: SessionState) -> Optional[Dict[str, Any]]:
    if session.current_image is None:
        return None
    width, height = session.current_image.size
    path_str = str(session.current_image_path) if session.current_image_path else None
    name = session.current_image_path.name if session.current_image_path else None
    return {"name": name, "path": path_str, "width": width, "height": height}


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
        "image": image_info(session),
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


@app.get("/healthz")
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


@app.post("/session")
@app.post("/session/create")
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


@app.post("/session/reset")
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


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and its associated workspace."""
    try:
        backend.session_store.delete(session_id)
        return build_response(message="Session deleted.")
    except Exception as exc:
        LOGGER.exception("Failed to delete session: %s", exc)
        return error_response(str(exc))


@app.post("/load_image")
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


@app.post("/ask")
def ask(request: AskRequest):
    """
    Handle a user question. 
    Pipeline: Generate answer -> Extract phrases -> Auto-ground -> ROI re-answer.
    """
    try:
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
                )
            
            # Clean the answer for display (remove internal tokens)
            answer = _clean_answer(result.get("answer", ""))
            print("[debug]Answer :", answer)
            
            data.update(
                {
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


@app.post("/ground")
def ground(request: GroundRequest):
    """Manually trigger grounding for specific phrase indices."""
    try:
        entry = _with_session(request.session_id)
        with entry.lock:
            with backend.model_lock:
                handle_ground(entry.state, request.indices)
            data = ground_payload(backend, entry.state)
            data["history"] = snapshot_history(entry.state)
        return build_response(data=data)
    except Exception as exc:
        LOGGER.exception("Ground failed: %s", exc)
        return error_response(str(exc))


@app.post("/tasks")
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


@app.get("/tasks/{task_id}")
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


@app.post("/clear")
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
