"""SQLite-backed task worker that runs the real model pipelines."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.interact import (  # noqa: E402
    SessionState,
    handle_ground,
    handle_load,
    history_turns,
    load_model,
    pipeline_default_ask,
)
from scripts.demo.web.backend.task_queue.paths import write_json  # noqa: E402
from .db import connect, init_db
from .queue import claim_next_task, mark_done, mark_failed

LOGGER = logging.getLogger("flmm.task_worker")
DEFAULT_SLEEP = 0.5


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


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


def build_args_from_env(cli_parse_args):
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["flmm-task-worker"]
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


class WorkerRuntime:
    """Holds shared model + per-session state for the worker loop."""

    def __init__(self):
        from scripts.demo.interact import parse_args as cli_parse_args  # noqa: E402

        self.args = build_args_from_env(cli_parse_args)
        cfg_path = Path(self.args.config)
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / cfg_path).resolve()
        LOGGER.info("Loading config: %s", cfg_path)
        self.cfg = Config.fromfile(cfg_path)
        LOGGER.info("Loading model for worker...")
        self.model = load_model(self.cfg, self.args)
        self.results_dir = Path(self.args.results_dir).expanduser().resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, SessionState] = {}

    def _rel(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.results_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _session(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session:
            return session
        session = SessionState(
            model=self.model,
            args=self.args,
            result_root=self.results_dir,
            session_id=session_id,
        )
        self.sessions[session_id] = session
        return session

    def _prepare_turn(self, session: SessionState, turn_idx: Optional[int]) -> int:
        resolved = history_turns(session) if turn_idx is None else int(turn_idx)
        session.turn_idx = resolved
        session.ground_id = 0
        session.attn_counters = {"i2t": 0, "t2i": 0}
        session.session_paths.turn_dir(resolved).mkdir(parents=True, exist_ok=True)
        return resolved

    def _serialize_phrases(self, session: SessionState) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, cand in enumerate(session.phrases):
            payload.append(
                {
                    "index": idx,
                    "text": cand.text,
                    "char_span": cand.char_span,
                    "token_span": cand.token_span,
                }
            )
        return payload

    def _serialize_ground(self, session: SessionState) -> Dict[str, Any]:
        records = []
        for idx, record in enumerate(session.last_records):
            records.append(
                {
                    "index": idx,
                    "phrase": record.phrase_text,
                    "overlay_path": self._rel(record.overlay_path),
                    "mask_path": self._rel(record.mask_path),
                    "roi_path": self._rel(record.roi_path),
                    "char_span": record.char_span,
                    "token_span": record.token_span,
                    "bbox": record.bbox,
                }
            )
        ground_idx = max(session.ground_id - 1, 0)
        ground_dir = session.session_paths.ground_dir(session.turn_idx, ground_idx)
        turn_dir = session.session_paths.turn_dir(session.turn_idx)
        return {
            "records": records,
            "turn_dir": self._rel(turn_dir),
            "ground_dir": self._rel(ground_dir),
        }

    def handle_ask(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._session(task["session_id"])
        turn_hint = payload.get("turn_idx") or task.get("turn_idx")
        self._prepare_turn(session, turn_hint)
        question = payload.get("question")
        if not question:
            raise ValueError("ASK payload requires 'question'.")
        image_path = payload.get("image_path")
        if image_path:
            handle_load(session, image_path)
        if session.current_image is None:
            raise ValueError("ASK requires image_path or a previously loaded image.")
        auto_topk = int(payload.get("auto_topk", 1))
        reset_history = bool(payload.get("reset_history", False))
        result = pipeline_default_ask(
            session,
            question,
            reset_history=reset_history,
            auto_topk=auto_topk,
            turn_idx=session.turn_idx,
        )
        return {
            "type": "ASK",
            "turn_idx": session.turn_idx,
            "history_turns": history_turns(session),
            "answer": result.get("answer"),
            "phrases": self._serialize_phrases(session),
            "ground": self._serialize_ground(session),
        }

    def handle_ground(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._session(task["session_id"])
        turn_hint = payload.get("turn_idx") or task.get("turn_idx") or session.turn_idx
        session.turn_idx = int(turn_hint)
        indices = payload.get("indices") or payload.get("phrase_indices") or []
        if not isinstance(indices, list) or not indices:
            raise ValueError("GROUND payload requires 'indices' list.")
        records = handle_ground(session, [int(i) for i in indices])
        if not records:
            raise ValueError("GROUND produced no records; check payload or previous ASK.")
        return {
            "type": "GROUND",
            "turn_idx": session.turn_idx,
            "history_turns": history_turns(session),
            "ground": self._serialize_ground(session),
        }

    def handle_attention(self, task: Dict[str, Any], payload: Dict[str, Any], kind: str) -> Dict[str, Any]:
        session = self._session(task["session_id"])
        turn_hint = payload.get("turn_idx") or task.get("turn_idx") or history_turns(session)
        session.turn_idx = int(turn_hint)
        attn_id = session.attn_counters.get(kind, 0)
        session.attn_counters[kind] = attn_id + 1
        attn_dir = session.session_paths.attn_dir(session.turn_idx, kind, attn_id)
        attn_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "task_id": task["id"],
            "session_id": session.session_id,
            "turn_idx": session.turn_idx,
            "attn_id": attn_id,
            "type": task["type"],
            "input": payload,
            "note": "placeholder attention artifact; plug in real tensor export later",
        }
        write_json(attn_dir / "config.json", config)
        # placeholder artifact
        (attn_dir / "attn.npz").write_bytes(b"")
        return {
            "type": task["type"],
            "turn_idx": session.turn_idx,
            "attn_id": attn_id,
            "attn_dir": self._rel(attn_dir),
            "config": "config.json",
        }

    def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        payload = task.get("input_json") or {}
        ttype = task.get("type")
        if ttype == "ASK":
            return self.handle_ask(task, payload)
        if ttype == "GROUND":
            return self.handle_ground(task, payload)
        if ttype == "ATTN_I2T":
            return self.handle_attention(task, payload, kind="i2t")
        if ttype == "ATTN_T2I":
            return self.handle_attention(task, payload, kind="t2i")
        raise ValueError(f"Unsupported task type: {ttype}")


def run_worker(db_path: Path, sleep_seconds: float = DEFAULT_SLEEP) -> None:
    runtime = WorkerRuntime()
    conn = connect(db_path)
    init_db(conn)
    worker_id = os.getenv("FLMM_WEB_WORKER_ID", str(uuid.uuid4()))
    LOGGER.info("Worker started id=%s db=%s", worker_id, db_path)

    stop = False

    def _handle_sigterm(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    while not stop:
        task = claim_next_task(conn, worker_id)
        if not task:
            time.sleep(sleep_seconds)
            continue
        try:
            output = runtime.handle_task(task)
            mark_done(conn, task["id"], output)
            LOGGER.info("task %s done", task["id"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("task %s failed: %s", task["id"], exc)
            mark_failed(conn, task["id"], str(exc))

    LOGGER.info("Worker exiting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task queue worker")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite task DB (defaults to FLMM_WEB_TASK_DB or results/task_queue.db)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Sleep seconds between polls when idle.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    runtime_probe_results = env_path("FLMM_WEB_RESULTS_DIR", Path.cwd() / "results")
    db_default = runtime_probe_results / "task_queue.db"
    db_path = args.db if args.db else env_path("FLMM_WEB_TASK_DB", db_default)
    run_worker(db_path=db_path, sleep_seconds=args.sleep)


if __name__ == "__main__":  # pragma: no cover
    main()
