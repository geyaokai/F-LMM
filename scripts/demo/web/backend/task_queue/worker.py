"""SQLite-backed task worker that runs the real model pipelines."""

from __future__ import annotations

import argparse
import logging
import numpy as np
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
    build_offsets,
    handle_ground,
    handle_load,
    history_turns,
    load_model,
    perform_ground_custom,
    pipeline_default_ask,
    serialize_phrase_candidate,
)
from scripts.demo.token_to_region import (  # noqa: E402
    build_token_to_region_heatmap,
    parse_token_span,
    render_overlay,
    token_text_from_ids,
)
from scripts.demo.region_to_token import (  # noqa: E402
    build_fallback_phrase_records,
    build_phrase_records,
    build_region_mask,
    build_region_to_token_scores,
    build_token_records,
    dump_json as dump_region_to_token_json,
    rank_records,
    render_region_artifacts,
)
from scripts.demo.web.backend.prompt_overrides import (
    apply_prompt_overrides,
)  # noqa: E402
from scripts.demo.web.backend.task_queue.paths import (  # noqa: E402
    ATTN_KIND_REGION_TO_TOKEN,
    ATTN_KIND_TOKEN_TO_REGION,
    TASK_TYPE_REGION_TO_TOKEN,
    TASK_TYPE_TOKEN_TO_REGION,
    canonical_attn_kind,
    canonical_task_type,
    write_json,
)
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
    args.max_image_side = env_int("FLMM_WEB_MAX_IMAGE_SIDE", args.max_image_side)
    args.max_history_turns = env_int(
        "FLMM_WEB_MAX_HISTORY_TURNS", args.max_history_turns
    )
    args.inspect_prompt = os.getenv("FLMM_WEB_INSPECT_PROMPT", args.inspect_prompt)
    args.extra_prompt = os.getenv("FLMM_WEB_EXTRA_PROMPT", args.extra_prompt or "")
    args.prompt_file = os.getenv("FLMM_WEB_PROMPT_FILE")
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
        results_dir = Path(self.args.results_dir).expanduser().resolve()
        LOGGER.info(
            "Worker runtime config=%s checkpoint=%s prompt_file=%s device=%s "
            "device_map=%s device_max_memory=%s results_dir=%s no_sam=%s",
            cfg_path,
            self.args.checkpoint,
            self.args.prompt_file,
            self.args.device,
            self.args.device_map,
            self.args.device_max_memory,
            results_dir,
            self.args.no_sam,
        )
        LOGGER.info("Loading config: %s", cfg_path)
        self.cfg = Config.fromfile(cfg_path)
        LOGGER.info("Loading model for worker...")
        apply_prompt_overrides(self.cfg, self.args, self.args.prompt_file)
        self.model = load_model(self.cfg, self.args)
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, SessionState] = {}
        self.results_mount = os.getenv("FLMM_WEB_RESULTS_MOUNT", "/results").rstrip("/")
        if self.results_mount and not self.results_mount.startswith("/"):
            self.results_mount = "/" + self.results_mount

    def _rel(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            rel = path.resolve().relative_to(self.results_dir).as_posix().lstrip("/")
            return f"{self.results_mount}/{rel}" if rel else self.results_mount
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
        session.attn_counters = {
            ATTN_KIND_TOKEN_TO_REGION: 0,
            ATTN_KIND_REGION_TO_TOKEN: 0,
        }
        session.session_paths.turn_dir(resolved).mkdir(parents=True, exist_ok=True)
        return resolved

    def _serialize_phrases(self, session: SessionState) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, cand in enumerate(session.phrases):
            payload.append(serialize_phrase_candidate(cand, index=idx))
        return payload

    def _normalize_bbox(self, bbox: Any) -> Optional[List[float]]:
        if not bbox:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return None
        # Heuristic: if bbox appears to be (x, y, w, h) with w/h smaller than x/y,
        # convert to (x1, y1, x2, y2).
        if x2 < x1 or y2 < y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)
        return [x1, y1, x2, y2]

    def _align_char_span(self, answer: str, phrase: str, span: Any) -> List[int]:
        try:
            start, end = span
        except Exception:
            start, end = 0, 0
        if not answer or not phrase:
            return [int(start), int(end)]
        snippet = answer[start:end]
        if snippet and phrase.lower() in snippet.lower():
            return [int(start), int(end)]
        lower_answer = answer.lower()
        lower_phrase = phrase.lower()
        idx = lower_answer.find(lower_phrase)
        if idx != -1:
            return [int(idx), int(idx + len(phrase))]
        return [int(start), int(end)]

    def _serialize_ground(
        self,
        session: SessionState,
        answer_text: Optional[str] = None,
        text_source: str = "answer",
    ) -> Dict[str, Any]:
        records = []
        answer_text = answer_text or ""
        for idx, record in enumerate(session.last_records):
            phrase = record.phrase_text or ""
            records.append(
                {
                    "index": idx,
                    "phrase": phrase,
                    "overlay_path": self._rel(record.overlay_path),
                    "mask_path": self._rel(record.mask_path),
                    "roi_path": self._rel(record.roi_path),
                    "char_span": self._align_char_span(
                        answer_text, phrase, record.char_span
                    ),
                    "token_span": record.token_span,
                    "bbox": self._normalize_bbox(record.bbox),
                }
            )
        ground_idx = max(session.ground_id - 1, 0)
        ground_dir = session.session_paths.ground_dir(session.turn_idx, ground_idx)
        turn_dir = session.session_paths.turn_dir(session.turn_idx)
        return {
            "records": records,
            "turn_dir": self._rel(turn_dir),
            "ground_dir": self._rel(ground_dir),
            "text_source": text_source,
        }

    def handle_ask(
        self, task: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        enable_roi = payload.get("enable_roi", True)
        if isinstance(enable_roi, str):
            enable_roi = enable_roi.strip().lower() not in ("0", "false", "no", "off")
        else:
            enable_roi = bool(enable_roi)
        result = pipeline_default_ask(
            session,
            question,
            reset_history=reset_history,
            auto_topk=auto_topk,
            turn_idx=session.turn_idx,
            enable_roi=enable_roi,
        )
        raw_answer = (
            result.get("original_answer")
            or result.get("raw_answer")
            or result.get("answer")
            or ""
        )
        final_answer = result.get("answer") or raw_answer
        return {
            "type": "ASK",
            "turn_idx": session.turn_idx,
            "history": [
                {"role": m["role"], "text": m["text"]} for m in session.history
            ],
            "history_turns": history_turns(session),
            "raw_answer": raw_answer,
            "answer": final_answer,
            "roi_answer": result.get("roi_answer"),
            "phrases": self._serialize_phrases(session),
            "verification": self._serialize_ground(
                session, answer_text=raw_answer, text_source="raw_answer"
            ),
            # backward-compatible alias
            "ground": self._serialize_ground(
                session, answer_text=raw_answer, text_source="raw_answer"
            ),
        }

    def handle_ground(
        self, task: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        session = self._session(task["session_id"])
        turn_hint = payload.get("turn_idx") or task.get("turn_idx") or session.turn_idx
        session.turn_idx = int(turn_hint)
        indices = payload.get("indices") or payload.get("phrase_indices") or []
        custom_items = payload.get("phrases") or payload.get("custom_phrases")
        if isinstance(indices, list) and indices:
            records = handle_ground(session, [int(i) for i in indices])
        elif custom_items is not None:
            if session.last_answer is None:
                raise ValueError("No cached answer. Run ASK first.")
            if not isinstance(custom_items, list) or not custom_items:
                raise ValueError(
                    "GROUND custom payload requires non-empty 'phrases' list."
                )
            records = perform_ground_custom(session, custom_items)
        else:
            raise ValueError("GROUND payload requires 'indices' or 'phrases'.")

        if not records:
            raise ValueError(
                "GROUND produced no records; check payload or previous ASK."
            )

        ground_data = self._serialize_ground(session)
        return {
            "type": "GROUND",
            "turn_idx": session.turn_idx,
            "history": [
                {"role": m["role"], "text": m["text"]} for m in session.history
            ],
            "history_turns": history_turns(session),
            "verification": ground_data,
            "records": ground_data.get("records"),
            # backward-compatible alias
            "ground": ground_data,
        }

    def handle_attention(
        self, task: Dict[str, Any], payload: Dict[str, Any], kind: str
    ) -> Dict[str, Any]:
        kind = canonical_attn_kind(kind)
        task_type = canonical_task_type(task.get("type"))
        kind_label = kind.upper()
        session = self._session(task["session_id"])
        turn_hint = (
            payload.get("turn_idx") or task.get("turn_idx") or history_turns(session)
        )
        session.turn_idx = int(turn_hint)
        attn_id = session.attn_counters.get(kind, 0)
        session.attn_counters[kind] = attn_id + 1
        attn_dir = session.session_paths.attn_dir(session.turn_idx, kind, attn_id)
        attn_dir.mkdir(parents=True, exist_ok=True)
        image_path = payload.get("image_path")
        if image_path:
            handle_load(session, image_path)
        if session.current_image is None:
            raise ValueError(
                f"{kind_label} requires image_path or a previously loaded image."
            )
        prompt = payload.get("prompt") or payload.get("question")
        if session.last_answer is None:
            if not prompt:
                raise ValueError(
                    f"{kind_label} requires prompt when no cached answer exists."
                )
            session.last_answer = session.model.answer(
                image=session.current_image,
                question=prompt,
                history=session.history,
                max_new_tokens=session.args.max_new_tokens,
            )
        answer_cache = session.last_answer
        attention_maps = answer_cache.get("attention_maps")
        if attention_maps is None:
            raise ValueError("answer cache has no attention_maps.")
        layer = payload.get("layer", "mean")
        head = payload.get("head", "mean")
        reduction = payload.get("reduction", "mean")
        if kind == ATTN_KIND_REGION_TO_TOKEN:
            bbox = payload.get("bbox")
            bbox_source = "payload"
            source_phrase = None
            record_index = payload.get("record_index")
            if bbox is None and record_index is not None:
                idx = int(record_index)
                if idx < 0 or idx >= len(session.last_records):
                    raise ValueError(
                        f"record_index out of range: {idx} (records={len(session.last_records)})"
                    )
                record = session.last_records[idx]
                if not record.bbox:
                    raise ValueError(f"ground record {idx} has no bbox.")
                bbox = list(record.bbox)
                bbox_source = "record_index"
                source_phrase = record.phrase_text
            if bbox is None:
                raise ValueError(
                    "REGION_TO_TOKEN requires 'bbox' or 'record_index'."
                )
            bbox_format = payload.get("bbox_format", "xyxy")
            raw_scores, norm_scores, per_layer_scores, meta = build_region_to_token_scores(
                attention_maps=attention_maps,
                bbox=bbox,
                image_size=session.current_image.size,
                layer=layer,
                head=head,
                reduction=reduction,
                bbox_format=bbox_format,
            )
            answer_text = answer_cache.get("output_text") or ""
            offsets = session.token_offsets
            if offsets is None and answer_text:
                offsets = build_offsets(session.model.tokenizer, answer_text)
                session.token_offsets = offsets
            if offsets is None:
                offsets = []
            all_tokens = build_token_records(
                session.model.tokenizer,
                answer_cache.get("output_ids"),
                offsets,
                raw_scores,
                norm_scores,
            )
            topk = int(payload.get("topk", 8))
            top_tokens = rank_records(all_tokens, topk=topk)
            phrase_records = build_phrase_records(norm_scores, session.phrases)
            top_phrases = rank_records(phrase_records, topk=topk, skip_blank_text=True)
            if not top_phrases:
                top_phrases = build_fallback_phrase_records(top_tokens)
            grid_shape = meta["grid_shape"]
            region_mask, _ = build_region_mask(
                meta["bbox"],
                session.current_image.size,
                (int(grid_shape["height"]), int(grid_shape["width"])),
            )
            bbox_overlay, region_overlay, region_heatmap = render_region_artifacts(
                session.current_image,
                meta["bbox"],
                region_mask,
            )
            bbox_overlay_path = attn_dir / "bbox_overlay.png"
            region_overlay_path = attn_dir / "region_overlay.png"
            region_heatmap_path = attn_dir / "region_heatmap.png"
            ranking_path = attn_dir / "ranking.json"
            bbox_overlay.save(bbox_overlay_path)
            region_overlay.save(region_overlay_path)
            region_heatmap.save(region_heatmap_path)
            layer_summary = []
            for idx, layer_scores in enumerate(per_layer_scores):
                token_idx = int(np.argmax(layer_scores)) if layer_scores.size else -1
                layer_id = idx if meta.get("layer") == "mean" else int(meta["layer"])
                layer_summary.append(
                    {
                        "layer": layer_id,
                        "max_token_index": token_idx,
                        "max_token_score": float(layer_scores[token_idx]) if token_idx >= 0 else 0.0,
                    }
                )
            meta["bbox_source"] = bbox_source
            meta["source_phrase"] = source_phrase
            meta["topk"] = topk
            meta["layer_summary"] = layer_summary
            ranking_payload = {
                "answer_text": answer_text,
                "bbox": meta["bbox"],
                "bbox_source": bbox_source,
                "source_phrase": source_phrase,
                "top_tokens": top_tokens,
                "top_phrases": top_phrases,
                "token_scores": all_tokens,
            }
            write_json(attn_dir / "meta.json", meta)
            dump_region_to_token_json(ranking_path, ranking_payload)
            config = {
                "task_id": task["id"],
                "session_id": session.session_id,
                "turn_idx": session.turn_idx,
                "attn_id": attn_id,
                "type": task_type,
                "input": payload,
                "meta": meta,
            }
            write_json(attn_dir / "config.json", config)
            return {
                "type": task_type,
                "turn_idx": session.turn_idx,
                "attn_id": attn_id,
                "attn_dir": self._rel(attn_dir),
                "bbox_overlay_png_url": self._rel(bbox_overlay_path),
                "region_overlay_png_url": self._rel(region_overlay_path),
                "region_heatmap_png_url": self._rel(region_heatmap_path),
                "ranking_json_url": self._rel(ranking_path),
                "meta": meta,
                "top_tokens": top_tokens,
                "top_phrases": top_phrases,
            }
        token_span_input = payload.get("token_span")
        if token_span_input is None:
            raise ValueError("TOKEN_TO_REGION requires token_span.")
        heatmap, meta = build_token_to_region_heatmap(
            attention_maps=attention_maps,
            token_span=token_span_input,
            layer=layer,
            head=head,
            reduction=reduction,
        )
        start, end = parse_token_span(token_span_input)
        token_text = token_text_from_ids(
            session.model.tokenizer,
            answer_cache.get("output_ids"),
            start,
            end,
        )
        if token_text:
            meta["token_text"] = token_text

        overlay_img, heatmap_img = render_overlay(session.current_image, heatmap)
        overlay_path = attn_dir / "overlay.png"
        heatmap_path = attn_dir / "heatmap.png"
        overlay_img.save(overlay_path)
        heatmap_img.save(heatmap_path)
        write_json(attn_dir / "meta.json", meta)

        config = {
            "task_id": task["id"],
            "session_id": session.session_id,
            "turn_idx": session.turn_idx,
            "attn_id": attn_id,
            "type": task_type,
            "input": payload,
            "meta": meta,
        }
        write_json(attn_dir / "config.json", config)
        return {
            "type": task_type,
            "turn_idx": session.turn_idx,
            "attn_id": attn_id,
            "attn_dir": self._rel(attn_dir),
            "overlay_png_url": self._rel(overlay_path),
            "heatmap_png_url": self._rel(heatmap_path),
            "meta": meta,
        }

    def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        payload = task.get("input_json") or {}
        ttype = canonical_task_type(task.get("type"))
        if ttype == "ASK":
            return self.handle_ask(task, payload)
        if ttype == "GROUND":
            return self.handle_ground(task, payload)
        if ttype == TASK_TYPE_TOKEN_TO_REGION:
            return self.handle_attention(task, payload, kind=ATTN_KIND_TOKEN_TO_REGION)
        if ttype == TASK_TYPE_REGION_TO_TOKEN:
            return self.handle_attention(task, payload, kind=ATTN_KIND_REGION_TO_TOKEN)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    runtime_probe_results = env_path("FLMM_WEB_RESULTS_DIR", Path.cwd() / "results")
    db_default = runtime_probe_results / "task_queue.db"
    db_path = args.db if args.db else env_path("FLMM_WEB_TASK_DB", db_default)
    run_worker(db_path=db_path, sleep_seconds=args.sleep)


if __name__ == "__main__":  # pragma: no cover
    main()
