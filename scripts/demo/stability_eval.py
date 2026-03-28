#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.interact import (  # noqa: E402
    GroundRecord,
    SessionState,
    handle_load,
    load_model,
    parse_args as interact_parse_args,
    pipeline_default_ask,
    serialize_phrase_candidate,
)
from scripts.demo.web.backend.prompt_overrides import apply_prompt_overrides  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed stability benchmark over ask -> ground -> ROI flows."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON file containing a list of samples or an object with key 'samples'.",
    )
    parser.add_argument(
        "--report-dir",
        default="scripts/demo/results/stability",
        help="Directory for aggregated reports and per-sample artifacts.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N samples when > 0.",
    )
    parser.add_argument(
        "--config",
        default="configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py",
        help="Model config path.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument(
        "--device-map",
        default="none",
        help='Device map passed to the Qwen base model. Default is "none" for stable eval; use "auto" to shard.',
    )
    parser.add_argument(
        "--device-max-memory",
        default=None,
        help='Optional per-device memory limits, e.g. "0:20GiB,1:20GiB".',
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--phrase-max-tokens", type=int, default=128)
    parser.add_argument("--max-phrases", type=int, default=6)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--auto-topk", type=int, default=1)
    parser.add_argument(
        "--enable-roi",
        action="store_true",
        help="Enable ROI re-answer by default when a sample does not override it.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional prompt override JSON/plain text file.",
    )
    parser.add_argument("--extra-prompt", default="", help="Append to every question.")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM refinement.")
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=0.3,
        help="Threshold for auto-tagging bbox_low_iou.",
    )
    return parser


def build_interact_defaults():
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["stability-eval"]
        return interact_parse_args()
    finally:
        sys.argv = argv_backup


def resolve_repo_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (REPO_ROOT / raw).resolve()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict):
        samples = payload.get("samples", [])
    else:
        raise ValueError("Manifest must be a list or an object with key 'samples'.")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest contains no samples.")
    normalized: List[Dict[str, Any]] = []
    for item in samples:
        if not isinstance(item, dict):
            raise ValueError("Each sample in the manifest must be an object.")
        normalized.append(item)
    return normalized


def make_run_dir(base_dir: Path, run_name: Optional[str]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / (run_name or stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_session_id(index: int, sample: Dict[str, Any]) -> str:
    raw = str(sample.get("id") or sample.get("sample_id") or f"sample_{index:04d}")
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("._")
    if not slug:
        slug = f"sample_{index:04d}"
    return f"{index:04d}_{slug[:80]}"


def relpath(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def serialize_phrases(session: SessionState) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, phrase in enumerate(session.phrases):
        payload.append(serialize_phrase_candidate(phrase, index=idx))
    return payload


def serialize_ground_records(
    records: List[GroundRecord], artifact_root: Path
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        payload.append(
            {
                "index": idx,
                "phrase": record.phrase_text,
                "char_span": [int(record.char_span[0]), int(record.char_span[1])],
                "token_span": [int(record.token_span[0]), int(record.token_span[1])],
                "bbox": [int(v) for v in record.bbox] if record.bbox else None,
                "overlay_path": relpath(record.overlay_path, artifact_root),
                "mask_path": relpath(record.mask_path, artifact_root),
                "roi_path": relpath(record.roi_path, artifact_root),
            }
        )
    return payload


def normalize_history(raw_history: Any) -> List[Dict[str, str]]:
    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        raise ValueError("history must be a list of {role, text}.")
    history: List[Dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            raise ValueError("history items must be objects.")
        role = str(item.get("role") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        if not role or not text:
            continue
        history.append({"role": role, "text": text})
    return history


def normalize_bbox(raw_bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if raw_bbox is None:
        return None
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("expected_bbox must be [x1, y1, x2, y2].")
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("expected_bbox must satisfy x2>x1 and y2>y1.")
    return (x1, y1, x2, y2)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_first_bbox(records: List[GroundRecord]) -> Optional[Tuple[float, float, float, float]]:
    for record in records:
        if record.bbox and len(record.bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in record.bbox]
            return (x1, y1, x2, y2)
    return None


def compute_iou(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


def normalize_expectations(sample: Dict[str, Any]) -> Dict[str, Any]:
    expectations = sample.get("expectations")
    if expectations is None:
        expectations = {}
    if not isinstance(expectations, dict):
        raise ValueError("expectations must be an object when provided.")
    answer_contains = expectations.get(
        "answer_contains", sample.get("answer_contains", [])
    )
    answer_not_contains = expectations.get(
        "answer_not_contains", sample.get("answer_not_contains", [])
    )
    if isinstance(answer_contains, str):
        answer_contains = [answer_contains]
    if isinstance(answer_not_contains, str):
        answer_not_contains = [answer_not_contains]
    return {
        "answer_contains": [str(x).strip() for x in answer_contains if str(x).strip()],
        "answer_not_contains": [
            str(x).strip() for x in answer_not_contains if str(x).strip()
        ],
        "expected_bbox": normalize_bbox(
            expectations.get("expected_bbox", sample.get("expected_bbox"))
        ),
    }


def evaluate_checks(
    answer: str,
    records: List[GroundRecord],
    expectations: Dict[str, Any],
) -> Dict[str, Any]:
    lower_answer = (answer or "").lower()
    contains_checks = []
    for needle in expectations["answer_contains"]:
        contains_checks.append(
            {
                "text": needle,
                "passed": needle.lower() in lower_answer,
            }
        )
    forbidden_checks = []
    for needle in expectations["answer_not_contains"]:
        forbidden_checks.append(
            {
                "text": needle,
                "passed": needle.lower() not in lower_answer,
            }
        )
    bbox_iou = None
    predicted_bbox = get_first_bbox(records)
    expected_bbox = expectations.get("expected_bbox")
    if expected_bbox and predicted_bbox:
        bbox_iou = compute_iou(predicted_bbox, expected_bbox)
    return {
        "answer_contains": contains_checks,
        "answer_not_contains": forbidden_checks,
        "expected_bbox": list(expected_bbox) if expected_bbox else None,
        "predicted_bbox": list(predicted_bbox) if predicted_bbox else None,
        "bbox_iou": bbox_iou,
    }


def derive_auto_failure_tags(
    *,
    result: Dict[str, Any],
    records: List[GroundRecord],
    session: SessionState,
    checks: Dict[str, Any],
    enable_roi: bool,
    bbox_iou_threshold: float,
) -> List[str]:
    tags: List[str] = []
    raw_answer = str(result.get("raw_answer") or result.get("original_answer") or "")
    verification = result.get("verification") or {}
    raw_artifacts = verification.get("raw_answer_artifacts") or []
    roi_artifacts = verification.get("roi_answer_artifacts") or []

    if not raw_answer.strip():
        tags.append("answer_empty")
    if raw_artifacts or roi_artifacts:
        tags.append("answer_decoding_artifact")
    if not session.phrases:
        tags.append("phrase_missing")
    if session.phrases and not records:
        tags.append("ground_missing")
    if records and any(record.bbox is None for record in records):
        tags.append("bbox_missing")
    if verification.get("error"):
        if records:
            tags.append("roi_error")
        else:
            tags.append("ground_error")
    if enable_roi and records and not result.get("roi_answer"):
        tags.append("roi_unused")
    if result.get("roi_answer") and raw_answer.strip() and result["roi_answer"].strip() != raw_answer.strip():
        tags.append("roi_answer_changed")
    if verification.get("roi_rejected_reason") == "answer_artifact":
        tags.append("roi_answer_rejected_artifact")

    for item in checks.get("answer_contains", []):
        if not item.get("passed"):
            tags.append("answer_missing_required_text")
            break
    for item in checks.get("answer_not_contains", []):
        if not item.get("passed"):
            tags.append("answer_contains_forbidden_text")
            break

    bbox_iou = checks.get("bbox_iou")
    if bbox_iou is not None and bbox_iou < bbox_iou_threshold:
        tags.append("bbox_low_iou")

    # Normalize and preserve order.
    normalized: List[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
    return normalized


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


REVIEW_ARTIFACT_MARKERS = (
    "addcriterion",
    "matchcondition",
    "guidid",
    "自动生成",
)


def _detect_review_artifacts(text: str) -> List[str]:
    lowered = (text or "").lower()
    hits: List[str] = []
    for marker in REVIEW_ARTIFACT_MARKERS:
        if marker.lower() in lowered:
            hits.append(marker)
    return hits


def _collect_decoding_artifacts(row: Dict[str, Any]) -> Dict[str, List[str]]:
    verification = row.get("verification") or {}
    artifact_map: Dict[str, List[str]] = {
        "raw_answer": list(verification.get("raw_answer_artifacts") or []),
        "roi_answer": list(verification.get("roi_answer_artifacts") or []),
    }
    field_values = {
        "raw_answer": row.get("raw_answer", ""),
        "final_answer": row.get("answer", ""),
        "roi_answer": row.get("roi_answer", ""),
    }
    for key, text in field_values.items():
        hits = _detect_review_artifacts(str(text or ""))
        merged = artifact_map.get(key, [])
        for marker in hits:
            if marker not in merged:
                merged.append(marker)
        if merged:
            artifact_map[key] = merged
    return {key: value for key, value in artifact_map.items() if value}


def _extract_failed_terms(checks: Dict[str, Any], key: str) -> List[str]:
    failed: List[str] = []
    for item in checks.get(key, []) or []:
        if not item.get("passed"):
            text = str(item.get("text") or "").strip()
            if text:
                failed.append(text)
    return failed


def _extract_all_terms(checks: Dict[str, Any], key: str) -> List[str]:
    values: List[str] = []
    for item in checks.get(key, []) or []:
        text = str(item.get("text") or "").strip()
        if text:
            values.append(text)
    return values


def _build_review_focus(row: Dict[str, Any]) -> List[str]:
    tags = set(row.get("auto_failure_tags") or [])
    focus: List[str] = []
    if row.get("status") != "ok":
        focus.append("pipeline_error")
    artifact_map = _collect_decoding_artifacts(row)
    if artifact_map or "answer_decoding_artifact" in tags:
        focus.append("decoding_artifact")
    if "bbox_low_iou" in tags:
        focus.append("grounding_tightness")
    if "phrase_missing" in tags or "ground_missing" in tags or "bbox_missing" in tags:
        focus.append("grounding_failure")
    if "answer_missing_required_text" in tags or "answer_contains_forbidden_text" in tags:
        focus.append("answer_semantics")
    if "roi_answer_changed" in tags or "roi_answer_rejected_artifact" in tags:
        focus.append("roi_effect")
    if not focus:
        focus.append("spot_check")
    return focus


def _collect_artifact_refs(row: Dict[str, Any]) -> Dict[str, Any]:
    overlay_paths: List[str] = []
    mask_paths: List[str] = []
    roi_paths: List[str] = []
    for record in row.get("ground_records", []) or []:
        overlay = record.get("overlay_path")
        mask = record.get("mask_path")
        roi = record.get("roi_path")
        if overlay:
            overlay_paths.append(str(overlay))
        if mask:
            mask_paths.append(str(mask))
        if roi:
            roi_paths.append(str(roi))
    return {
        "session_dir": row.get("session_dir"),
        "loaded_image_path": row.get("loaded_image_path"),
        "overlay_paths": overlay_paths,
        "mask_paths": mask_paths,
        "roi_paths": roi_paths,
    }


def _build_review_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    checks = row.get("checks") or {}
    artifact_map = _collect_decoding_artifacts(row)
    verification = row.get("verification") or {}
    roi_answer_accepted = verification.get("roi_answer_accepted")
    roi_rejected_reason = verification.get("roi_rejected_reason")
    if roi_answer_accepted is None:
        roi_answer = str(row.get("roi_answer") or "").strip()
        final_answer = str(row.get("answer") or "").strip()
        if roi_answer:
            roi_answer_accepted = roi_answer == final_answer
            if (
                roi_answer_accepted is False
                and artifact_map.get("roi_answer")
                and not artifact_map.get("raw_answer")
            ):
                roi_rejected_reason = "answer_artifact_inferred"
    payload = {
        "schema_version": "review_v2",
        "sample_id": row["sample_id"],
        "status": row.get("status"),
        "question": row.get("question", ""),
        "notes": row.get("notes", ""),
        "model_outputs": {
            "raw_answer": row.get("raw_answer", ""),
            "final_answer": row.get("answer", ""),
            "roi_answer": row.get("roi_answer", ""),
        },
        "auto_review": {
            "tags": row.get("auto_failure_tags", []),
            "review_focus": _build_review_focus(row),
            "required_text": _extract_all_terms(checks, "answer_contains"),
            "forbidden_text": _extract_all_terms(checks, "answer_not_contains"),
            "failed_required_text": _extract_failed_terms(checks, "answer_contains"),
            "hit_forbidden_text": _extract_failed_terms(checks, "answer_not_contains"),
            "bbox_iou": checks.get("bbox_iou"),
            "expected_bbox": checks.get("expected_bbox"),
            "predicted_bbox": checks.get("predicted_bbox"),
            "decoding_artifacts": artifact_map,
            "roi_answer_accepted": roi_answer_accepted,
            "roi_rejected_reason": roi_rejected_reason,
        },
        "artifact_refs": _collect_artifact_refs(row),
        "manual_failure_tags": row.get("manual_failure_tags", []),
        "manual_review": {
            "grounding_judgment": "",
            "answer_judgment": "",
            "roi_judgment": "",
            "primary_issue": "",
            "confirmed_auto_tags": [],
            "extra_manual_tags": [],
            "comment": "",
        },
    }
    return payload


def write_review_template(path: Path, sample_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in sample_rows:
        payload = _build_review_payload(row)
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    cli_args = build_parser().parse_args()
    manifest_path = resolve_repo_path(cli_args.manifest)
    samples = load_manifest(manifest_path)
    if cli_args.limit > 0:
        samples = samples[: cli_args.limit]

    report_dir = make_run_dir(resolve_repo_path(cli_args.report_dir), cli_args.run_name)
    artifact_root = report_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_args = build_interact_defaults()
    model_args.config = cli_args.config
    model_args.checkpoint = cli_args.checkpoint
    if model_args.checkpoint:
        ckpt_path = Path(model_args.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (REPO_ROOT / ckpt_path).resolve()
        model_args.checkpoint = str(ckpt_path)
    model_args.device = cli_args.device
    model_args.device_map = cli_args.device_map
    model_args.device_max_memory = cli_args.device_max_memory
    model_args.max_new_tokens = cli_args.max_new_tokens
    model_args.phrase_max_tokens = cli_args.phrase_max_tokens
    model_args.max_phrases = cli_args.max_phrases
    model_args.max_history_turns = cli_args.max_history_turns
    model_args.extra_prompt = cli_args.extra_prompt or ""
    model_args.no_sam = cli_args.no_sam
    model_args.results_dir = str(artifact_root)
    model_args.image = None

    cfg_path = resolve_repo_path(model_args.config)
    cfg = Config.fromfile(str(cfg_path))
    apply_prompt_overrides(cfg, model_args, cli_args.prompt_file)
    model = load_model(cfg, model_args)

    report_rows: List[Dict[str, Any]] = []
    tag_counts: Counter[str] = Counter()
    success_count = 0

    for index, sample in enumerate(samples):
        sample_id = sanitize_session_id(index, sample)
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "manifest_index": index,
            "source_id": sample.get("id") or sample.get("sample_id"),
            "notes": sample.get("notes", ""),
            "manual_failure_tags": sample.get("manual_failure_tags", []),
        }
        report_rows.append(row)

        try:
            image_path = sample.get("image_path")
            question = str(sample.get("question") or "").strip()
            if not image_path:
                raise ValueError("sample.image_path is required.")
            if not question:
                raise ValueError("sample.question is required.")

            history = normalize_history(sample.get("history"))
            expectations = normalize_expectations(sample)
            enable_roi = parse_bool(sample.get("enable_roi", cli_args.enable_roi))
            auto_topk = int(sample.get("auto_topk", cli_args.auto_topk))

            session = SessionState(
                model=model,
                args=model_args,
                result_root=artifact_root,
                session_id=sample_id,
            )

            t0 = time.perf_counter()
            handle_load(session, str(image_path))
            load_elapsed = time.perf_counter() - t0
            session.history = list(history)

            t1 = time.perf_counter()
            result = pipeline_default_ask(
                session,
                question,
                reset_history=False,
                auto_topk=auto_topk,
                enable_roi=enable_roi,
            )
            ask_elapsed = time.perf_counter() - t1
            total_elapsed = time.perf_counter() - t0

            records = list(session.last_records)
            checks = evaluate_checks(
                result.get("answer", "") or result.get("raw_answer", ""),
                records,
                expectations,
            )
            auto_tags = derive_auto_failure_tags(
                result=result,
                records=records,
                session=session,
                checks=checks,
                enable_roi=enable_roi,
                bbox_iou_threshold=cli_args.bbox_iou_threshold,
            )
            tag_counts.update(auto_tags)
            success_count += 1

            row.update(
                {
                    "status": "ok",
                    "image_path": str(image_path),
                    "question": question,
                    "history": history,
                    "enable_roi": enable_roi,
                    "auto_topk": auto_topk,
                    "timing_s": {
                        "load_image": round(load_elapsed, 4),
                        "ask_pipeline": round(ask_elapsed, 4),
                        "total": round(total_elapsed, 4),
                    },
                    "session_dir": relpath(session.session_dir, artifact_root),
                    "loaded_image_path": relpath(session.current_image_path, artifact_root),
                    "raw_answer": result.get("raw_answer") or result.get("original_answer") or "",
                    "answer": result.get("answer") or "",
                    "roi_answer": result.get("roi_answer"),
                    "verification": result.get("verification") or {},
                    "phrases": serialize_phrases(session),
                    "ground_records": serialize_ground_records(records, artifact_root),
                    "expectations": expectations,
                    "checks": checks,
                    "auto_failure_tags": auto_tags,
                    "history_turns_after": len(session.history) // 2,
                }
            )
        except Exception as exc:
            auto_tags = ["sample_error"]
            tag_counts.update(auto_tags)
            row.update(
                {
                    "status": "error",
                    "image_path": sample.get("image_path"),
                    "question": sample.get("question"),
                    "auto_failure_tags": auto_tags,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_path": str(manifest_path),
        "config": str(cfg_path),
        "checkpoint": model_args.checkpoint,
        "num_samples": len(samples),
        "num_success": success_count,
        "num_failed": len(samples) - success_count,
        "artifact_root": str(artifact_root),
        "tag_counts": dict(sorted(tag_counts.items())),
    }
    report = {"summary": summary, "samples": report_rows}

    write_json(report_dir / "report.json", report)
    write_review_template(report_dir / "review_template.jsonl", report_rows)

    print(f"[stability] wrote report: {report_dir / 'report.json'}")
    print(f"[stability] wrote review template: {report_dir / 'review_template.jsonl'}")
    print(f"[stability] artifact root: {artifact_root}")


if __name__ == "__main__":
    main()
