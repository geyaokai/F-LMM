#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.interact import (  # noqa: E402
    build_offsets,
    build_phrase_candidates,
    dedupe_phrase_candidates,
    dominant_connected_component,
    extract_phrases_via_model,
    load_image,
    load_model,
    mask_to_box,
    resolve_phrase_to_spans,
    serialize_phrase_candidate,
)
from scripts.demo.token_to_region import (  # noqa: E402
    build_token_to_region_heatmap,
    render_overlay,
)
from scripts.demo.web.backend.prompt_overrides import (  # noqa: E402
    apply_prompt_overrides,
)
from scripts.demo.web.backend.task_queue.paths import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare online generate-cache grounding with teacher-forced "
            "full-forward grounding on the same answer phrase."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--device-max-memory", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--answer-text",
        default=None,
        help=(
            "Optional fixed answer text. When set, build the answer cache from "
            "this text directly instead of calling model.answer()."
        ),
    )
    parser.add_argument(
        "--phrase",
        default=None,
        help="Phrase to compare. If omitted, the first extracted phrase is used.",
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Occurrence index when the phrase appears multiple times in the answer.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--phrase-max-tokens", type=int, default=128)
    parser.add_argument("--max-phrases", type=int, default=6)
    parser.add_argument("--max-history-turns", type=int, default=0)
    parser.add_argument(
        "--history-file",
        default=None,
        help="Optional JSON file containing prior chat history as a list of {role, text}.",
    )
    parser.add_argument("--extra-prompt", default="")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--head", default="mean")
    parser.add_argument("--reduction", default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--out-dir",
        default="tests/results/compare_generate_teacher_forced_ground",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def resolve_run_dir(base_dir: Path, run_name: Optional[str]) -> Path:
    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def build_generation_prompt_text(
    model,
    image: Image.Image,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    conversation = model._build_conversation(image, question, history=history)
    return model.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def build_teacher_forced_cache(
    model,
    image: Image.Image,
    question: str,
    answer_text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return model.build_answer_cache(
        image=image,
        question=question,
        answer_text=answer_text,
        history=history,
    )


def load_history(history_file: Optional[str]) -> Optional[List[Dict[str, str]]]:
    if not history_file:
        return None
    raw = Path(history_file).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("history_file must contain a JSON list.")
    history: List[Dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"history[{idx}] must be an object.")
        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError(f"history[{idx}].role must be 'user' or 'assistant'.")
        if not text:
            raise ValueError(f"history[{idx}].text must be non-empty.")
        history.append({"role": role, "text": text})
    return history


def select_phrase(
    model,
    question: str,
    answer_text: str,
    phrase_override: Optional[str],
    occurrence: int,
    max_tokens: int,
    max_phrases: int,
) -> Tuple[str, Tuple[int, int], Tuple[int, int], List[Dict[str, Any]]]:
    offsets = build_offsets(model.tokenizer, answer_text)
    phrase_texts = extract_phrases_via_model(
        model,
        question,
        answer_text,
        max_tokens=max_tokens,
        limit=max_phrases,
    )
    candidates = dedupe_phrase_candidates(
        build_phrase_candidates(answer_text, phrase_texts, offsets)
    )

    candidate_payload = [
        serialize_phrase_candidate(cand)
        for cand in candidates
    ]

    if phrase_override:
        resolved = resolve_phrase_to_spans(
            answer_text,
            offsets,
            phrase_override,
            occurrence=occurrence,
        )
        if resolved is None:
            raise ValueError(f'Phrase "{phrase_override}" not found in answer text.')
        char_span, token_span = resolved
        return phrase_override, char_span, token_span, candidate_payload

    if not candidates:
        raise ValueError(
            "No phrase candidates extracted from the generated answer. "
            "Pass --phrase explicitly or fix phrase extraction first."
        )
    first = candidates[0]
    return first.text, first.char_span, first.token_span, candidate_payload


def strip_mask_arrays(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("coarse_mask", None)
    cleaned.pop("final_mask", None)
    return cleaned


def save_mask_image(mask: np.ndarray, path: Path) -> None:
    binary = np.asarray(mask, dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((binary * 255).astype(np.uint8), mode="L").save(path)


def blend_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 128, 0),
    alpha: float = 0.65,
) -> Image.Image:
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    overlay = image_np.copy()
    overlay[binary] = (
        overlay[binary] * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    )
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def annotate_image(image: Image.Image, label: str) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 28), color=(255, 255, 255))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 28), fill=(245, 245, 245))
    draw.text((8, 7), label, fill=(20, 20, 20))
    return canvas


def save_comparison_panel(
    image: Image.Image,
    run_dir: Path,
    generated: Dict[str, Any],
    teacher: Dict[str, Any],
) -> None:
    heat_row = [
        annotate_image(image, "image"),
        annotate_image(Image.open(generated["heat_overlay_path"]), "generate heat"),
        annotate_image(Image.open(teacher["heat_overlay_path"]), "teacher heat"),
    ]
    ground_row = [
        annotate_image(image, "image"),
        annotate_image(Image.open(generated["final_overlay_path"]), "generate ground"),
        annotate_image(Image.open(teacher["final_overlay_path"]), "teacher ground"),
    ]

    def concat_row(items: List[Image.Image]) -> Image.Image:
        width = sum(item.width for item in items)
        height = max(item.height for item in items)
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        offset = 0
        for item in items:
            canvas.paste(item, (offset, 0))
            offset += item.width
        return canvas

    rows = [concat_row(heat_row), concat_row(ground_row)]
    total_h = sum(row.height for row in rows)
    total_w = max(row.width for row in rows)
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    offset_y = 0
    for row in rows:
        canvas.paste(row, (0, offset_y))
        offset_y += row.height
    canvas.save(run_dir / "comparison_panel.png")


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def save_mode_artifacts(
    *,
    mode_name: str,
    model,
    image: Image.Image,
    answer_cache: Dict[str, Any],
    token_span: Tuple[int, int],
    char_span: Tuple[int, int],
    phrase_text: str,
    out_dir: Path,
    use_sam: bool,
    layer: Any,
    head: Any,
    reduction: str,
) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    pred_masks, sam_masks = model.ground(
        image=image,
        positive_ids=[token_span],
        hidden_states=answer_cache["hidden_states"],
        attention_maps=answer_cache["attention_maps"],
        meta_data=answer_cache["meta_data"],
        use_sam=use_sam,
    )
    coarse_mask = pred_masks[0].detach().float().cpu().numpy() > 0
    final_source = sam_masks if use_sam else pred_masks
    final_mask = final_source[0].detach().float().cpu().numpy() > 0

    coarse_primary = dominant_connected_component(coarse_mask)
    final_primary = dominant_connected_component(final_mask)
    coarse_bbox = mask_to_box(coarse_primary) or mask_to_box(coarse_mask)
    final_bbox = mask_to_box(final_primary) or mask_to_box(final_mask)

    coarse_mask_path = mode_dir / "coarse_mask.png"
    final_mask_path = mode_dir / "final_mask.png"
    coarse_overlay_path = mode_dir / "coarse_overlay.png"
    final_overlay_path = mode_dir / "final_overlay.png"
    heat_overlay_path = mode_dir / "heat_overlay.png"
    heatmap_path = mode_dir / "heatmap.png"
    roi_path = mode_dir / "roi.png"

    save_mask_image(coarse_mask, coarse_mask_path)
    save_mask_image(final_mask, final_mask_path)
    blend_mask(image, coarse_mask).save(coarse_overlay_path)
    blend_mask(image, final_mask).save(final_overlay_path)

    heatmap, heat_meta = build_token_to_region_heatmap(
        attention_maps=answer_cache["attention_maps"],
        token_span=token_span,
        layer=layer,
        head=head,
        reduction=reduction,
    )
    heat_overlay, heat_rgb = render_overlay(image, heatmap)
    heat_overlay.save(heat_overlay_path)
    heat_rgb.save(heatmap_path)

    if final_bbox is not None:
        image.crop(final_bbox).save(roi_path)

    meta = {
        "mode": mode_name,
        "phrase": phrase_text,
        "char_span": [int(char_span[0]), int(char_span[1])],
        "token_span": [int(token_span[0]), int(token_span[1])],
        "coarse_bbox": list(coarse_bbox) if coarse_bbox else None,
        "final_bbox": list(final_bbox) if final_bbox else None,
        "attention_meta": heat_meta,
        "answer_text": answer_cache.get("output_text"),
        "use_sam": bool(use_sam),
    }
    write_json(mode_dir / "meta.json", to_serializable(meta))

    return {
        "mode": mode_name,
        "coarse_mask": coarse_mask,
        "final_mask": final_mask,
        "coarse_bbox": list(coarse_bbox) if coarse_bbox else None,
        "final_bbox": list(final_bbox) if final_bbox else None,
        "coarse_mask_path": str(coarse_mask_path),
        "final_mask_path": str(final_mask_path),
        "coarse_overlay_path": str(coarse_overlay_path),
        "final_overlay_path": str(final_overlay_path),
        "heat_overlay_path": str(heat_overlay_path),
        "heatmap_path": str(heatmap_path),
        "roi_path": str(roi_path) if final_bbox else None,
        "heat_meta": heat_meta,
    }


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            args.checkpoint = str((REPO_ROOT / ckpt_path).resolve())
    run_dir = resolve_run_dir(Path(args.out_dir), args.run_name)

    cfg = Config.fromfile(str(cfg_path))
    apply_prompt_overrides(cfg, args, args.prompt_file)
    model = load_model(cfg, args)
    image, image_path = load_image(args.image)
    history = load_history(args.history_file)

    if args.answer_text:
        answer_text = args.answer_text.strip()
        print("[Step] Building answer cache from explicit answer_text...")
        generated_cache = model.build_answer_cache(
            image=image,
            question=args.question,
            answer_text=answer_text,
            history=history,
        )
        answer_source = "explicit_answer_text"
    else:
        print("[Step] Running generate-based answer cache...")
        generated_cache = model.answer(
            image=image,
            question=args.question,
            history=history,
            max_new_tokens=args.max_new_tokens,
        )
        answer_text = generated_cache["output_text"]
        answer_source = "model.answer"
    print(f"[Answer] {answer_text}")

    phrase_text, char_span, token_span, candidates = select_phrase(
        model=model,
        question=args.question,
        answer_text=answer_text,
        phrase_override=args.phrase,
        occurrence=args.occurrence,
        max_tokens=args.phrase_max_tokens,
        max_phrases=args.max_phrases,
    )
    print(
        f"[Phrase] text={phrase_text!r} char_span={char_span} token_span={token_span}"
    )

    print("[Step] Building teacher-forced cache...")
    teacher_cache = build_teacher_forced_cache(
        model=model,
        image=image,
        question=args.question,
        answer_text=answer_text,
        history=history,
    )

    generate_prompt = build_generation_prompt_text(
        model, image, args.question, history=history
    )
    (run_dir / "generate_prompt.txt").write_text(
        generate_prompt, encoding="utf-8"
    )
    (run_dir / "teacher_forced_prompt.txt").write_text(
        teacher_cache["prompt_text"], encoding="utf-8"
    )

    print("[Step] Grounding on generate cache...")
    generated_result = save_mode_artifacts(
        mode_name="generate",
        model=model,
        image=image,
        answer_cache=generated_cache,
        token_span=token_span,
        char_span=char_span,
        phrase_text=phrase_text,
        out_dir=run_dir,
        use_sam=not args.no_sam,
        layer=args.layer,
        head=args.head,
        reduction=args.reduction,
    )

    print("[Step] Grounding on teacher-forced cache...")
    teacher_result = save_mode_artifacts(
        mode_name="teacher_forced",
        model=model,
        image=image,
        answer_cache=teacher_cache,
        token_span=token_span,
        char_span=char_span,
        phrase_text=phrase_text,
        out_dir=run_dir,
        use_sam=not args.no_sam,
        layer=args.layer,
        head=args.head,
        reduction=args.reduction,
    )

    save_comparison_panel(image, run_dir, generated_result, teacher_result)

    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(cfg_path),
        "checkpoint": args.checkpoint,
        "image_path": str(image_path),
        "question": args.question,
        "history": history,
        "answer_text": answer_text,
        "answer_source": answer_source,
        "phrase": phrase_text,
        "char_span": [int(char_span[0]), int(char_span[1])],
        "token_span": [int(token_span[0]), int(token_span[1])],
        "phrase_candidates": candidates,
        "generate": {
            "attention_shape": list(generated_cache["attention_maps"].shape),
            "hidden_shape": list(generated_cache["hidden_states"].shape),
            "answer_token_span_in_full": generated_cache.get(
                "answer_token_span_in_full"
            ),
            "output_ids_alignment_source": generated_cache.get(
                "output_ids_alignment_source"
            ),
            "artifacts": to_serializable(strip_mask_arrays(generated_result)),
        },
        "teacher_forced": {
            "attention_shape": list(teacher_cache["attention_maps"].shape),
            "hidden_shape": list(teacher_cache["hidden_states"].shape),
            "answer_token_span_in_full": teacher_cache["answer_token_span_in_full"],
            "output_ids_alignment_source": teacher_cache.get(
                "output_ids_alignment_source"
            ),
            "artifacts": to_serializable(strip_mask_arrays(teacher_result)),
        },
        "diff": {
            "coarse_mask_iou": binary_iou(
                generated_result["coarse_mask"], teacher_result["coarse_mask"]
            ),
            "final_mask_iou": binary_iou(
                generated_result["final_mask"], teacher_result["final_mask"]
            ),
        },
    }
    write_json(run_dir / "report.json", to_serializable(report))
    print(f"[Done] Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
