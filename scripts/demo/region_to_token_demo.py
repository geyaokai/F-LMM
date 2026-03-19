#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from mmengine.config import Config
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.region_to_token import (  # noqa: E402
    build_fallback_phrase_records,
    build_phrase_records,
    build_region_mask,
    build_region_to_token_scores,
    build_token_records,
    dump_json,
    rank_records,
    render_region_artifacts,
)
from scripts.demo.interact import (  # noqa: E402
    build_offsets,
    build_phrase_candidates,
    dedupe_phrase_candidates,
    extract_phrases_via_model,
    load_model,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate region-to-token rankings for a selected bbox."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py",
        help="Path to Qwen config.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map for the Qwen base model ("auto" or "none").',
    )
    parser.add_argument("--device-max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--extra-prompt", default="")
    parser.add_argument("--max-history-turns", type=int, default=0)
    parser.add_argument("--phrase-max-tokens", type=int, default=128)
    parser.add_argument("--max-phrases", type=int, default=6)

    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--prompt", required=True, help="Prompt / question.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Region bbox in original image coordinates.",
    )
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--head", default="mean")
    parser.add_argument(
        "--reduction", default="mean", choices=["mean", "max", "sum"]
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--out-dir", default="scripts/demo/results/region_to_token")
    return parser.parse_args()


def _load_image(path_str: str) -> Image.Image:
    path = resolve_image_path(path_str)
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    cfg_path = (
        args.config if Path(args.config).is_absolute() else str(REPO_ROOT / args.config)
    )
    cfg = Config.fromfile(cfg_path)
    model = load_model(cfg, args)
    image = _load_image(args.image)
    answer = model.answer(
        image=image, question=args.prompt, max_new_tokens=args.max_new_tokens
    )
    answer_text = answer.get("output_text") or ""
    offsets = build_offsets(model.tokenizer, answer_text) if answer_text else []
    phrase_texts = extract_phrases_via_model(
        model,
        args.prompt,
        answer_text,
        args.phrase_max_tokens,
        args.max_phrases,
    )
    phrases = dedupe_phrase_candidates(
        build_phrase_candidates(answer_text, phrase_texts, offsets)
    )

    raw_scores, norm_scores, per_layer_scores, meta = build_region_to_token_scores(
        attention_maps=answer.get("attention_maps"),
        bbox=args.bbox,
        image_size=image.size,
        layer=args.layer,
        head=args.head,
        reduction=args.reduction,
    )
    token_records = build_token_records(
        model.tokenizer,
        answer.get("output_ids"),
        offsets,
        raw_scores,
        norm_scores,
    )
    top_tokens = rank_records(token_records, topk=args.topk)
    phrase_records = build_phrase_records(norm_scores, phrases)
    top_phrases = rank_records(phrase_records, topk=args.topk)
    if not top_phrases:
        top_phrases = build_fallback_phrase_records(top_tokens)

    grid_shape = meta["grid_shape"]
    region_mask, _ = build_region_mask(
        meta["bbox"],
        image.size,
        (int(grid_shape["height"]), int(grid_shape["width"])),
    )
    bbox_overlay, region_overlay, region_heatmap = render_region_artifacts(
        image,
        meta["bbox"],
        region_mask,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex[:8]
    bbox_overlay_path = out_dir / f"bbox_overlay_{stem}.png"
    region_overlay_path = out_dir / f"region_overlay_{stem}.png"
    region_heatmap_path = out_dir / f"region_heatmap_{stem}.png"
    meta_path = out_dir / f"meta_{stem}.json"
    ranking_path = out_dir / f"ranking_{stem}.json"
    bbox_overlay.save(bbox_overlay_path)
    region_overlay.save(region_overlay_path)
    region_heatmap.save(region_heatmap_path)

    meta["topk"] = int(args.topk)
    meta["layer_summary"] = [
        {
            "layer": int(idx) if meta.get("layer") == "mean" else int(meta["layer"]),
            "max_token_index": int(layer_scores.argmax()) if layer_scores.size else -1,
            "max_token_score": float(layer_scores.max()) if layer_scores.size else 0.0,
        }
        for idx, layer_scores in enumerate(per_layer_scores)
    ]
    dump_json(meta_path, meta)
    dump_json(
        ranking_path,
        {
            "prompt": args.prompt,
            "answer_text": answer_text,
            "bbox": meta["bbox"],
            "top_tokens": top_tokens,
            "top_phrases": top_phrases,
            "token_scores": token_records,
        },
    )

    print(str(bbox_overlay_path))
    print(str(region_overlay_path))
    print(json.dumps({"top_tokens": top_tokens, "top_phrases": top_phrases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
