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

from scripts.demo.token_to_region import (  # noqa: E402
    build_token_to_region_heatmap,
    render_overlay,
    token_text_from_ids,
)
from scripts.demo.interact import load_model, resolve_image_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate token-to-region heatmap for a token span."
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

    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--prompt", required=True, help="Prompt / question.")
    parser.add_argument("--token-start", type=int, required=True)
    parser.add_argument("--token-end", type=int, required=True)
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--head", default="mean")
    parser.add_argument("--reduction", default="mean", choices=["mean", "max"])
    parser.add_argument("--out-dir", default="scripts/demo/results/token_to_region")
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
    output = model.answer(image=image, question=args.prompt, max_new_tokens=args.max_new_tokens)
    heatmap, meta = build_token_to_region_heatmap(
        attention_maps=output.get("attention_maps"),
        token_span={"start": args.token_start, "end": args.token_end},
        layer=args.layer,
        head=args.head,
        reduction=args.reduction,
    )
    token_text = token_text_from_ids(
        model.tokenizer, output.get("output_ids"), args.token_start, args.token_end
    )
    if token_text:
        meta["token_text"] = token_text
    overlay, heatmap_img = render_overlay(image, heatmap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex[:8]
    overlay_path = out_dir / f"overlay_{stem}.png"
    heatmap_path = out_dir / f"heatmap_{stem}.png"
    overlay.save(overlay_path)
    heatmap_img.save(heatmap_path)
    meta_path = out_dir / f"meta_{stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(overlay_path))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
