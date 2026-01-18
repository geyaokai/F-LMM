from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


TokenSpanInput = Union[Dict[str, Any], Iterable[int], Tuple[int, int]]


def parse_token_span(span: TokenSpanInput) -> Tuple[int, int]:
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")
    else:
        items = list(span)
        if len(items) != 2:
            raise ValueError("token_span must have [start, end].")
        start, end = items
    if start is None or end is None:
        raise ValueError("token_span requires start/end.")
    start_i = int(start)
    end_i = int(end)
    if end_i <= start_i:
        raise ValueError("token_span end must be greater than start.")
    return start_i, end_i


def _resolve_index(value: Any, total: int, name: str) -> Optional[int]:
    if value is None or value == "mean":
        return None
    idx = int(value)
    if idx < 0:
        idx = total + idx
    if idx < 0 or idx >= total:
        raise ValueError(f"{name} index out of range: {idx} (total={total})")
    return idx


def _select_layers_heads(
    attn: torch.Tensor, layer: Any, head: Any
) -> Tuple[torch.Tensor, Any, Any]:
    layer_idx = _resolve_index(layer, attn.shape[0], "layer")
    head_idx = _resolve_index(head, attn.shape[1], "head")
    if layer_idx is not None:
        attn = attn[layer_idx : layer_idx + 1]
        layer_value: Any = int(layer_idx)
    else:
        layer_value = "mean"
    if head_idx is not None:
        attn = attn[:, head_idx : head_idx + 1]
        head_value: Any = int(head_idx)
    else:
        head_value = "mean"
    return attn, layer_value, head_value


def build_i2t_heatmap(
    attention_maps: torch.Tensor,
    token_span: TokenSpanInput,
    layer: Any = "mean",
    head: Any = "mean",
    reduction: str = "mean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if attention_maps is None:
        raise ValueError("attention_maps missing; run answer() first.")
    if not torch.is_tensor(attention_maps):
        attention_maps = torch.tensor(attention_maps)
    if attention_maps.dim() != 5:
        raise ValueError("attention_maps must be [layers, heads, tokens, H, W].")
    start, end = parse_token_span(token_span)
    token_len = int(attention_maps.shape[2])
    if start < 0 or end > token_len:
        raise ValueError(f"token_span out of range: {start}-{end} (len={token_len})")

    attn = attention_maps.detach().float().cpu()
    attn, layer_value, head_value = _select_layers_heads(attn, layer, head)
    token_slice = attn[:, :, start:end]
    if reduction == "max":
        token_reduced = token_slice.max(dim=2).values
    else:
        token_reduced = token_slice.mean(dim=2)
    heatmap = token_reduced.mean(dim=(0, 1))
    heat = heatmap.numpy()
    min_v = float(heat.min())
    max_v = float(heat.max())
    if max_v > min_v:
        heat = (heat - min_v) / (max_v - min_v)
    else:
        heat = np.zeros_like(heat, dtype=np.float32)
    meta = {
        "grid_h": int(heat.shape[0]),
        "grid_w": int(heat.shape[1]),
        "layer": layer_value,
        "head": head_value,
        "reduction": reduction,
        "normalize": "minmax",
        "token_span": {"start": int(start), "end": int(end), "right_open": True},
        "span_type": "token_index",
    }
    return heat.astype(np.float32), meta


def _heat_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heat = np.clip(heatmap, 0.0, 1.0)
    red = (255.0 * heat).astype(np.uint8)
    green = (140.0 * np.sqrt(heat)).astype(np.uint8)
    blue = np.zeros_like(red, dtype=np.uint8)
    return np.stack([red, green, blue], axis=-1)


def render_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[Image.Image, Image.Image]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    heat_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_uint8, mode="L")
    heat_img = heat_img.resize(image.size, resample=Image.BILINEAR)
    heat_arr = np.array(heat_img, dtype=np.float32) / 255.0
    heat_rgb = _heat_to_rgb(heat_arr)
    base = np.array(image, dtype=np.float32)
    alpha_map = (alpha * heat_arr).reshape(heat_arr.shape + (1,))
    overlay = base * (1.0 - alpha_map) + heat_rgb.astype(np.float32) * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB"), Image.fromarray(heat_rgb, mode="RGB")


def token_text_from_ids(tokenizer, output_ids: Any, start: int, end: int) -> str:
    if output_ids is None:
        return ""
    if torch.is_tensor(output_ids):
        ids = output_ids.tolist()
    else:
        ids = list(output_ids)
    snippet = ids[start:end]
    if not snippet:
        return ""
    return tokenizer.decode(snippet, skip_special_tokens=True)


def dump_meta(path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
