from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.demo.token_to_region import render_overlay, token_text_from_ids


BBoxInput = Union[Dict[str, Any], Iterable[float], Tuple[float, float, float, float]]


def parse_bbox(
    bbox: BBoxInput,
    bbox_format: str = "xyxy",
) -> Tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        if "xyxy" in bbox:
            values = bbox["xyxy"]
            bbox_format = "xyxy"
        elif {"x1", "y1", "x2", "y2"}.issubset(bbox):
            values = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            bbox_format = "xyxy"
        elif {"x", "y", "w", "h"}.issubset(bbox):
            values = [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]
            bbox_format = "xywh"
        else:
            raise ValueError("bbox dict must provide xyxy or xywh fields.")
    else:
        values = list(bbox)
    if len(values) != 4:
        raise ValueError("bbox must have 4 values.")
    x1, y1, x2, y2 = [float(v) for v in values]
    if bbox_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2>x1 and y2>y1.")
    return x1, y1, x2, y2


def _resolve_index(value: Any, total: int, name: str) -> int | None:
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


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v > min_v:
        return ((scores - min_v) / (max_v - min_v)).astype(np.float32)
    return np.zeros_like(scores, dtype=np.float32)


def build_region_mask(
    bbox: BBoxInput,
    image_size: Tuple[int, int],
    grid_shape: Tuple[int, int],
    bbox_format: str = "xyxy",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_w, image_h = image_size
    grid_h, grid_w = grid_shape
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_size must be positive.")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_shape must be positive.")
    x1, y1, x2, y2 = parse_bbox(bbox, bbox_format=bbox_format)
    x1 = min(max(0.0, x1), float(image_w))
    x2 = min(max(0.0, x2), float(image_w))
    y1 = min(max(0.0, y1), float(image_h))
    y2 = min(max(0.0, y2), float(image_h))
    if x2 <= x1:
        x2 = min(float(image_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h), y1 + 1.0)
    col_start = max(0, min(grid_w - 1, int(np.floor(x1 / image_w * grid_w))))
    col_end = max(col_start + 1, min(grid_w, int(np.ceil(x2 / image_w * grid_w))))
    row_start = max(0, min(grid_h - 1, int(np.floor(y1 / image_h * grid_h))))
    row_end = max(row_start + 1, min(grid_h, int(np.ceil(y2 / image_h * grid_h))))
    mask = np.zeros((grid_h, grid_w), dtype=np.float32)
    mask[row_start:row_end, col_start:col_end] = 1.0
    meta = {
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "bbox_format": "xyxy",
        "image_size": {"width": int(image_w), "height": int(image_h)},
        "grid_shape": {"height": int(grid_h), "width": int(grid_w)},
        "grid_bbox": [int(col_start), int(row_start), int(col_end), int(row_end)],
        "region_cell_count": int(mask.sum()),
    }
    return mask, meta


def build_region_to_token_scores(
    attention_maps: torch.Tensor,
    bbox: BBoxInput,
    image_size: Tuple[int, int],
    layer: Any = "mean",
    head: Any = "mean",
    reduction: str = "mean",
    bbox_format: str = "xyxy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if attention_maps is None:
        raise ValueError("attention_maps missing; run answer() first.")
    if not torch.is_tensor(attention_maps):
        attention_maps = torch.tensor(attention_maps)
    if attention_maps.dim() != 5:
        raise ValueError("attention_maps must be [layers, heads, tokens, H, W].")
    attn = attention_maps.detach().float().cpu()
    attn, layer_value, head_value = _select_layers_heads(attn, layer, head)
    grid_h = int(attn.shape[-2])
    grid_w = int(attn.shape[-1])
    region_mask, region_meta = build_region_mask(
        bbox=bbox,
        image_size=image_size,
        grid_shape=(grid_h, grid_w),
        bbox_format=bbox_format,
    )
    mask_t = torch.from_numpy(region_mask).to(attn)
    masked = attn * mask_t.view(1, 1, 1, grid_h, grid_w)
    if reduction == "max":
        spatial_scores = masked.amax(dim=(-1, -2))
    elif reduction == "sum":
        spatial_scores = masked.sum(dim=(-1, -2))
    else:
        denom = float(region_mask.sum())
        if denom <= 0:
            raise ValueError("Selected region is empty after grid projection.")
        spatial_scores = masked.sum(dim=(-1, -2)) / denom
    per_token_raw = spatial_scores.mean(dim=(0, 1)).numpy().astype(np.float32)
    per_token_norm = normalize_scores(per_token_raw)
    per_layer_token = spatial_scores.mean(dim=1).numpy().astype(np.float32)
    meta = {
        "layer": layer_value,
        "head": head_value,
        "reduction": reduction,
        "normalize": "minmax",
        "score_axis": "generated_token_index",
        "analysis_type": "region_to_token",
    }
    meta.update(region_meta)
    return per_token_raw, per_token_norm, per_layer_token, meta


def build_token_records(
    tokenizer,
    output_ids: Any,
    offsets: Sequence[Tuple[int, int]],
    raw_scores: np.ndarray,
    norm_scores: np.ndarray,
) -> List[Dict[str, Any]]:
    if output_ids is None:
        raise ValueError("output_ids missing from answer cache.")
    if torch.is_tensor(output_ids):
        ids = output_ids.tolist()
    else:
        ids = list(output_ids)
    total = min(len(ids), int(raw_scores.shape[0]), int(norm_scores.shape[0]))
    records: List[Dict[str, Any]] = []
    for idx in range(total):
        if idx < len(offsets):
            start, end = offsets[idx]
        else:
            start, end = (0, 0)
        records.append(
            {
                "index": int(idx),
                "text": token_text_from_ids(tokenizer, ids, idx, idx + 1),
                "char_span": [int(start), int(end)],
                "raw_score": float(raw_scores[idx]),
                "score": float(norm_scores[idx]),
            }
        )
    return records


def rank_records(
    records: Sequence[Dict[str, Any]],
    topk: int = 5,
    *,
    skip_blank_text: bool = True,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for record in records:
        text = str(record.get("text") or "")
        if skip_blank_text and not text.strip():
            continue
        ranked.append(dict(record))
    ranked.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("index", 0))))
    return ranked[: max(int(topk), 0)]


def build_phrase_records(
    token_scores: np.ndarray,
    phrases: Sequence[Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    total_tokens = int(token_scores.shape[0])
    for idx, phrase in enumerate(phrases):
        text = getattr(phrase, "text", None)
        char_span = getattr(phrase, "char_span", None)
        token_span = getattr(phrase, "token_span", None)
        if isinstance(phrase, dict):
            text = phrase.get("text", text)
            char_span = phrase.get("char_span", char_span)
            token_span = phrase.get("token_span", token_span)
        if not text or not token_span or len(token_span) < 2:
            continue
        start = max(0, int(token_span[0]))
        end = min(total_tokens, int(token_span[1]))
        if end <= start:
            continue
        span_scores = token_scores[start:end]
        if span_scores.size == 0:
            continue
        if not char_span or len(char_span) < 2:
            char_span = [0, 0]
        records.append(
            {
                "index": int(idx),
                "text": str(text),
                "char_span": [int(char_span[0]), int(char_span[1])],
                "token_span": [int(start), int(end)],
                "score": float(span_scores.mean()),
                "score_max": float(span_scores.max()),
            }
        )
    return records


def build_fallback_phrase_records(
    ranked_tokens: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, token in enumerate(ranked_tokens):
        records.append(
            {
                "index": int(idx),
                "text": token.get("text", ""),
                "char_span": list(token.get("char_span", [0, 0])),
                "token_span": [int(token.get("index", 0)), int(token.get("index", 0)) + 1],
                "score": float(token.get("score", 0.0)),
                "score_max": float(token.get("score", 0.0)),
            }
        )
    return records


def render_region_artifacts(
    image: Image.Image,
    bbox: Sequence[float],
    region_mask: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    region_overlay, region_heatmap = render_overlay(image, region_mask, alpha=alpha)
    bbox_overlay = image.copy()
    draw = ImageDraw.Draw(bbox_overlay)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
    return bbox_overlay, region_overlay, region_heatmap


def dump_json(path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
