#!/usr/bin/env python3
"""
import 部分
dataclass 定义
  - PhraseCandidate        # 存储候选短语及其字符/Token跨度
  - GroundRecord           # 记录一次 grounding 的输出文件和信息
  - AskCommand             # 解析 "ask" 命令的参数
  - SessionState           # 会话状态（模型、当前图像、历史、grounding 记录等）

工具函数
  - 历史管理 (append_history_entry, clear_history, history_turns)
  - 答案清理 (detect_answer_artifacts, clean_answer_for_display)
  - 设备映射解析 (parse_device_map_arg, parse_max_memory_arg, resolve_inference_device)
  - 模型辅助模块移动 (move_auxiliary_modules)
  - 模型加载 (load_model)
  - 图像加载 (load_image, resolve_image_path)
  - 短语提取 (extract_phrases_via_model, _extract_phrases_spacy, _rerank_phrases_with_model)
  - Token 偏移处理 (build_offsets, char_to_token, token_span_to_char)
  - 短语候选构建和去重 (build_phrase_candidates, dedupe_phrase_candidates)
  - grounding 核心 (perform_ground, perform_ground_custom, mask_to_box, blend_mask, save_ground_outputs)
  - 图像尺寸适配 (ensure_min_image_size)
  - 命令处理函数 (handle_load, handle_ask, handle_ground, handle_inspect, handle_cot_resample)
  - 核心管道 (pipeline_default_ask)
  - 帮助 (print_help)
  - 主循环 (main)
"""

import argparse
from collections import deque
import json
import os
import re
import shlex
import sys
import uuid
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Enable arrow-key editing when readline exists.
    import readline  # noqa: F401
except ImportError:
    readline = None  # type: ignore

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from xtuner.registry import BUILDER

try:
    import spacy

    _spacy_nlp = None
    _spacy_warned = False
except ImportError:
    spacy = None
    _spacy_nlp = None
    _spacy_warned = False

# import pdb;pdb.set_trace()
from scripts.demo.checkpoint_utils import guess_load_checkpoint  # noqa: E402
from scripts.demo.utils import colors  # noqa: E402
from scripts.demo.web.backend.task_queue.paths import (  # noqa: E402
    ATTN_KIND_REGION_TO_TOKEN,
    ATTN_KIND_TOKEN_TO_REGION,
    SessionPaths,
)

_PHRASE_BREAK_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "beside",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "near",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "which",
    "while",
    "with",
    "wearing",
}

_PHRASE_LEADING_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "its",
}


@dataclass
class PhraseCandidate:
    text: str
    char_span: Tuple[int, int]
    token_span: Tuple[int, int]


@dataclass
class GroundRecord:
    overlay_path: Path
    mask_path: Path
    roi_path: Optional[Path]
    phrase_text: str
    token_span: Tuple[int, int]
    char_span: Tuple[int, int]
    roi_image: Optional[Image.Image]
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class AskCommand:
    mode: str  # 'default', 'roi', 'cot'
    index: Optional[int]
    question: str
    reset_history: bool = False


@dataclass
class SessionState:
    model: Any
    args: argparse.Namespace
    result_root: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    current_image: Optional[Image.Image] = None
    current_image_path: Optional[Path] = None
    last_answer: Optional[dict] = None
    phrases: List[PhraseCandidate] = field(default_factory=list)
    token_offsets: Optional[List[Tuple[int, int]]] = None
    ground_id: int = 0
    attn_counters: Dict[str, int] = field(
        default_factory=lambda: {
            ATTN_KIND_TOKEN_TO_REGION: 0,
            ATTN_KIND_REGION_TO_TOKEN: 0,
        }
    )
    last_records: List[GroundRecord] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    turn_idx: int = 0
    session_paths: SessionPaths = field(init=False)
    session_dir: Path = field(init=False)

    def __post_init__(self):
        self.session_paths = SessionPaths(self.result_root, self.session_id)
        self.session_dir = self.session_paths.session_root
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_paths.turns_dir.mkdir(parents=True, exist_ok=True)
        self.session_paths.images_dir.mkdir(parents=True, exist_ok=True)

    def reset_answer(self):
        self.last_answer = None
        self.phrases = []
        self.token_offsets = None
        self.last_records = []
        self.history.clear()
        self.ground_id = 0
        self.attn_counters = {
            ATTN_KIND_TOKEN_TO_REGION: 0,
            ATTN_KIND_REGION_TO_TOKEN: 0,
        }


def append_history_entry(session: SessionState, role: str, text: str):
    text = (text or "").strip()
    if not text or session.args.max_history_turns <= 0:
        return
    session.history.append({"role": role, "text": text})
    limit = max(session.args.max_history_turns * 2, 0)
    if limit > 0 and len(session.history) > limit:
        session.history = session.history[-limit:]


def clear_history(session: SessionState):
    if session.history:
        session.history.clear()
        print("[History] Cleared conversation context.")
    else:
        print("[History] Context already empty.")


def history_turns(session: SessionState) -> int:
    return len(session.history) // 2


ANSWER_ARTIFACT_MARKERS = (
    "addcriterion",
    "matchcondition",
    "guidid",
    "自动生成",
)


def detect_answer_artifacts(text: str) -> List[str]:
    lowered = (text or "").lower()
    hits: List[str] = []
    for marker in ANSWER_ARTIFACT_MARKERS:
        if marker.lower() in lowered:
            hits.append(marker)
    return hits


def clean_answer_for_display(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    cut = len(text)
    for marker in ANSWER_ARTIFACT_MARKERS:
        pos = lowered.find(marker.lower())
        if pos >= 0:
            cut = min(cut, pos)
    cleaned = text[:cut].strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_device_map_arg(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    normalized = raw_value.strip()
    if not normalized or normalized.lower() in {
        "none",
        "single",
        "disable",
        "disabled",
    }:
        return None
    return normalized


def parse_max_memory_arg(raw_value: Optional[str]) -> Optional[Dict[str, str]]:
    if not raw_value:
        return None
    entries = [chunk.strip() for chunk in raw_value.split(",") if chunk.strip()]
    if not entries:
        return None
    limits: Dict[str, str] = {}
    for item in entries:
        if ":" not in item:
            raise ValueError(
                f'Invalid max-memory entry "{item}". Use format "0:20GiB,1:20GiB".'
            )
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f'Invalid max-memory entry "{item}".')
        limits[key] = value
    return limits or None


def resolve_inference_device(name: str) -> torch.device:
    """Return a torch.device and ensure CUDA devices are usable."""
    try:
        device = torch.device(name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid device "{name}": {exc}') from exc
    if device.type != "cuda":
        return device
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        raise RuntimeError(f'Unable to initialize CUDA device "{name}": {exc}') from exc
    if not cuda_available:
        raise RuntimeError(
            f'CUDA device "{name}" requested, but CUDA is unavailable. '
            'Check your NVIDIA driver/CUDA setup or pass "--device cpu".'
        )
    try:
        total = torch.cuda.device_count()
    except Exception as exc:
        raise RuntimeError(
            f'Failed to query CUDA devices while resolving "{name}": {exc}'
        ) from exc
    if device.index is not None and (device.index < 0 or device.index >= total):
        raise RuntimeError(
            f'Device "{name}" targets cuda:{device.index}, but only {total} device(s) are visible. '
            "Adjust CUDA_VISIBLE_DEVICES or select a valid --device."
        )
    return device


def move_auxiliary_modules(model, device: torch.device):
    for attr in ("mask_head", "sam", "text_proj"):
        module = getattr(model, attr, None)
        if module is not None:
            module.to(device)
    if hasattr(model, "text_layer_weights"):
        model.text_layer_weights = torch.nn.Parameter(
            model.text_layer_weights.to(device)
        )


def infer_qwen_primary_device(
    qwen_model, fallback_device: torch.device
) -> torch.device:
    # Prefer qwen_model.device property if it exists and is valid.
    existing = getattr(qwen_model, "device", None)
    if existing is not None:
        try:
            return torch.device(existing)
        except (TypeError, RuntimeError):
            pass
    device_map = getattr(qwen_model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        first_target = next(iter(device_map.values()))
        try:
            device = torch.device(first_target)
        except (TypeError, RuntimeError):
            device = fallback_device
    else:
        device = fallback_device
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Qwen-FLMM demo: ask → ground → inspect."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py",
        help="Path to Qwen config (default: 7B Qwen2.5-VL).",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for the Qwen base model. "
        'Use "auto" to shard across visible GPUs or "none" to disable.',
    )
    parser.add_argument(
        "--device-max-memory",
        default=None,
        help="Optional per-device memory hints when using --device-map. "
        'Format: "0:20GiB,1:20GiB".',
    )
    parser.add_argument("--image", default=None, help="Optional image to preload.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--phrase-max-tokens", type=int, default=128)
    parser.add_argument("--max-phrases", type=int, default=6)
    parser.add_argument("--results-dir", default="scripts/demo/results/qwen")
    parser.add_argument("--inspect-prompt", default="Describe this region in detail.")
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM refinement.")
    parser.add_argument(
        "--extra-prompt",
        default="",
        help="Force append text to every question (default: empty).",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=4,
        help="Number of previous QA turns to keep in context (default: 4, 0 to disable).",
    )
    return parser.parse_args()


def load_model(cfg, args):
    device_map = parse_device_map_arg(args.device_map)
    max_memory = parse_max_memory_arg(args.device_max_memory) if device_map else None
    model_cfg = cfg.model
    qwen_cfg = model_cfg.get("model", None)
    if device_map and qwen_cfg is not None:
        qwen_cfg["device_map"] = device_map
        if max_memory:
            qwen_cfg["max_memory"] = max_memory
    model = BUILDER.build(cfg.model)
    if args.checkpoint:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Model] Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    target_device = resolve_inference_device(args.device)
    args.device = str(target_device)
    if device_map:
        move_auxiliary_modules(model, target_device)
        qwen_device = infer_qwen_primary_device(model.qwen_model, target_device)
        print(
            f'[Model] Using device_map="{device_map}" (primary device: {qwen_device})'
        )
    else:
        model = model.to(target_device)
        qwen_device = target_device
    model = model.eval()
    # cache the inferred device on FrozenQwen to avoid repeated heuristics
    if hasattr(model, "set_qwen_device"):
        model.set_qwen_device(qwen_device)
    append_prompt = args.extra_prompt or ""
    model._prepare_for_generation(
        image_processor=cfg.image_processor,
        prompt_template=cfg.prompt_template,
        max_new_tokens=args.max_new_tokens,
        additional_prompt=append_prompt,
        max_history_turns=args.max_history_turns,
    )
    model.phrase_extract_prompt = getattr(args, "phrase_extract_prompt", None)
    model.phrase_rerank_prompt = getattr(args, "phrase_rerank_prompt", None)
    model.roi_extra_prompt = getattr(args, "roi_extra_prompt", None)
    return model


def resolve_image_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        parts = raw.parts
        if parts and parts[0] == REPO_ROOT.name:
            tail = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            candidates.append(REPO_ROOT / tail)
    for cand in candidates:
        resolved = cand.resolve(strict=False)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(candidates[-1] if candidates else raw)


def load_image(path_str: str) -> Tuple[Image.Image, Path]:
    path = resolve_image_path(path_str)
    image = Image.open(path).convert("RGB")
    return image, path


def parse_ask_command(arg_str: str) -> AskCommand:
    parser = argparse.ArgumentParser(prog="ask", add_help=False, exit_on_error=False)
    parser.add_argument("--reset-history", action="store_true")
    parser.add_argument("question", nargs="*")
    tokens = shlex.split(arg_str)
    try:
        opts = parser.parse_args(tokens)
    except (SystemExit, argparse.ArgumentError) as exc:
        raise ValueError("无法解析 ask 参数。") from exc
    question = " ".join(opts.question).strip()
    return AskCommand(
        mode="default", index=None, question=question, reset_history=opts.reset_history
    )


def _get_spacy():
    global _spacy_nlp, _spacy_warned
    if _spacy_nlp is not None:
        return _spacy_nlp
    if spacy is None:
        if not _spacy_warned:
            print("[Debug] spaCy import unavailable; phrase extraction will use model/regex fallback.")
            _spacy_warned = True
        return None
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        if not _spacy_warned:
            print("[Debug] spaCy model 'en_core_web_sm' unavailable; phrase extraction will use model/regex fallback.")
            _spacy_warned = True
        _spacy_nlp = None
    return _spacy_nlp


def _extract_phrases_spacy(
    answer_text: str, limit: int
) -> List[Tuple[str, Tuple[int, int]]]:
    nlp = _get_spacy()
    if nlp is None:
        return []
    doc = nlp(answer_text)
    seen = set()
    phrases: List[Tuple[str, Tuple[int, int]]] = []

    def add_phrase(text: str, start: int, end: int):
        cleaned = text.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        word_count = len(cleaned.split())
        if word_count == 0 or word_count > 3:
            return
        seen.add(key)
        phrases.append((cleaned, (start, end)))

    for chunk in doc.noun_chunks:
        add_phrase(chunk.text, chunk.start_char, chunk.end_char)

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and token.is_alpha:
            add_phrase(token.text, token.idx, token.idx + len(token.text))

    return phrases[:limit]


def _extract_phrases_regex(
    answer_text: str, limit: int
) -> List[Tuple[str, Tuple[int, int]]]:
    token_matches = list(re.finditer(r"[A-Za-z]+(?:['-][A-Za-z]+)?", answer_text))
    if not token_matches:
        return []

    chunks: List[List[re.Match[str]]] = []
    current: List[re.Match[str]] = []
    for match in token_matches:
        word = match.group(0).lower()
        if word in _PHRASE_BREAK_WORDS:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(match)
    if current:
        chunks.append(current)

    phrases: List[Tuple[str, Tuple[int, int]]] = []
    seen: set[str] = set()

    def add_phrase(start_idx: int, end_idx: int, chunk: List[re.Match[str]]) -> None:
        if start_idx < 0 or end_idx >= len(chunk) or start_idx > end_idx:
            return
        start = chunk[start_idx].start()
        end = chunk[end_idx].end()
        text = answer_text[start:end].strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        word_count = len(text.split())
        if word_count == 0 or word_count > 3:
            return
        seen.add(key)
        phrases.append((text, (start, end)))

    for chunk in chunks:
        start_idx = 0
        while start_idx < len(chunk):
            if chunk[start_idx].group(0).lower() in _PHRASE_LEADING_WORDS:
                start_idx += 1
            else:
                break
        if start_idx >= len(chunk):
            continue
        trimmed = chunk[start_idx:]
        add_phrase(0, len(trimmed) - 1, trimmed)
        if len(trimmed) >= 2:
            add_phrase(len(trimmed) - 2, len(trimmed) - 1, trimmed)
        add_phrase(len(trimmed) - 1, len(trimmed) - 1, trimmed)
        if len(phrases) >= limit:
            break

    return phrases[:limit]


def _rerank_phrases_with_model(
    model,
    question: str,
    answer_text: str,
    candidates: List[Tuple[str, Tuple[int, int]]],
    max_tokens: int,
    limit: int,
) -> List[Tuple[str, Tuple[int, int]]]:
    if not candidates:
        return []
    question = str(question or "").strip()
    answer_text = str(answer_text or "")
    numbered = [f"{i}. {p[0]}" for i, p in enumerate(candidates)]
    override = getattr(model, "phrase_rerank_prompt", None)
    if override:
        try:
            prompt = override.format(
                question=question,
                answer=answer_text,
                candidates="\n".join(numbered),
                limit=limit,
            )
        except Exception:
            prompt = (
                f"{override}\n\nQuestion:\n{question}\n\nAnswer:\n{answer_text}\nCandidates:\n"
                + "\n".join(numbered)
                + f"\nSelect up to {limit}.\n"
            )
    else:
        prompt = (
            f"<|im_start|>system\n"
            "You rerank candidate noun phrases for visual grounding.\n"
            "The user question defines what matters.\n"
            "The selected phrases will be used for image grounding / segmentation, so prioritize concrete visible entities in the image, "
            "such as objects, people, animals, body parts, or clearly visible scene items.\n"
            "Prefer the entity that most directly answers the question, then keep closely related visible support entities if helpful.\n"
            "Avoid abstract concepts, pure attributes, colors, sizes, positions, directions, counts, pronouns, and generic filler unless no better visible entity exists.\n"
            "Keep the original candidate text exactly as written. Do not invent new phrases.\n"
            f'Return JSON only: {{"phrases": ["phrase_a", ...]}} with at most {limit} items.\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Example 1:\n"
            "Question: Where is the shampoo?\n"
            "Answer: The shampoo is on the dresser, to the left of the mirror.\n"
            "Candidates:\n"
            "0. shampoo\n1. dresser\n2. left\n3. mirror\n"
            'Output: {"phrases": ["shampoo", "dresser", "mirror"]}\n\n'
            "Example 2:\n"
            "Question: What color is the car?\n"
            "Answer: The car is red and parked beside a tree.\n"
            "Candidates:\n"
            "0. car\n1. red\n2. tree\n"
            'Output: {"phrases": ["car", "tree"]}\n\n'
            "Now solve the real case.\n"
            f"Question:\n{question}\n"
            f"Answer:\n{answer_text}\n"
            "Candidates:\n" + "\n".join(numbered) + "\n"
            f"Select up to {limit} phrases. Prefer visible entities most relevant to the question.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    encoded = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.qwen_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.qwen_device)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    outputs = model.qwen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=pad_id,
    )
    gen = outputs[0, input_ids.shape[-1] :]
    text = model.tokenizer.decode(gen, skip_special_tokens=True).strip()
    print(f"[Debug] Phrase rerank output: {text}")
    chosen: List[str] = []
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            raw_list = data.get("phrases", [])
            chosen = [s for s in raw_list if isinstance(s, str)]
        except json.JSONDecodeError:
            chosen = []
    if not chosen:
        return candidates[:limit]
    normalized_candidates = []
    for candidate in candidates:
        key = candidate[0].lower().strip()
        normalized_candidates.append((key, candidate))

    filtered: List[Tuple[str, Tuple[int, int]]] = []
    seen = set()
    for item in chosen:
        key = item.lower().strip()
        matched = None
        for cand_key, candidate in normalized_candidates:
            if cand_key == key:
                matched = candidate
                break
        if matched is None:
            for cand_key, candidate in normalized_candidates:
                if key and (key in cand_key or cand_key in key):
                    matched = candidate
                    break
        if matched is None:
            continue
        matched_key = matched[0].lower().strip()
        if matched_key in seen:
            continue
        filtered.append(matched)
        seen.add(matched_key)

    if not filtered:
        return candidates[:limit]
    return filtered[:limit]


def extract_phrases_via_model(
    model, question: str, answer_text: str, max_tokens: int, limit: int
) -> List[Tuple[str, Tuple[int, int]]]:
    question = str(question or "").strip()
    answer_text = str(answer_text or "")
    spacy_candidates = _extract_phrases_spacy(answer_text, limit=limit * 3)
    if spacy_candidates:
        return _rerank_phrases_with_model(
            model, question, answer_text, spacy_candidates, max_tokens, limit
        )
    override = getattr(model, "phrase_extract_prompt", None)
    if override:
        try:
            prompt = override.format(question=question, answer=answer_text)
        except Exception:
            prompt = f"{override}\n\nQuestion:\n{question}\n\nAnswer:\n{answer_text}\n"
    else:
        prompt = (
            "<|im_start|>system\n"
            "You extract distinct noun phrases from the assistant answer. "
            "Prefer concise phrases that refer to visible entities relevant to the user question. "
            'Return pure JSON like {"phrases": ["phrase a", ...]}.\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Question:\n"
            f"{question}\n"
            "Answer:\n"
            f"{answer_text}\n"
            "List concise noun phrases that appear in the answer and are best suited for visual grounding. JSON only.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    encoded = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.qwen_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.qwen_device)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    outputs = model.qwen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=pad_id,
    )
    gen = outputs[0, input_ids.shape[-1] :]
    text = model.tokenizer.decode(gen, skip_special_tokens=True).strip()
    print(f"[Debug] Phrase extraction output: {text}")
    match = re.search(r"\{.*\}", text, flags=re.S)
    phrases: List[str] = []
    if match:
        try:
            data = json.loads(match.group(0))
            raw_list = data.get("phrases", [])
            for item in raw_list:
                if isinstance(item, str):
                    phrases.append(item.strip())
        except json.JSONDecodeError:
            pass
    if not phrases:
        candidates = re.split(r"[\n,;/]+", text)
        phrases = [c.strip() for c in candidates if c.strip()]
    dedup: List[Tuple[str, Tuple[int, int]]] = []
    seen = set()
    for phrase in phrases:
        key = phrase.lower()
        if not key or key in seen:
            continue
        dedup.append((phrase, (-1, -1)))
        seen.add(key)
        if len(dedup) >= limit:
            break
    if dedup:
        return dedup

    regex_candidates = _extract_phrases_regex(answer_text, limit=limit * 3)
    if regex_candidates:
        print(f"[Debug] Regex fallback phrases: {[p[0] for p in regex_candidates]}")
        return _rerank_phrases_with_model(
            model, question, answer_text, regex_candidates, max_tokens, limit
        )
    return []


def build_offsets(tokenizer, answer_text: str) -> List[Tuple[int, int]]:
    encoded = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if "offset_mapping" in encoded:
        offsets = encoded["offset_mapping"][0].tolist()
    elif encoded.encodings:
        offsets = encoded.encodings[0].offsets
    else:
        raise ValueError("Tokenizer must provide offset_mapping")
    clean_offsets = []
    for start, end in offsets:
        if (start, end) == (0, 0) and clean_offsets:
            clean_offsets.append((clean_offsets[-1][1], clean_offsets[-1][1]))
        else:
            clean_offsets.append((int(start), int(end)))
    return clean_offsets


def char_to_token(offsets: Sequence[Tuple[int, int]], char_pos: int) -> int:
    for idx, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return idx
    return len(offsets) - 1


def build_phrase_candidates(
    answer_text: str,
    phrase_texts: List[Tuple[str, Tuple[int, int]]],
    offsets: List[Tuple[int, int]],
) -> List[PhraseCandidate]:
    lower_text = answer_text.lower()
    cursor = 0
    candidates: List[PhraseCandidate] = []
    for phrase, span in phrase_texts:
        raw = phrase.strip()
        if not raw:
            continue
        target = raw.lower()
        if span != (-1, -1):
            start, end = span
        else:
            start = lower_text.find(target, cursor)
            if start == -1:
                start = lower_text.find(target)
            if start == -1:
                continue
            end = start + len(raw)
            cursor = end
        token_start = char_to_token(offsets, start)
        token_end = char_to_token(offsets, max(start, end - 1)) + 1
        if token_end <= token_start:
            token_end = token_start + 1
        candidates.append(
            PhraseCandidate(
                text=raw, char_span=(start, end), token_span=(token_start, token_end)
            )
        )
    return candidates


def dedupe_phrase_candidates(
    candidates: List[PhraseCandidate],
) -> List[PhraseCandidate]:
    """Remove duplicate/contained phrase candidates while preserving display order."""
    if not candidates:
        return candidates
    indexed = list(enumerate(candidates))
    # Prefer longer spans; tie-break by earlier start to stabilize selection.
    indexed.sort(
        key=lambda item: (
            -max(0, item[1].char_span[1] - item[1].char_span[0]),
            item[1].char_span[0],
        )
    )
    kept_indices: set[int] = set()
    kept_spans: List[Tuple[int, int]] = []
    seen_text: set[str] = set()
    for idx, cand in indexed:
        text = (cand.text or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_text:
            continue
        start, end = cand.char_span
        if end <= start:
            continue
        contained = False
        for ks, ke in kept_spans:
            if start >= ks and end <= ke:
                contained = True
                break
        if contained:
            continue
        kept_indices.add(idx)
        kept_spans.append((start, end))
        seen_text.add(key)
    return [cand for idx, cand in enumerate(candidates) if idx in kept_indices]


def _find_first_span_ci(
    text: str, needle: str, occurrence: int = 0
) -> Optional[Tuple[int, int]]:
    if not text or not needle:
        return None
    hay = text.lower()
    ndl = needle.lower().strip()
    if not ndl:
        return None
    start = 0
    for _ in range(max(occurrence, 0) + 1):
        pos = hay.find(ndl, start)
        if pos == -1:
            return None
        start = pos + len(ndl)
    return pos, pos + len(needle)


def _clip_char_span(
    answer_text: str, char_span: Sequence[int]
) -> Optional[Tuple[int, int]]:
    if not answer_text or not char_span or len(char_span) < 2:
        return None
    try:
        cs, ce = int(char_span[0]), int(char_span[1])
    except Exception:
        return None
    max_len = len(answer_text)
    orig = (cs, ce)
    cs = max(0, min(cs, max_len))
    ce = max(0, min(ce, max_len))
    if (cs, ce) != orig:
        print(f"[Ground] Clip char_span {orig} -> {(cs, ce)}")
    if ce <= cs:
        return None
    return cs, ce


def token_span_to_char(
    offsets: Sequence[Tuple[int, int]], token_span: Sequence[int]
) -> Optional[Tuple[int, int]]:
    if not offsets or not token_span or len(token_span) < 2:
        return None
    try:
        ts, te = int(token_span[0]), int(token_span[1])
    except Exception:
        return None
    if te <= ts or ts < 0:
        return None
    te_idx = min(te - 1, len(offsets) - 1)
    ts_idx = min(ts, len(offsets) - 1)
    start = offsets[ts_idx][0]
    end = offsets[te_idx][1]
    return int(start), int(end)


def resolve_phrase_to_spans(
    answer_text: str,
    offsets: Sequence[Tuple[int, int]],
    phrase_text: str,
    *,
    char_span: Optional[Sequence[int]] = None,
    token_span: Optional[Sequence[int]] = None,
    occurrence: int = 0,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Resolve a phrase to (char_span, token_span).

    Priority:
    1) token_span if provided
    2) char_span if provided
    3) search phrase_text in answer_text (case-insensitive) and derive spans
    """
    if token_span is not None:
        try:
            ts, te = int(token_span[0]), int(token_span[1])
            if te > ts >= 0:
                cs = None
                if char_span is not None and len(char_span) >= 2:
                    cs = _clip_char_span(answer_text, char_span)
                if cs is None:
                    cs = token_span_to_char(offsets, (ts, te))
                if cs is None:
                    cs = (0, 0)
                return cs, (ts, te)
        except Exception:
            pass

    if char_span is not None and len(char_span) >= 2:
        clipped = _clip_char_span(answer_text, char_span)
        if clipped is not None:
            cs, ce = clipped
            token_start = char_to_token(offsets, cs)
            token_end = char_to_token(offsets, max(cs, ce - 1)) + 1
            if token_end <= token_start:
                token_end = token_start + 1
            return (cs, ce), (int(token_start), int(token_end))

    found = _find_first_span_ci(answer_text, phrase_text, occurrence=occurrence)
    if not found:
        return None
    cs, ce = found
    token_start = char_to_token(offsets, cs)
    token_end = char_to_token(offsets, max(cs, ce - 1)) + 1
    if token_end <= token_start:
        token_end = token_start + 1
    return (int(cs), int(ce)), (int(token_start), int(token_end))


def perform_ground_custom(
    session: SessionState, items: List[Dict[str, Any]]
) -> List[GroundRecord]:
    """Ground user-selected phrases without changing the default rerank pipeline.

    Each item may contain:
      - text / phrase (required unless token_span given)
      - char_span: [start, end] (optional)
      - token_span: [start, end] (optional)
    """
    ensure_image(session)
    if session.last_answer is None:
        print("[Ground] Ask a question first.")
        return []
    if not items:
        print("[Ground] Provide phrases to ground.")
        return []

    answer_text = session.last_answer.get("output_text") or ""
    offsets = session.token_offsets
    if offsets is None:
        offsets = build_offsets(session.model.tokenizer, answer_text)
        session.token_offsets = offsets

    positive_ids: List[Tuple[int, int]] = []
    labels: List[str] = []
    resolved_char_spans: List[Tuple[int, int]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        phrase_text = str(item.get("text") or item.get("phrase") or "").strip()
        cs_in = item.get("char_span")
        ts_in = item.get("token_span")
        has_char_span = bool(cs_in) and isinstance(cs_in, (list, tuple))
        has_token_span = bool(ts_in) and isinstance(ts_in, (list, tuple))
        occ_in = item.get("occurrence")
        occurrence = 0
        if isinstance(occ_in, int):
            occurrence = occ_in
        else:
            try:
                occurrence = int(occ_in) if occ_in is not None else 0
            except Exception:
                occurrence = 0
        resolved = resolve_phrase_to_spans(
            answer_text,
            offsets,
            phrase_text,
            char_span=cs_in,
            token_span=ts_in,
            occurrence=occurrence,
        )
        if not resolved:
            if phrase_text and not has_char_span and not has_token_span:
                raise ValueError("Text not found in answer.")
            continue
        cs, ts = resolved
        positive_ids.append(ts)
        labels.append(phrase_text or answer_text[cs[0] : cs[1]])
        resolved_char_spans.append(cs)

    if not positive_ids:
        print("[Ground] No valid phrases resolved from payload.")
        return []

    pred_masks, sam_masks = session.model.ground(
        image=session.current_image,
        positive_ids=positive_ids,
        hidden_states=session.last_answer["hidden_states"],
        attention_maps=session.last_answer["attention_maps"],
        meta_data=session.last_answer["meta_data"],
        use_sam=not session.args.no_sam,
    )
    mask_tensor = sam_masks if not session.args.no_sam else pred_masks
    masks = mask_tensor.detach().cpu().numpy()
    records = save_ground_outputs(session.current_image, masks, labels, session)
    session.ground_id += 1
    for rec, span, cs in zip(records, positive_ids, resolved_char_spans):
        rec.token_span = span
        rec.char_span = cs
    session.last_records = records
    return records


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1 + 1, y1 + 1


def dominant_connected_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return binary

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_coords: List[Tuple[int, int]] = []

    ys, xs = np.where(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[Tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        coords: List[Tuple[int, int]] = []
        while queue:
            y, x = queue.pop()
            coords.append((y, x))
            y0 = max(0, y - 1)
            y1 = min(height, y + 2)
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(coords) > len(best_coords):
            best_coords = coords

    if not best_coords:
        return binary
    component = np.zeros_like(binary, dtype=bool)
    ys_best, xs_best = zip(*best_coords)
    component[np.array(ys_best), np.array(xs_best)] = True
    return component


def blend_mask(
    image_np: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]
) -> np.ndarray:
    overlay = image_np.copy()
    overlay_mask = mask.astype(bool)
    if overlay_mask.sum() == 0:
        return image_np.copy()
    overlay[overlay_mask] = (
        overlay[overlay_mask] * 0.35 + np.array(color, dtype=np.float32) * 0.65
    )
    return overlay


def save_ground_outputs(
    image: Image.Image, masks: np.ndarray, labels: List[str], session: SessionState
) -> List[GroundRecord]:
    run_dir = session.session_paths.ground_dir(session.turn_idx, session.ground_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_np = np.array(image).astype(np.float32)
    records: List[GroundRecord] = []
    for idx, mask in enumerate(masks):
        binary = mask > 0
        primary_binary = dominant_connected_component(binary)
        mask_img = Image.fromarray((binary * 255).astype(np.uint8))
        mask_path = run_dir / f"mask_{idx:02d}.png"
        mask_img.save(mask_path)
        overlay_np = blend_mask(base_np, binary, colors[idx % len(colors)])
        overlay_img = Image.fromarray(overlay_np.astype(np.uint8))
        overlay_path = run_dir / f"overlay_{idx:02d}.png"
        overlay_img.save(overlay_path)
        bbox = mask_to_box(primary_binary)
        if bbox is None:
            bbox = mask_to_box(binary)
        roi_path = None
        roi_image = None
        if bbox:
            roi_image = image.crop(bbox)
            roi_path = run_dir / f"roi_{idx:02d}.png"
            roi_image.save(roi_path)
        phrase_text = labels[idx] if idx < len(labels) else ""
        records.append(
            GroundRecord(
                overlay_path=overlay_path,
                mask_path=mask_path,
                roi_path=roi_path,
                phrase_text=phrase_text,
                token_span=(0, 0),
                char_span=(0, 0),
                roi_image=roi_image,
                bbox=bbox,
            )
        )
    combined = blend_mask(base_np, masks.sum(axis=0) > 0, (255, 128, 0))
    Image.fromarray(combined.astype(np.uint8)).save(run_dir / "summary.png")
    return records


def ensure_image(session: SessionState):
    if session.current_image is None:
        raise RuntimeError("Load an image first with `load <path>`.")


def infer_min_image_side(model: Any) -> Optional[int]:
    patch_size = getattr(model, "patch_size", None)
    merge_size = getattr(model, "merge_size", None)
    if patch_size:
        return int(patch_size * (merge_size or 1))
    processor = getattr(model, "processor", None)
    if processor is not None and hasattr(processor, "image_processor"):
        image_processor = processor.image_processor
        patch_size = getattr(image_processor, "patch_size", None)
        if patch_size:
            merge_size = getattr(image_processor, "merge_size", None)
            return int(patch_size * (merge_size or 1))
    return None


def ensure_min_image_size(
    image: Image.Image, min_side: int
) -> Tuple[Image.Image, bool]:
    if min_side <= 0:
        return image, False
    width, height = image.size
    if width >= min_side and height >= min_side:
        return image, False
    scale = max(min_side / width, min_side / height)
    new_width = max(min_side, int(round(width * scale)))
    new_height = max(min_side, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.BICUBIC)
    return resized, True


def handle_load(session: SessionState, path: str):
    image, resolved = load_image(path)
    target = session.session_paths.image_path(Path(resolved).name)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if Path(resolved).resolve() != target.resolve():
            shutil.copy2(resolved, target)
    except Exception:
        # Fallback to saving via PIL if copy fails (e.g., remote streams)
        image.save(target)
    session.current_image = image
    session.current_image_path = target
    session.reset_answer()
    print(f"[Load] Image loaded: {session.current_image_path.name} {image.size}")


def handle_ask(session: SessionState, question: str):
    try:
        ask_cmd = parse_ask_command(question)
    except ValueError as exc:
        print(f"[Ask] {exc}")
        return
        return
    try:
        result = pipeline_default_ask(
            session, ask_cmd.question, reset_history=ask_cmd.reset_history, auto_topk=1
        )
    except Exception as exc:
        print(f"[Ask] Failed: {exc}")
        return
    answer_text = result.get("answer", "")
    print(f"[Answer]\n{answer_text}\n")
    turns = history_turns(session)
    if turns > 0:
        print(f"[Ask] 当前上下文包含 {turns} 轮对话。")
    candidates = result.get("phrases", [])
    if not candidates:
        print("[Ask] No candidate phrases detected.")
    else:
        print("[Phrases]")
        for idx, cand in enumerate(candidates):
            start, end = cand.char_span
            snippet = answer_text[start:end]
            print(f'  {idx}: "{snippet}" chars[{start}:{end}] tokens{cand.token_span}')
    if result.get("roi_answer"):
        print("[Ask] ROI re-answer used.")


def handle_ground(session: SessionState, indices: List[int]):
    records = perform_ground(session, indices)
    for idx, rec in enumerate(records):
        roi_info = f", ROI: {rec.roi_path.name}" if rec.roi_path else ""
        print(
            f"[Ground] #{idx} mask saved to {rec.overlay_path} (mask: {rec.mask_path.name}{roi_info})"
        )


def _first_bbox_record(
    records: List[GroundRecord],
) -> Optional[Tuple[int, int, int, int]]:
    for rec in records:
        if rec.bbox and len(rec.bbox) == 4:
            return tuple(int(v) for v in rec.bbox)
    return None


def perform_ground(session: SessionState, indices: List[int]) -> List[GroundRecord]:
    ensure_image(session)
    if session.last_answer is None:
        print("[Ground] Ask a question first.")
        return []
    if not indices:
        print("[Ground] Provide phrase indices, e.g., `ground 0 1`.")
        return []
    phrases = session.phrases
    if not phrases:
        print("[Ground] No phrase candidates available.")
        return []
    positive_ids = []
    labels = []
    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(phrases):
            print(f"[Ground] Invalid index {idx}.")
            return []
        candidate = phrases[idx]
        positive_ids.append(candidate.token_span)
        labels.append(candidate.text)
        selected.append(candidate)
    pred_masks, sam_masks = session.model.ground(
        image=session.current_image,
        positive_ids=positive_ids,
        hidden_states=session.last_answer["hidden_states"],
        attention_maps=session.last_answer["attention_maps"],
        meta_data=session.last_answer["meta_data"],
        use_sam=not session.args.no_sam,
    )
    mask_tensor = sam_masks if not session.args.no_sam else pred_masks
    masks = mask_tensor.detach().cpu().numpy()
    records = save_ground_outputs(session.current_image, masks, labels, session)
    session.ground_id += 1
    for rec, span, candidate in zip(records, positive_ids, selected):
        rec.token_span = span
        rec.char_span = candidate.char_span
    session.last_records = records
    return records


def pipeline_default_ask(
    session: SessionState,
    question: str,
    reset_history: bool = False,
    auto_topk: int = 1,
    turn_idx: Optional[int] = None,
    enable_roi: bool = True,
) -> Dict[str, Any]:
    """Unified pipeline: answer -> auto ground -> ROI re-answer (DeepSeek-style)."""
    ensure_image(session)
    if reset_history:
        clear_history(session)
    resolved_turn_idx = history_turns(session) if turn_idx is None else int(turn_idx)
    session.turn_idx = resolved_turn_idx
    session.ground_id = 0
    session.attn_counters = {
        ATTN_KIND_TOKEN_TO_REGION: 0,
        ATTN_KIND_REGION_TO_TOKEN: 0,
    }
    session.session_paths.turn_dir(session.turn_idx).mkdir(parents=True, exist_ok=True)
    if not question:
        raise ValueError("Question must not be empty.")
    output = session.model.answer(
        image=session.current_image,
        question=question,
        history=session.history,
        max_new_tokens=session.args.max_new_tokens,
    )
    answer_text = output["output_text"]
    raw_answer_artifacts = detect_answer_artifacts(answer_text)
    display_raw_answer = clean_answer_for_display(answer_text) or answer_text
    offsets = build_offsets(session.model.tokenizer, answer_text)
    print(f"[Debug]  anwer_text: {answer_text}")
    phrase_texts = extract_phrases_via_model(
        session.model,
        question,
        answer_text,
        session.args.phrase_max_tokens,
        session.args.max_phrases,
    )
    print(f"[Debug] Extracted phrases: {[p[0] for p in phrase_texts]}")
    candidates = build_phrase_candidates(answer_text, phrase_texts, offsets)
    candidates = dedupe_phrase_candidates(candidates)
    print(f"[Debug] Built {len(candidates)} phrase candidates.")
    session.last_answer = output
    session.token_offsets = offsets
    session.phrases = candidates
    session.last_records = []
    final_answer = display_raw_answer
    verification: Dict[str, Any] = {
        "original_answer": answer_text,
        "raw_answer_cleaned": display_raw_answer,
        "raw_answer_artifacts": raw_answer_artifacts,
        "used": False,
    }
    if auto_topk > 0 and candidates:
        indices = list(range(min(auto_topk, len(candidates))))
        try:
            records = perform_ground(session, indices)
            bbox = _first_bbox_record(records)
            verification.update(
                {
                    "used": bool(records) if enable_roi else False,
                }
            )
            if enable_roi and bbox:
                try:
                    roi_result = session.model.visual_cot_resample(
                        image=session.current_image,
                        question=question,
                        bbox=bbox,
                        answer_cache=session.last_answer,
                        max_new_tokens=session.args.max_new_tokens,
                        extra_prompt=getattr(session.model, "roi_extra_prompt", ""),
                    )
                    roi_answer = roi_result.get("answer_text", "")
                    roi_answer_artifacts = detect_answer_artifacts(roi_answer)
                    roi_answer_cleaned = clean_answer_for_display(roi_answer) or roi_answer
                    if roi_answer:
                        verification.update(
                            {
                                "roi_answer": roi_answer,
                                "roi_answer_cleaned": roi_answer_cleaned,
                                "roi_answer_artifacts": roi_answer_artifacts,
                                "roi_bbox": roi_result.get("roi_bbox", bbox),
                                "roi_prompt": roi_result.get("prompt"),
                            }
                        )
                        if roi_answer_artifacts and not raw_answer_artifacts and display_raw_answer:
                            verification["roi_answer_accepted"] = False
                            verification["roi_rejected_reason"] = "answer_artifact"
                        else:
                            final_answer = roi_answer_cleaned
                            verification["roi_answer_accepted"] = True
                except Exception as exc:
                    verification["error"] = str(exc)
        except Exception as exc:
            verification["error"] = str(exc)
            verification["used"] = False
    append_history_entry(session, "user", question)
    append_history_entry(session, "assistant", final_answer)
    return {
        "answer": final_answer,
        "raw_answer": display_raw_answer,
        "original_answer": answer_text,
        "roi_answer": verification.get("roi_answer"),
        "roi_bbox": verification.get("roi_bbox"),
        "phrases": candidates,
        "verification": verification,
    }


def handle_inspect(session: SessionState, idx: int, question: Optional[str]):
    print("[Inspect] Legacy ROI inspect removed; use default ask flow.")


def handle_cot_resample(session: SessionState, idx: int, question: str):
    print("[CoT] Legacy CoT resample removed; use default ask flow.")


def print_help():
    print("Commands:")
    print("  load <image_path>        Load image for the current session.")
    print(
        "  ask <question>           Ask Qwen (pipeline: answer -> auto ground -> ROI re-answer)."
    )
    print("  ask --reset-history ...  Clear context before asking the next question.")
    print(
        "  ground <idx ...>         Ground one or more phrase indices from the last answer."
    )
    print("  clear                    Clear stored multi-turn conversation context.")
    print("  help                     Show this message.")
    print("  exit / quit              Terminate the demo.")


def main():
    args = parse_args()
    cfg_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(REPO_ROOT, args.config)
    )
    cfg = Config.fromfile(cfg_path)
    model = load_model(cfg, args)
    session = SessionState(
        model=model,
        args=args,
        result_root=Path(args.results_dir).expanduser().resolve(),
    )
    if args.image:
        try:
            handle_load(session, args.image)
        except Exception as exc:
            print(f"[Load] Failed: {exc}")
    print_help()
    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            print("[Exit]")
            break
        cmd, *rest = raw.split(" ", 1)
        cmd = cmd.lower()
        arg_str = rest[0].strip() if rest else ""
        try:
            if cmd == "help":
                print_help()
            elif cmd == "load":
                if not arg_str:
                    print("[Load] Provide image path.")
                else:
                    handle_load(session, arg_str)
            elif cmd == "ask":
                handle_ask(session, arg_str)
            elif cmd == "ground":
                if not arg_str:
                    print("[Ground] Provide indices.")
                else:
                    try:
                        indices = [int(tok) for tok in shlex.split(arg_str)]
                    except ValueError:
                        print("[Ground] Indices must be integers.")
                        continue
                    handle_ground(session, indices)
            elif cmd == "clear":
                clear_history(session)
            else:
                print(f'[Warn] Unknown command "{cmd}". Type `help` for usage.')
        except Exception as exc:
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()
