#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.utils import colors  # noqa: E402


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


@dataclass
class SessionState:
    model: Any
    args: argparse.Namespace
    result_root: Path
    current_image: Optional[Image.Image] = None
    current_image_path: Optional[Path] = None
    last_answer: Optional[dict] = None
    phrases: List[PhraseCandidate] = field(default_factory=list)
    token_offsets: Optional[List[Tuple[int, int]]] = None
    ground_counter: int = 0
    last_records: List[GroundRecord] = field(default_factory=list)

    def __post_init__(self):
        self.session_dir = self.result_root / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def reset_answer(self):
        self.last_answer = None
        self.phrases = []
        self.token_offsets = None
        self.last_records = []


def parse_device_map_arg(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    normalized = raw_value.strip()
    if not normalized or normalized.lower() in {'none', 'single', 'disable', 'disabled'}:
        return None
    return normalized


def parse_max_memory_arg(raw_value: Optional[str]) -> Optional[Dict[str, str]]:
    if not raw_value:
        return None
    entries = [chunk.strip() for chunk in raw_value.split(',') if chunk.strip()]
    if not entries:
        return None
    limits: Dict[str, str] = {}
    for item in entries:
        if ':' not in item:
            raise ValueError(f'Invalid max-memory entry "{item}". Use format "0:20GiB,1:20GiB".')
        key, value = item.split(':', 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f'Invalid max-memory entry "{item}".')
        limits[key] = value
    return limits or None


def move_auxiliary_modules(model, device: torch.device):
    for attr in ('mask_head', 'sam', 'text_proj'):
        module = getattr(model, attr, None)
        if module is not None:
            module.to(device)
    if hasattr(model, 'text_layer_weights'):
        model.text_layer_weights = torch.nn.Parameter(model.text_layer_weights.to(device))


def infer_qwen_primary_device(qwen_model, fallback_device: torch.device) -> torch.device:
    existing = getattr(qwen_model, 'device', None)
    if existing is not None:
        try:
            device = torch.device(existing)
            qwen_model.device = device
            return device
        except (TypeError, RuntimeError):
            pass
    device_map = getattr(qwen_model, 'hf_device_map', None)
    if isinstance(device_map, dict) and device_map:
        first_target = next(iter(device_map.values()))
        try:
            device = torch.device(first_target)
        except (TypeError, RuntimeError):
            device = fallback_device
    else:
        device = fallback_device
    qwen_model.device = device
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive Qwen-FLMM demo: ask → ground → inspect.')
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py',
        help='Path to Qwen config (default: 7B Qwen2.5-VL).')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--device', default='cuda', help='Device for inference.')
    parser.add_argument('--device-map', default='auto',
                        help='Device map for the Qwen base model. '
                             'Use "auto" to shard across visible GPUs or "none" to disable.')
    parser.add_argument('--device-max-memory', default=None,
                        help='Optional per-device memory hints when using --device-map. '
                             'Format: "0:20GiB,1:20GiB".')
    parser.add_argument('--image', default=None, help='Optional image to preload.')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--phrase-max-tokens', type=int, default=64)
    parser.add_argument('--max-phrases', type=int, default=6)
    parser.add_argument('--results-dir', default='scripts/demo/results/qwen')
    parser.add_argument('--inspect-prompt', default='Describe this region in detail.')
    parser.add_argument('--no-sam', action='store_true', help='Skip SAM refinement.')
    parser.add_argument('--extra-prompt', default='',
                        help='Force append text to every question (default: empty).')
    return parser.parse_args()


def load_model(cfg, args):
    device_map = parse_device_map_arg(args.device_map)
    max_memory = parse_max_memory_arg(args.device_max_memory) if device_map else None
    model_cfg = cfg.model
    qwen_cfg = model_cfg.get('model', None)
    if device_map and qwen_cfg is not None:
        qwen_cfg['device_map'] = device_map
        if max_memory:
            qwen_cfg['max_memory'] = max_memory
    model = BUILDER.build(cfg.model)
    if args.checkpoint:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'[Model] Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')
    target_device = torch.device(args.device)
    if device_map:
        move_auxiliary_modules(model, target_device)
        qwen_device = infer_qwen_primary_device(model.qwen_model, target_device)
        print(f'[Model] Using device_map="{device_map}" (primary device: {qwen_device})')
    else:
        model = model.to(target_device)
        qwen_device = target_device
    model = model.eval()
    model.qwen_model.device = qwen_device
    append_prompt = args.extra_prompt or ''
    model._prepare_for_generation(
        image_processor=cfg.image_processor,
        prompt_template=cfg.prompt_template,
        max_new_tokens=args.max_new_tokens,
        additional_prompt=append_prompt,
    )
    return model


def load_image(path_str: str) -> Image.Image:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert('RGB')
    return image


def extract_phrases_via_model(model, answer_text: str, max_tokens: int, limit: int) -> List[str]:
    prompt = (
        "<|im_start|>system\n"
        "You extract distinct noun phrases from the assistant answer. "
        "Return pure JSON like {\"phrases\": [\"phrase a\", ...]}.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Answer:\n"
        f"{answer_text}\n"
        "List concise noun phrases that appear in the answer. JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    encoded = model.tokenizer(
        prompt, return_tensors='pt', add_special_tokens=False)
    input_ids = encoded['input_ids'].to(model.qwen_model.device)
    attention_mask = encoded.get('attention_mask')
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.qwen_model.device)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    outputs = model.qwen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=pad_id,
    )
    gen = outputs[0, input_ids.shape[-1]:]
    text = model.tokenizer.decode(gen, skip_special_tokens=True).strip()
    match = re.search(r'\{.*\}', text, flags=re.S)
    phrases: List[str] = []
    if match:
        try:
            data = json.loads(match.group(0))
            raw_list = data.get('phrases', [])
            for item in raw_list:
                if isinstance(item, str):
                    phrases.append(item.strip())
        except json.JSONDecodeError:
            pass
    if not phrases:
        candidates = re.split(r'[\n,;/]+', text)
        phrases = [c.strip() for c in candidates if c.strip()]
    dedup = []
    seen = set()
    for phrase in phrases:
        key = phrase.lower()
        if not key or key in seen:
            continue
        dedup.append(phrase)
        seen.add(key)
        if len(dedup) >= limit:
            break
    return dedup


def build_offsets(tokenizer, answer_text: str) -> List[Tuple[int, int]]:
    encoded = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors='pt')
    if 'offset_mapping' in encoded:
        offsets = encoded['offset_mapping'][0].tolist()
    elif encoded.encodings:
        offsets = encoded.encodings[0].offsets
    else:
        raise ValueError('Tokenizer must provide offset_mapping')
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


def build_phrase_candidates(answer_text: str,
                            phrase_texts: List[str],
                            offsets: List[Tuple[int, int]]) -> List[PhraseCandidate]:
    lower_text = answer_text.lower()
    cursor = 0
    candidates: List[PhraseCandidate] = []
    for phrase in phrase_texts:
        raw = phrase.strip()
        if not raw:
            continue
        target = raw.lower()
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
        candidates.append(PhraseCandidate(
            text=raw,
            char_span=(start, end),
            token_span=(token_start, token_end)))
    return candidates



def mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1 + 1, y1 + 1


def blend_mask(image_np: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    overlay = image_np.copy()
    overlay_mask = mask.astype(bool)
    if overlay_mask.sum() == 0:
        return image_np.copy()
    overlay[overlay_mask] = (
        overlay[overlay_mask] * 0.35 + np.array(color, dtype=np.float32) * 0.65)
    return overlay


def save_ground_outputs(image: Image.Image,
                        masks: np.ndarray,
                        labels: List[str],
                        session: SessionState) -> List[GroundRecord]:
    run_dir = session.session_dir / f'round_{session.ground_counter:02d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    base_np = np.array(image).astype(np.float32)
    records: List[GroundRecord] = []
    for idx, mask in enumerate(masks):
        binary = mask > 0
        mask_img = Image.fromarray((binary * 255).astype(np.uint8))
        mask_path = run_dir / f'mask_{idx:02d}.png'
        mask_img.save(mask_path)
        overlay_np = blend_mask(base_np, binary, colors[idx % len(colors)])
        overlay_img = Image.fromarray(overlay_np.astype(np.uint8))
        overlay_path = run_dir / f'overlay_{idx:02d}.png'
        overlay_img.save(overlay_path)
        bbox = mask_to_box(binary)
        roi_path = None
        roi_image = None
        if bbox:
            roi_image = image.crop(bbox)
            roi_path = run_dir / f'roi_{idx:02d}.png'
            roi_image.save(roi_path)
        phrase_text = labels[idx] if idx < len(labels) else ''
        records.append(GroundRecord(
            overlay_path=overlay_path,
            mask_path=mask_path,
            roi_path=roi_path,
            phrase_text=phrase_text,
            token_span=(0, 0),
            char_span=(0, 0),
            roi_image=roi_image))
    combined = blend_mask(base_np, masks.sum(axis=0) > 0, (255, 128, 0))
    Image.fromarray(combined.astype(np.uint8)).save(run_dir / 'summary.png')
    return records


def ensure_image(session: SessionState):
    if session.current_image is None:
        raise RuntimeError('Load an image first with `load <path>`.')


def handle_load(session: SessionState, path: str):
    image = load_image(path)
    session.current_image = image
    session.current_image_path = Path(path).expanduser().resolve()
    session.reset_answer()
    print(f'[Load] Image loaded: {session.current_image_path.name} {image.size}')


def handle_ask(session: SessionState, question: str):
    ensure_image(session)
    if not question.strip():
        print('[Ask] Question must not be empty.')
        return
    print('[Ask] Running Qwen...')
    output = session.model.answer(
        image=session.current_image,
        question=question,
        max_new_tokens=session.args.max_new_tokens)
    answer_text = output['output_text']
    print(f'[Answer]\n{answer_text}\n')
    offsets = build_offsets(session.model.tokenizer, answer_text)
    phrase_texts = extract_phrases_via_model(
        session.model, answer_text, session.args.phrase_max_tokens, session.args.max_phrases)
    candidates = build_phrase_candidates(answer_text, phrase_texts, offsets)
    session.last_answer = output
    session.token_offsets = offsets
    session.phrases = candidates
    session.last_records = []
    if not candidates:
        print('[Ask] No candidate phrases detected.')
    else:
        print('[Phrases]')
        for idx, cand in enumerate(candidates):
            start, end = cand.char_span
            snippet = answer_text[start:end]
            print(f'  {idx}: "{snippet}" chars[{start}:{end}] tokens{cand.token_span}')


def handle_ground(session: SessionState, indices: List[int]):
    ensure_image(session)
    if session.last_answer is None:
        print('[Ground] Ask a question first.')
        return
    if not indices:
        print('[Ground] Provide phrase indices, e.g., `ground 0 1`.')
        return
    phrases = session.phrases
    if not phrases:
        print('[Ground] No phrase candidates available.')
        return
    positive_ids = []
    labels = []
    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(phrases):
            print(f'[Ground] Invalid index {idx}.')
            return
        candidate = phrases[idx]
        positive_ids.append(candidate.token_span)
        labels.append(candidate.text)
        selected.append(candidate)
    print(f'[Ground] Extracting masks for {labels} ...')
    pred_masks, sam_masks = session.model.ground(
        image=session.current_image,
        positive_ids=positive_ids,
        hidden_states=session.last_answer['hidden_states'],
        attention_maps=session.last_answer['attention_maps'],
        meta_data=session.last_answer['meta_data'],
        use_sam=not session.args.no_sam)
    mask_tensor = sam_masks if not session.args.no_sam else pred_masks
    masks = mask_tensor.detach().cpu().numpy()
    session.ground_counter += 1
    records = save_ground_outputs(session.current_image, masks, labels, session)
    for rec, span, candidate in zip(records, positive_ids, selected):
        rec.token_span = span
        rec.char_span = candidate.char_span
    session.last_records = records
    for idx, rec in enumerate(records):
        roi_info = f', ROI: {rec.roi_path.name}' if rec.roi_path else ''
        print(f'[Ground] #{idx} mask saved to {rec.overlay_path} (mask: {rec.mask_path.name}{roi_info})')


def handle_inspect(session: SessionState, idx: int, question: Optional[str]):
    ensure_image(session)
    if not session.last_records:
        print('[Inspect] Run `ground` first.')
        return
    if idx < 0 or idx >= len(session.last_records):
        print(f'[Inspect] Invalid mask index {idx}.')
        return
    record = session.last_records[idx]
    roi = record.roi_image
    if roi is None:
        print('[Inspect] Selected mask has no ROI (empty mask).')
        return
    prompt = question.strip() if question else session.args.inspect_prompt
    print(f'[Inspect] Asking on ROI with prompt: {prompt}')
    output = session.model.answer(
        image=roi,
        question=prompt,
        max_new_tokens=session.args.max_new_tokens)
    print(f'[Inspect #{idx}] {output["output_text"]}')


def print_help():
    print('Commands:')
    print('  load <image_path>        Load image for the current session.')
    print('  ask <question>           Ask Qwen about the loaded image.')
    print('  ground <idx ...>         Ground one or more phrase indices from the last answer.')
    print('  inspect <idx> [prompt]   Run answer() on ROI cropped from mask idx.')
    print('  help                     Show this message.')
    print('  exit / quit              Terminate the demo.')


def main():
    args = parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    cfg = Config.fromfile(cfg_path)
    model = load_model(cfg, args)
    session = SessionState(
        model=model,
        args=args,
        result_root=Path(args.results_dir).expanduser().resolve())
    if args.image:
        try:
            handle_load(session, args.image)
        except Exception as exc:
            print(f'[Load] Failed: {exc}')
    print_help()
    while True:
        try:
            raw = input('>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n[Exit]')
            break
        if not raw:
            continue
        if raw.lower() in ('exit', 'quit'):
            print('[Exit]')
            break
        cmd, *rest = raw.split(' ', 1)
        cmd = cmd.lower()
        arg_str = rest[0].strip() if rest else ''
        try:
            if cmd == 'help':
                print_help()
            elif cmd == 'load':
                if not arg_str:
                    print('[Load] Provide image path.')
                else:
                    handle_load(session, arg_str)
            elif cmd == 'ask':
                handle_ask(session, arg_str)
            elif cmd == 'ground':
                if not arg_str:
                    print('[Ground] Provide indices.')
                else:
                    try:
                        indices = [int(tok) for tok in shlex.split(arg_str)]
                    except ValueError:
                        print('[Ground] Indices must be integers.')
                        continue
                    handle_ground(session, indices)
            elif cmd == 'inspect':
                if not arg_str:
                    print('[Inspect] Usage: inspect <idx> [prompt]')
                    continue
                tokens = shlex.split(arg_str)
                if not tokens:
                    print('[Inspect] Usage: inspect <idx> [prompt]')
                    continue
                try:
                    mask_idx = int(tokens[0])
                except ValueError:
                    print('[Inspect] First argument must be an integer index.')
                    continue
                question = ' '.join(tokens[1:]) if len(tokens) > 1 else ''
                handle_inspect(session, mask_idx, question)
            else:
                print(f'[Warn] Unknown command "{cmd}". Type `help` for usage.')
        except Exception as exc:
            print(f'[Error] {exc}')


if __name__ == '__main__':
    main()
