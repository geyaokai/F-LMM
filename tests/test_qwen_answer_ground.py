#!/usr/bin/env python3
import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config  # noqa: E402
from xtuner.registry import BUILDER  # noqa: E402
from xtuner.model.utils import guess_load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FrozenQwenSAM.answer() -> ground() on one sample.')
    parser.add_argument(
        '--config',
        default='configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py'
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--question', default='Where is the object you are referring to?')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--span-length', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--skip-batches', type=int, default=0)
    parser.add_argument('--sample-offset', type=int, default=0)
    parser.add_argument('--no-sam', action='store_true')
    parser.add_argument('--mask-index', type=int, default=0)
    return parser.parse_args()


def build_dataset(cfg_dict):
    dataset_cfg = deepcopy(cfg_dict)
    dataset_type = dataset_cfg.pop('type')
    if callable(dataset_type):
        return dataset_type(**dataset_cfg)
    dataset_cfg['type'] = dataset_type
    return BUILDER.build(dataset_cfg)


def build_collate_fn(cfg_dict):
    if cfg_dict is None:
        return None
    collate_cfg = deepcopy(cfg_dict)
    fn = collate_cfg.pop('type')
    if callable(fn) and not collate_cfg:
        return fn
    if callable(fn):
        def wrapper(instances):
            return fn(instances, **collate_cfg)
        return wrapper
    raise TypeError('collate_fn type must be callable.')


def build_dataloader(cfg_dict, dataset):
    loader_cfg = deepcopy(cfg_dict)
    batch_size = loader_cfg.pop('batch_size', 1)
    num_workers = loader_cfg.pop('num_workers', 0)
    collate_fn = build_collate_fn(loader_cfg.pop('collate_fn', None))
    pin_memory = loader_cfg.pop('pin_memory', False)
    prefetch_factor = loader_cfg.pop('prefetch_factor', None)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=loader_cfg.get('persistent_workers', False),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def select_sample(batch, offset):
    instances = batch['data'] if isinstance(batch, dict) and 'data' in batch else batch
    if offset >= len(instances):
        raise IndexError(f'sample_offset {offset} out of range for batch length {len(instances)}')
    return instances[offset]


def maybe_override_loader_cfg(cfg, args):
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train_dataloader.num_workers = args.num_workers


def compute_bbox(mask: torch.Tensor):
    mask = mask > 0
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    x0 = int(xs.min().item())
    x1 = int(xs.max().item() + 1)
    y0 = int(ys.min().item())
    y1 = int(ys.max().item() + 1)
    return x0, y0, x1, y1


def main():
    args = parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.train_dataloader.dataset)
    maybe_override_loader_cfg(cfg, args)
    dataloader = build_dataloader(cfg.train_dataloader, dataset)
    iterator = iter(dataloader)
    for _ in range(args.skip_batches):
        next(iterator)
    batch = next(iterator)
    sample = select_sample(batch, args.sample_offset)
    image = sample['image']
    image_grid = sample['image_grid_thw']
    print('image_grid_thw:', image_grid)
    model = BUILDER.build(cfg.model)
    if args.checkpoint:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}')
    model = model.to(args.device).eval()
    model._prepare_for_generation(
        image_processor=cfg.image_processor,
        prompt_template=cfg.prompt_template,
        max_new_tokens=args.max_new_tokens,
        additional_prompt=cfg.get('prompt', ''),
    )
    with torch.no_grad():
        answer_output = model.answer(
            image=image, question=args.question, max_new_tokens=args.max_new_tokens)
    hidden_states = answer_output['hidden_states']
    # attention_maps shape: (gen_len, num_layers, num_heads, seq_len, seq_len)
    attention_maps = answer_output['attention_maps']
    meta_data = answer_output['meta_data']
    print('Answer text:', answer_output['output_text'])
    print('hidden_states:', tuple(hidden_states.shape))
    print('attention_maps:', tuple(attention_maps.shape))
    print('meta_data:', meta_data)
    vision_tokens = answer_output.get('vision_tokens')
    if vision_tokens is not None:
        print('vision_tokens:', tuple(vision_tokens.shape))
    else:
        print('vision_tokens: None')

    gen_len = hidden_states.shape[0]
    span = min(args.span_length, gen_len)
    assert span > 0, 'Generation produced zero tokens.'
    positive_ids = [(0, span)]
    with torch.no_grad():
        pred_masks, sam_masks = model.ground(
            image=image,
            positive_ids=positive_ids,
            hidden_states=hidden_states,
            attention_maps=attention_maps,
            meta_data=meta_data,
            use_sam=not args.no_sam,
        )
    print('pred_masks:', tuple(pred_masks.shape))
    print('sam_masks:', tuple(sam_masks.shape))
    masks = sample['masks']
    if isinstance(masks, torch.Tensor):
        mask_tensor = masks
    else:
        mask_tensor = torch.as_tensor(masks)
    mask_idx = min(args.mask_index, mask_tensor.shape[0] - 1)
    bbox = compute_bbox(mask_tensor[mask_idx])
    if bbox is None:
        print('BBox: None (empty mask)')
    else:
        x0, y0, x1, y1 = bbox
        print(f'BBox (mask {mask_idx}): {(x0, y0, x1, y1)}')
        grid_tensor = image_grid[0] if image_grid.dim() == 2 else image_grid
        grid_t, grid_h, grid_w = [int(v) for v in grid_tensor.tolist()]
        patch_unit = model.patch_size * model.merge_size
        qwen_h = grid_h // model.merge_size
        qwen_w = grid_w // model.merge_size
        col_start = max(0, math.floor(x0 / patch_unit))
        col_end = min(qwen_w, math.ceil(x1 / patch_unit))
        row_start = max(0, math.floor(y0 / patch_unit))
        row_end = min(qwen_h, math.ceil(y1 / patch_unit))
        print(f'Patch rows [{row_start}, {row_end}), cols [{col_start}, {col_end})')
        roi_token_cnt = max(row_end - row_start, 0) * max(col_end - col_start, 0)
        print('ROI token count:', roi_token_cnt)
    print('Test finished.')


if __name__ == '__main__':
    main()
