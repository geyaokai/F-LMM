#!/usr/bin/env python3
"""Build dataloader from config, grab one sample, and run FrozenQwen forward."""

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='F-LMM/configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py',
        help='Path to the training config used during debugging.',
    )
    parser.add_argument('--device', default='cuda', help='Device for model forward pass.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override dataloader batch size.')
    parser.add_argument('--num-workers', type=int, default=None, help='Override dataloader num_workers.')
    parser.add_argument('--skip-batches', type=int, default=0, help='Number of batches to skip before debugging sample.')
    parser.add_argument('--sample-offset', type=int, default=0, help='Index inside the chosen batch.')
    parser.add_argument('--no-forward', action='store_true', help='Only dump sample info, skip model forward.')
    parser.add_argument('--pdb', action='store_true', help='Enter pdb right before forward call.')
    parser.add_argument('--dump-summary', action='store_true', help='Print tensor shapes for the picked sample.')
    return parser.parse_args()


def _clone_cfg(cfg_item: Any) -> Any:
    return deepcopy(cfg_item)


def build_dataset(dataset_cfg) -> torch.utils.data.Dataset:
    from xtuner.registry import BUILDER

    dataset_cfg = _clone_cfg(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    if callable(dataset_type):
        return dataset_type(**dataset_cfg)
    dataset_cfg['type'] = dataset_type
    return BUILDER.build(dataset_cfg)


def build_collate_fn(collate_cfg):
    if collate_cfg is None:
        return None
    collate_cfg = _clone_cfg(collate_cfg)
    fn = collate_cfg.pop('type')
    if callable(fn) and not collate_cfg:
        return fn
    if callable(fn):
        def wrapper(instances: Sequence[Dict]):
            return fn(instances, **collate_cfg)

        return wrapper
    raise TypeError('collate_fn type must be callable in debug script.')


def build_dataloader(cfg, dataset) -> DataLoader:
    batch_size = cfg.get('batch_size', 1)
    num_workers = cfg.get('num_workers', 0)
    collate_fn = build_collate_fn(cfg.get('collate_fn', None))
    pin_memory = cfg.get('pin_memory', False)
    prefetch_factor = cfg.get('prefetch_factor', None)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=cfg.get('persistent_workers', False),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_model(cfg, device: str):
    from xtuner.registry import BUILDER

    model = BUILDER.build(cfg.model)
    model = model.to(device).eval()
    return model


def select_sample(batch: Any, offset: int):
    if isinstance(batch, dict) and 'data' in batch:
        instances = batch['data']
    else:
        instances = batch
    if offset >= len(instances):
        raise IndexError(f'sample_offset {offset} out of range for batch length {len(instances)}')
    return instances[offset]


def summarize_data_sample(sample: Dict[str, Any]):
    print('Prepared sample keys:', list(sample.keys()))
    for key in ['input_ids', 'attention_mask', 'mask_ids', 'pixel_values', 'masks', 'image_grid_thw']:
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, torch.Tensor):
            print(f'  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}')
        else:
            print(f'  - {key}: type={type(value)}')
    meta = sample.get('meta_data')
    if meta is not None:
        print(f"  - meta_data.image_shape: {meta.get('image_shape')}")
        print(f"  - meta_data.padded_shape: {meta.get('padded_shape')}")


def main():
    args = parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)

    from mmengine.config import Config

    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.train_dataloader.dataset)

    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train_dataloader.num_workers = args.num_workers

    dataloader = build_dataloader(cfg.train_dataloader, dataset)
    iterator = iter(dataloader)
    try:
        for _ in range(args.skip_batches):
            next(iterator)
    except StopIteration as exc:
        raise RuntimeError('skip-batches exceeds dataloader length') from exc
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError('Dataloader exhausted before retrieving sample') from exc
    sample = select_sample(batch, args.sample_offset)

    if args.dump_summary:
        summarize_data_sample(sample)

    if args.no_forward:
        print('Sample prepared, skipping forward pass (--no-forward set).')
        return

    if args.pdb:
        import pdb

        pdb.set_trace()

    model = build_model(cfg, args.device)
    with torch.no_grad():
        outputs = model._forward(sample)
    print('Forward finished. Output keys:', list(outputs.keys()))


if __name__ == '__main__':
    main()
