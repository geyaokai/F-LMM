from __future__ import annotations

import os.path as osp

import torch


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def guess_load_checkpoint(path: str):
    if osp.isfile(path):
        state_dict = _torch_load_cpu(path)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict

    if osp.isdir(path):
        try:
            from xtuner.utils.zero_to_any_dtype import (
                get_state_dict_from_zero_checkpoint,
            )
        except ImportError as exc:
            raise ImportError(
                "Directory checkpoints require xtuner DeepSpeed conversion utils. "
                "Use a merged .pth checkpoint or install the missing dependency stack."
            ) from exc
        return get_state_dict_from_zero_checkpoint(
            osp.dirname(path), osp.basename(path)
        )

    raise FileNotFoundError(f"Cannot find checkpoint: {path}")
