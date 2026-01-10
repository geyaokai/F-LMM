import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from mmengine.config import Config
from mmengine.logging import print_log
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint


def _now():
    return time.perf_counter()


@torch.no_grad()
def try_generate_one_pass_attn(model, inputs, max_new_tokens: int):
    """
    Try to get attentions directly from generate().
    If supported, HF generate may return a GenerateOutput-like object with .attentions.
    """
    qwen = model.qwen_model
    device = model.qwen_device

    gen_kwargs = dict(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device, dtype=qwen.dtype),
        image_grid_thw=inputs["image_grid_thw"].to(device),
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id),
        # try to force generate() to return extra tensors
        return_dict_in_generate=True,
        output_attentions=True,
        output_scores=False,
    )
    if gen_kwargs["attention_mask"] is not None:
        gen_kwargs["attention_mask"] = gen_kwargs["attention_mask"].to(device)

    t0 = _now()
    out = qwen.generate(**gen_kwargs)
    dt = _now() - t0

    # out can be either Tensor (fallback) or an object/dict with sequences/attentions
    sequences = None
    attentions = None
    out_type = type(out).__name__

    if isinstance(out, torch.Tensor):
        sequences = out
    elif isinstance(out, dict):
        sequences = out.get("sequences", None)
        attentions = out.get("attentions", None)
    else:
        # HF GenerateOutput
        sequences = getattr(out, "sequences", None)
        attentions = getattr(out, "attentions", None)

    return {
        "ok": sequences is not None,
        "out_type": out_type,
        "time_s": dt,
        "sequences": sequences,
        "attentions": attentions,
    }


def summarize_generate_attentions(attentions):
    """
    HF generate attentions are often nested like:
      attentions[step][layer] = [batch, heads, 1, past_len]
    or sometimes:
      attentions[layer] = ...
    We only print a robust summary.
    """
    if attentions is None:
        return "attentions=None"

    try:
        # Try common "per step" format: tuple(steps) of tuple(layers) of tensors
        num_steps = len(attentions)
        first_step = attentions[0]
        num_layers = len(first_step) if hasattr(first_step, "__len__") else None
        first = first_step[0] if num_layers else first_step
        shape = tuple(first.shape) if hasattr(first, "shape") else None
        return f"attentions: steps={num_steps}, layers={num_layers}, first_tensor_shape={shape}"
    except Exception as e:
        return f"attentions: <unrecognized structure> ({e})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to mmengine config, e.g. configs/qwen/xxx.py")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint .pth")
    parser.add_argument("--image", required=True, type=str, help="Path to an image file")
    parser.add_argument("--question", default="Describe this image.", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument(
        "--hf_offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to avoid HuggingFace Hub timeouts.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # Build model from cfg.model
    model = BUILDER.build(cfg.model)
    ckpt = guess_load_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print_log(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Prepare generation settings (your FrozenQwenSAM requires this)
    if not hasattr(model, "_prepare_for_generation"):
        raise RuntimeError("Model has no _prepare_for_generation(). Is this FrozenQwenSAM?")
    if getattr(model, "processor", None) is None:
        raise RuntimeError("model.processor is None. Please ensure cfg.model.processor is set.")

    model._prepare_for_generation(
        image_processor=model.processor,
        prompt_template={"SYSTEM": ""},  # keep minimal
        max_new_tokens=args.max_new_tokens,
        additional_prompt="",
        max_history_turns=0,
    )

    image = Image.open(Path(args.image)).convert("RGB")

    # Build the same inputs as answer() does (so we can call generate() directly)
    conversation = model._build_conversation(image, args.question, history=None)
    prompt_text = model.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )

    print_log("=== Test A: one-pass generate() attentions (LVLM-style) ===")
    a = try_generate_one_pass_attn(model, inputs, max_new_tokens=args.max_new_tokens)
    print_log(f"generate() return type: {a['out_type']}")
    print_log(f"generate() time: {a['time_s']:.3f}s")
    print_log(summarize_generate_attentions(a["attentions"]))

    if a["attentions"] is None:
        print_log(
            "Result: generate() did NOT expose attentions. "
            "In this case, your current 'generate then forward()' (two-pass) is needed."
        )
    else:
        print_log(
            "Result: generate() exposed attentions. "
            "You can potentially refactor to avoid the second forward()."
        )

    print_log("\n=== Test B: answer() end-to-end (single-pass generate) ===")
    t0 = _now()
    out = model.answer(image=image, question=args.question, max_new_tokens=args.max_new_tokens)
    dt = _now() - t0

    attn_maps = out.get("attention_maps", None)
    grid = out.get("image_grid_thw", None)
    print_log(f"answer() time: {dt:.3f}s")
    if grid is not None:
        print_log(f"image_grid_thw: {grid.numpy().tolist()}")
    if attn_maps is None:
        print_log("answer() returned no attention_maps")
    else:
        # attention_maps: [num_layers, num_heads, gen_len, H, W]
        print_log(f"attention_maps shape: {tuple(attn_maps.shape)}")
        # quick sanity: finite + non-negative
        finite = torch.isfinite(attn_maps).all().item()
        print_log(f"attention_maps finite: {finite}")

    print_log("\nDone.")


if __name__ == "__main__":
    main()