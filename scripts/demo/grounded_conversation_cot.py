import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from mmengine.config import Config
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from scripts.demo.utils import colors

random.shuffle(colors)
RESULT_DIR = 'scripts/demo/results'


def ensure_results_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)
    return RESULT_DIR


def run_visual_cot(model, image_path, question, color_idx=0):
    """Run a single Visual CoT pass and save overlay result."""
    image = Image.open(image_path).convert('RGB')
    thought, bbox, answer, pred_mask = model.visual_cot_v1(image, question)

    image_np = np.array(image).astype(np.float32)
    mask = pred_mask.detach().cpu().numpy() > 0
    overlay_color = np.array(colors[color_idx % len(colors)]).reshape((1, 1, 3))
    image_np[mask] = image_np[mask] * 0.2 + overlay_color * 0.8

    result_image = Image.fromarray(image_np.astype(np.uint8))
    save_dir = ensure_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    save_path = os.path.join(save_dir, f'{Path(image_path).stem}_{timestamp}.jpg')
    result_image.save(save_path)

    return {
        'question': question,
        'thought': thought,
        'answer': answer,
        'bbox': bbox,
        'image_path': image_path,
        'save_path': save_path,
    }


def print_run_summary(result):
    print(f"\nQuestion: {result['question']}")
    print(f"Thought: {result['thought']}")
    print(f"Answer: {result['answer']}")
    print(f"Grounding BBox: {result['bbox']}")
    print(f"Result saved to {result['save_path']}")


def interactive_session(model, default_image):
    """Provide a lightweight CLI loop for Visual CoT demo."""
    commands = (
        "Commands:\n"
        "  /image <path>  Load or switch the current image\n"
        "  /show          Display the currently selected image\n"
        "  /help          Print this help message\n"
        "  /exit          Quit interactive mode\n"
        "Enter a natural question to run Visual CoT on the current image."
    )
    current_image = None
    if default_image:
        expanded = Path(default_image).expanduser()
        if expanded.is_file():
            current_image = str(expanded)
        else:
            print(f"[Warning] Default image {default_image} not found. "
                  "Use /image <path> to set one.")
    print("Entering interactive Visual CoT mode.")
    print(commands)
    color_idx = 0
    while True:
        try:
            user_input = input("\nQuestion (/help for options): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in {'/exit', 'exit', 'quit'}:
            print("Bye!")
            break
        if lower_input in {'/help', 'help'}:
            print(commands)
            continue
        if lower_input in {'/show', 'show'}:
            if current_image:
                print(f"Current image: {current_image}")
            else:
                print("No image selected. Use /image <path> first.")
            continue
        if lower_input.startswith('/image'):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                if current_image:
                    print(f"Current image: {current_image}")
                else:
                    print("No image selected. Use /image <path>.")
                continue
            candidate = Path(parts[1]).expanduser()
            if candidate.is_file():
                current_image = str(candidate)
                print(f"Using image: {current_image}")
            else:
                print(f"Image not found: {candidate}")
            continue

        if current_image is None:
            print("Please load an image first with /image <path>.")
            continue

        try:
            result = run_visual_cot(model, current_image, user_input, color_idx=color_idx)
            print_run_summary(result)
            color_idx += 1
        except Exception as exc:  # pragma: no cover - debugging convenience
            print(f"[Error] Failed to run Visual CoT: {exc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--image', default='data/coco/val2017/000000000632.jpg', type=str)
    parser.add_argument('--text', default='Where is the shampoo?', type=str)
    parser.add_argument('--checkpoint', default='checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth', type=str)
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable an interactive loop for multiple questions.')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    image_processor = cfg.image_processor

    model = BUILDER.build(cfg.model)
    state_dict = guess_load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # 初始化生成配置（需包含Visual CoT所需参数）
    model._prepare_for_generation(
        image_processor=image_processor,
        prompt_template=prompt_template,
        max_thought_tokens=16,
        max_new_tokens=512,
        lmm_name=cfg.lmm_name,
        use_sam=args.use_sam
    )
    model = model.cuda().eval()

    if args.interactive:
        interactive_session(model, args.image)
    else:
        try:
            result = run_visual_cot(model, args.image, args.text)
            print_run_summary(result)
        except FileNotFoundError as exc:
            print(f"[Error] Failed to open image: {exc}")
