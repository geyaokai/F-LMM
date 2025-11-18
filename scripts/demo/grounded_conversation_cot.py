import argparse
import torch
import numpy as np
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from scripts.demo.utils import colors
import os
import random
random.shuffle(colors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--image', default='data/coco/val2017/000000000632.jpg', type=str)
    parser.add_argument('--text', default='Where is the shampoo?', type=str)
    parser.add_argument('--checkpoint', default='checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth', type=str)
    parser.add_argument('--use_sam', action='store_true')
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

    image = Image.open(args.image)
    
    # 使用Visual CoT v1自动定位
    thought, bbox, answer, pred_mask = model.visual_cot_v1(image, args.text)
    
    # 输出结果
    print(f"Question: {args.text}")
    print(f"Thought: {thought}")
    print(f"Answer: {answer}")
    print(f"Grounding BBox: {bbox}")

    # 可视化结果
    image_np = np.array(image).astype(np.float32)
    mask = pred_mask.cpu().numpy() > 0
    image_np[mask] = image_np[mask] * 0.2 + np.array(colors[0]).reshape((1, 1, 3)) * 0.8

    result_image = Image.fromarray(image_np.astype(np.uint8))
    os.makedirs('scripts/demo/results', exist_ok=True)
    result_image.save('scripts/demo/results/example.jpg')
    print("Result saved to scripts/demo/results/example.jpg")
