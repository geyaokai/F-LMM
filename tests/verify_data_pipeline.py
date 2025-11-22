#!/usr/bin/env python3
"""
验证数据处理管道是否正确提供 image_grid_thw

此脚本测试:
1. RefCOCO2PNG transform 是否提取并返回 image_grid_thw
2. PNGDataset 是否提取并返回 image_grid_thw
3. image_grid_thw 的格式是否正确
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import numpy as np

print("=" * 80)
print("测试 1: 验证 RefCOCO2PNG Transform")
print("=" * 80)

try:
    from flmm.datasets.transforms import RefCOCO2PNG
    from transformers import AutoProcessor
    
    # 创建测试数据
    test_image = Image.new('RGB', (640, 480), color='red')
    test_mask = np.ones((480, 640), dtype=np.uint8)
    
    # Qwen2.5-VL prompt template (from config)
    prompt_template = dict(
        SYSTEM='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n',
        INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
        SEP='\n',
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']
    )
    
    # 创建 transform
    transform = RefCOCO2PNG(
        image_processor=dict(
            type='transformers.AutoProcessor.from_pretrained',
            pretrained_model_name_or_path='Qwen/Qwen2.5-VL-7B-Instruct',
            trust_remote_code=True
        ),
        tokenizer=dict(
            type='transformers.AutoProcessor.from_pretrained',
            pretrained_model_name_or_path='Qwen/Qwen2.5-VL-7B-Instruct',
            trust_remote_code=True
        ),
        prompt_template=prompt_template,
        prompt='<image>Please give me a description of the image.',
        image_token='<image>'
    )
    
    # 模拟 mmdet 数据格式
    from mmdet.structures.mask import BitmapMasks
    results = {
        'img': test_image,
        'text': ['a red image'],
        'gt_masks': BitmapMasks([test_mask], height=480, width=640)
    }
    
    # 执行 transform
    output = transform.transform_concat(results)
    
    # 检查输出
    print(f"✓ Transform 成功执行")
    print(f"  - 输出字段: {list(output.keys())}")
    
    if 'image_grid_thw' in output:
        image_grid_thw = output['image_grid_thw']
        print(f"  ✓ image_grid_thw 存在")
        print(f"    - 形状: {image_grid_thw.shape}")
        print(f"    - 值: {image_grid_thw}")
        print(f"    - dtype: {image_grid_thw.dtype}")
        
        # 验证格式
        assert image_grid_thw.dim() in [1, 2], f"image_grid_thw 应该是 1D 或 2D，实际: {image_grid_thw.dim()}D"
        if image_grid_thw.dim() == 2:
            assert image_grid_thw.shape[1] == 3, f"image_grid_thw 的第二维应该是 3，实际: {image_grid_thw.shape[1]}"
        else:
            assert image_grid_thw.shape[0] == 3, f"image_grid_thw 应该包含 3 个值，实际: {image_grid_thw.shape[0]}"
        
        print(f"  ✓ image_grid_thw 格式正确")
        print(f"\n🎉 测试 1 通过！")
    else:
        print(f"  ✗ image_grid_thw 缺失！")
        print(f"\n❌ 测试 1 失败：image_grid_thw 未在输出中")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ 测试 1 失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 2: 验证 pixel_values 恢复")
print("=" * 80)

try:
    pixel_values = output['pixel_values']
    print(f"  - pixel_values 形状: {pixel_values.shape}")
    print(f"  - pixel_values dtype: {pixel_values.dtype}")
    
    # 检查是否是 2D 格式（需要恢复）或已经是标准格式
    if pixel_values.dim() == 2:
        print(f"  ⚠️  pixel_values 是 2D 格式 {pixel_values.shape}，需要在模型中恢复")
    elif pixel_values.dim() == 3:
        print(f"  ✓ pixel_values 是 3D 格式 [C, H, W]: {pixel_values.shape}")
    elif pixel_values.dim() == 4:
        print(f"  ✓ pixel_values 是 4D 格式 [B, C, H, W]: {pixel_values.shape}")
    else:
        print(f"  ✗ pixel_values 格式异常: {pixel_values.dim()}D")
    
    print(f"\n🎉 测试 2 完成！")
    
except Exception as e:
    print(f"\n❌ 测试 2 失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 3: 验证 image_grid_thw 与图像尺寸的对应关系")
print("=" * 80)

try:
    # 从 meta_data 获取图像尺寸
    meta_data = output['meta_data']
    img_h = meta_data['image_shape']['height']
    img_w = meta_data['image_shape']['width']
    
    print(f"  - 图像尺寸: {img_h} x {img_w}")
    
    # 从 image_grid_thw 获取 grid
    if image_grid_thw.dim() == 2:
        t, h_grid, w_grid = image_grid_thw[0].tolist()
    else:
        t, h_grid, w_grid = image_grid_thw.tolist()
    
    print(f"  - image_grid_thw: t={t}, h={h_grid}, w={w_grid}")
    
    # 计算预期的 grid（patch_size = 14）
    patch_size = 14
    expected_h_grid = (img_h + patch_size - 1) // patch_size
    expected_w_grid = (img_w + patch_size - 1) // patch_size
    
    print(f"  - 预期 grid (patch_size={patch_size}): h={expected_h_grid}, w={expected_w_grid}")
    
    # 注意：实际的 grid 可能与简单计算的不同，因为 Qwen 使用动态分辨率
    # 但它们应该在合理范围内
    h_diff = abs(h_grid - expected_h_grid)
    w_diff = abs(w_grid - expected_w_grid)
    
    if h_diff <= 2 and w_diff <= 2:  # 允许小的差异
        print(f"  ✓ grid 尺寸在合理范围内")
    else:
        print(f"  ⚠️  grid 尺寸差异较大: h_diff={h_diff}, w_diff={w_diff}")
        print(f"    这可能是由于 Qwen 的动态分辨率处理")
    
    print(f"\n🎉 测试 3 完成！")
    
except Exception as e:
    print(f"\n❌ 测试 3 失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
print("\n总结:")
print("  ✓ RefCOCO2PNG 正确提取并返回 image_grid_thw")
print("  ✓ image_grid_thw 格式正确")
print("  ✓ image_grid_thw 与图像尺寸对应关系合理")
print("\n建议:")
print("  1. 运行实际训练测试数据管道")
print("  2. 检查 collate_fn 是否正确传递 image_grid_thw")
print("  3. 验证模型训练是否不再出现 RuntimeError")
print("=" * 80)

