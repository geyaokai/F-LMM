#!/usr/bin/env python3
"""
测试 Qwen 配置文件是否可以正常加载

用法：
    python scripts/test_qwen_config.py
    python scripts/test_qwen_config.py --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py
"""

import sys
import argparse
from pathlib import Path
import torch
from mmengine.config import Config
from xtuner.registry import BUILDER

def test_config_loading(config_path):
    """测试配置文件加载"""
    print("=" * 80)
    print(f"测试配置文件: {config_path}")
    print("=" * 80)
    
    # 1. 加载配置文件
    print("\n[1/5] 加载配置文件...")
    try:
        cfg = Config.fromfile(config_path)
        print(f"✓ 配置文件加载成功")
        print(f"  - 模型类型: {cfg.model.type}")
        print(f"  - Batch size: {cfg.batch_size}")
        print(f"  - 学习率: {cfg.lr}")
        print(f"  - 最大 epochs: {cfg.max_epochs}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False
    
    # 2. 测试 Processor 构建
    print("\n[2/5] 测试 Processor 构建...")
    try:
        processor = BUILDER.build(cfg.processor)
        print(f"✓ Processor 构建成功: {type(processor)}")
        print(f"  - Tokenizer: {type(processor.tokenizer)}")
        if hasattr(processor, 'image_processor'):
            print(f"  - Image Processor: {type(processor.image_processor)}")
        
        # 测试特殊 token
        special_tokens = ['<image>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>']
        print("\n  特殊 Token IDs:")
        for token in special_tokens:
            try:
                token_ids = processor.tokenizer.encode(token, add_special_tokens=False)
                print(f"    {token}: {token_ids}")
            except:
                pass
    except Exception as e:
        print(f"✗ Processor 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试模型配置（不实际加载权重）
    print("\n[3/5] 测试模型配置...")
    try:
        print(f"  模型配置:")
        print(f"    - 类型: {cfg.model.type}")
        print(f"    - SAM 模型: {cfg.model.sam.model_name}")
        print(f"    - Mask Head 输入通道: {cfg.model.mask_head.in_channels}")
        print(f"    - 基础通道数: {cfg.model.mask_head.base_channels}")
        print(f"✓ 模型配置验证通过")
    except Exception as e:
        print(f"✗ 模型配置验证失败: {e}")
        return False
    
    # 4. 测试数据集配置
    print("\n[4/5] 测试数据集配置...")
    try:
        print(f"  数据集配置:")
        print(f"    - 数据集数量: {len(cfg.datasets_list)}")
        for i, ds in enumerate(cfg.datasets_list):
            ds_type = ds.get('type', 'Unknown')
            print(f"    - 数据集 {i+1}: {ds_type}")
        print(f"  DataLoader 配置:")
        print(f"    - Batch size: {cfg.train_dataloader.batch_size}")
        print(f"    - Workers: {cfg.train_dataloader.num_workers}")
        print(f"✓ 数据集配置验证通过")
    except Exception as e:
        print(f"✗ 数据集配置验证失败: {e}")
        return False
    
    # 5. 测试优化器配置
    print("\n[5/5] 测试优化器配置...")
    try:
        print(f"  优化器配置:")
        print(f"    - 类型: {cfg.optim_wrapper.optimizer.type}")
        print(f"    - 学习率: {cfg.optim_wrapper.optimizer.lr}")
        print(f"    - 梯度累积: {cfg.optim_wrapper.accumulative_counts}")
        print(f"    - 混合精度: {cfg.optim_wrapper.dtype}")
        print(f"  学习率调度器:")
        for i, sched in enumerate(cfg.param_scheduler):
            print(f"    - 调度器 {i+1}: {sched.type}")
        print(f"✓ 优化器配置验证通过")
    except Exception as e:
        print(f"✗ 优化器配置验证失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ 所有配置验证通过！")
    print("=" * 80)
    
    return True


def test_model_forward(config_path):
    """测试模型前向传播（可选，需要 GPU 和数据）"""
    print("\n" + "=" * 80)
    print("测试模型前向传播")
    print("=" * 80)
    print("\n注意：这个测试需要：")
    print("  1. 可用的 GPU")
    print("  2. SAM checkpoint: checkpoints/sam_vit_l_0b3195.pth")
    print("  3. 实际的训练数据")
    print("\n跳过此测试。如果需要测试模型前向传播，请参考 test_qwen_interface.py")
    

def main():
    parser = argparse.ArgumentParser(description='测试 Qwen 配置文件')
    parser.add_argument('--config', 
                        default='configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py',
                        help='配置文件路径')
    parser.add_argument('--test-forward', action='store_true',
                        help='测试模型前向传播（需要 GPU 和数据）')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 测试配置加载
    success = test_config_loading(args.config)
    
    if not success:
        print("\n配置验证失败！")
        sys.exit(1)
    
    # 可选：测试模型前向传播
    if args.test_forward:
        test_model_forward(args.config)
    
    print("\n建议下一步：")
    print("  1. 确保已安装 Qwen 相关依赖（见 scripts/README_qwen_test.md）")
    print("  2. 准备训练数据（COCO + RefCOCO 数据集）")
    print("  3. 下载 SAM checkpoint: checkpoints/sam_vit_l_0b3195.pth")
    print("  4. 运行训练:")
    print(f"     ./train.sh --config {args.config} --gpus 2")


if __name__ == '__main__':
    main()

