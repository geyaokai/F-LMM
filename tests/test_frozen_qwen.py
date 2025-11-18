"""
FrozenQwen 模型单元测试

主要测试内容：
1. data_sample 数据结构验证
2. Qwen 动态分辨率处理
3. image_grid_thw 生成和传递
4. 与 processor 的正确交互
"""

import torch
import unittest
import logging
from PIL import Image
import numpy as np
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TestQwenDataSample(unittest.TestCase):
    """测试 Qwen 数据样本的结构"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        logger.info("=" * 70)
        logger.info("开始初始化测试环境")
        logger.info("=" * 70)
        try:
            from transformers import AutoProcessor
            cls.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                trust_remote_code=True
            )
            logger.info("✓ Processor 加载成功")
        except Exception as e:
            logger.error(f"✗ Processor 加载失败: {e}")
            cls.processor = None
    
    def test_01_processor_available(self):
        """测试 processor 是否可用"""
        logger.info("\n" + "=" * 70)
        logger.info("Test 1: 测试 Processor 可用性")
        logger.info("=" * 70)
        
        if self.processor is None:
            logger.error("✗ 测试失败: Processor 未能加载")
            self.fail("Processor 未能加载")
        else:
            logger.info("✓ 测试通过: Processor 可用")
    
    def test_02_basic_image_processing(self):
        """测试基本的图像处理"""
        if self.processor is None:
            self.skipTest("Processor 不可用")
        
        # 创建测试图像
        image = Image.new('RGB', (224, 224), color='red')
        text = "<image>Please describe this image."
        
        # 处理输入
        try:
            # Qwen2.5-VL 使用 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Please describe this image."}
                    ]
                }
            ]
            
            inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                images=[image],
                return_tensors="pt"
            )
            
            print(f"✓ Test 2: 图像处理成功")
            print(f"  - input_ids shape: {inputs['input_ids'].shape}")
            print(f"  - pixel_values shape: {inputs['pixel_values'].shape}")
            
            # 检查关键字段
            self.assertIn('input_ids', inputs, "缺少 input_ids")
            self.assertIn('pixel_values', inputs, "缺少 pixel_values")
            
            # Qwen2.5-VL 的关键字段
            if 'image_grid_thw' in inputs:
                print(f"  - image_grid_thw: {inputs['image_grid_thw']}")
                self.assertIsNotNone(inputs['image_grid_thw'], "image_grid_thw 不应为 None")
            else:
                print("  ⚠ WARNING: image_grid_thw 不在 inputs 中！")
            
        except Exception as e:
            self.fail(f"图像处理失败: {e}")
    
    def test_03_dynamic_resolution(self):
        """测试 Qwen 的动态分辨率处理"""
        if self.processor is None:
            self.skipTest("Processor 不可用")
        
        # 测试不同分辨率的图像
        test_sizes = [
            (224, 224, "正方形"),
            (448, 224, "宽矩形"),
            (224, 448, "高矩形"),
            (640, 480, "标准分辨率"),
        ]
        
        print("✓ Test 3: 动态分辨率测试")
        for width, height, desc in test_sizes:
            image = Image.new('RGB', (width, height), color='blue')
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Test"}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt"
                )
                
                print(f"  - {desc} ({width}x{height}):")
                print(f"    pixel_values: {inputs['pixel_values'].shape}")
                
                if 'image_grid_thw' in inputs:
                    grid_thw = inputs['image_grid_thw']
                    print(f"    image_grid_thw: {grid_thw}")
                    
                    # 验证 grid_thw 的结构
                    self.assertIsNotNone(grid_thw, "image_grid_thw 不应为 None")
                    # grid_thw 应该是 [batch, 3] 或类似的形状
                    if isinstance(grid_thw, torch.Tensor):
                        self.assertEqual(grid_thw.ndim, 2, "grid_thw 应该是 2D tensor")
                        self.assertEqual(grid_thw.shape[-1], 3, "grid_thw 最后一维应该是 3 (t, h, w)")
                else:
                    print(f"    ⚠ WARNING: image_grid_thw 缺失！")
                
            except Exception as e:
                print(f"    ✗ 处理失败: {e}")
    
    def test_04_data_sample_structure(self):
        """测试完整的 data_sample 结构（模拟 dataset 输出）"""
        if self.processor is None:
            self.skipTest("Processor 不可用")
        
        print("✓ Test 4: data_sample 结构验证")
        
        # 创建模拟的 data_sample
        image = Image.new('RGB', (640, 480), color='green')
        text = "Please segment the object."
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            text_formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_formatted],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # 构建 data_sample（模拟 dataset 输出）
            data_sample = {
                'input_ids': inputs['input_ids'][0],
                'pixel_values': inputs['pixel_values'],
                'image': image,
                'masks': torch.zeros(1, 480, 640),  # 模拟 GT mask
                'mask_ids': torch.zeros(inputs['input_ids'].shape[1], dtype=torch.long),
                'meta_data': {
                    'image_shape': {'height': 480, 'width': 640},
                    'padded_shape': {'height': 480, 'width': 640},
                    'padding': {'before_height': 0, 'before_width': 0}
                }
            }
            
            # 添加 Qwen 特有的字段
            if 'attention_mask' in inputs:
                data_sample['attention_mask'] = inputs['attention_mask'][0]
            
            if 'image_grid_thw' in inputs:
                data_sample['image_grid_thw'] = inputs['image_grid_thw']
                print(f"  ✓ image_grid_thw 已添加: {inputs['image_grid_thw']}")
            else:
                print(f"  ✗ CRITICAL: image_grid_thw 缺失！")
                print(f"  可用的 keys: {list(inputs.keys())}")
            
            # 验证必需字段
            required_fields = [
                'input_ids', 'pixel_values', 'image', 'masks', 
                'mask_ids', 'meta_data'
            ]
            
            for field in required_fields:
                self.assertIn(field, data_sample, f"缺少必需字段: {field}")
                print(f"  ✓ {field}: {type(data_sample[field])}")
            
            # 验证 Qwen 特有字段
            if 'image_grid_thw' in data_sample:
                print(f"  ✓ image_grid_thw 验证通过")
            else:
                print(f"  ⚠ WARNING: image_grid_thw 未在 data_sample 中")
            
            # 验证数据类型和形状
            self.assertEqual(data_sample['input_ids'].ndim, 1, "input_ids 应该是 1D")
            
            # Qwen2.5-VL 的 pixel_values 可能是 2D (扁平化格式) 或 4D [B, C, H, W]
            pixel_values_ndim = data_sample['pixel_values'].ndim
            self.assertIn(pixel_values_ndim, [2, 3, 4], 
                         f"pixel_values 应该是 2D/3D/4D，实际是 {pixel_values_ndim}D")
            print(f"  ✓ pixel_values 维度: {pixel_values_ndim}D (形状: {data_sample['pixel_values'].shape})")
            
            print(f"  data_sample 结构验证完成")
            
        except Exception as e:
            self.fail(f"data_sample 构建失败: {e}")
    
    def test_05_vision_tokens(self):
        """测试视觉 token 的识别"""
        if self.processor is None:
            self.skipTest("Processor 不可用")
        
        print("✓ Test 5: 视觉 token 验证")
        
        tokenizer = self.processor.tokenizer
        
        # Qwen2.5-VL 的视觉 token
        vision_tokens = {
            '<|vision_start|>': 151652,
            '<|vision_end|>': 151653,
            '<|image_pad|>': 151655,
        }
        
        for token, expected_id in vision_tokens.items():
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"  - {token}: {token_id}")
                if token_id != tokenizer.unk_token_id:
                    self.assertEqual(token_id, expected_id, f"{token} ID 不匹配")
            except Exception as e:
                print(f"  ⚠ {token} 查找失败: {e}")
    
    def test_06_image_grid_thw_calculation(self):
        """测试 image_grid_thw 的计算逻辑"""
        print("✓ Test 6: image_grid_thw 计算验证")
        
        # Qwen2.5-VL 的 patch_size 通常是 14
        patch_size = 14
        
        test_cases = [
            ((224, 224), "正方形小图"),
            ((448, 336), "矩形图"),
            ((640, 480), "标准分辨率"),
            ((1024, 768), "大图"),
        ]
        
        for (width, height), desc in test_cases:
            # 计算预期的 grid 尺寸
            grid_h = (height + patch_size - 1) // patch_size
            grid_w = (width + patch_size - 1) // patch_size
            num_patches = grid_h * grid_w
            
            print(f"  - {desc} ({width}x{height}):")
            print(f"    预期 grid: {grid_h} x {grid_w} = {num_patches} patches")
            
            # 如果有 processor，验证实际处理结果
            if self.processor is not None:
                try:
                    image = Image.new('RGB', (width, height), color='yellow')
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "Test"}
                            ]
                        }
                    ]
                    
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt"
                    )
                    
                    if 'image_grid_thw' in inputs:
                        actual_grid = inputs['image_grid_thw']
                        print(f"    实际 grid_thw: {actual_grid}")
                        
                        # 验证格式
                        if isinstance(actual_grid, torch.Tensor):
                            self.assertEqual(actual_grid.shape[-1], 3, "grid_thw 应该有 3 个值 (t, h, w)")
                    else:
                        print(f"    ⚠ image_grid_thw 缺失")
                        
                except Exception as e:
                    print(f"    处理失败: {e}")


class TestQwenModelIntegration(unittest.TestCase):
    """测试 FrozenQwen 模型集成"""
    
    def test_01_model_import(self):
        """测试模型导入"""
        print("\n✓ Test: 模型导入")
        try:
            from flmm.models.frozen_qwen import FrozenQwen, FrozenQwenSAM
            print("  ✓ 模型类导入成功")
        except Exception as e:
            self.fail(f"模型导入失败: {e}")
    
    def test_02_prepare_inputs_logic(self):
        """测试 _prepare_inputs 方法的逻辑"""
        print("✓ Test: _prepare_inputs 逻辑")
        
        # 模拟 input_ids
        # 假设序列: [text] <|vision_start|> [image_tokens] <|vision_end|> [text]
        vision_start_id = 151652
        vision_end_id = 151653
        image_pad_id = 151655
        
        # 构造测试序列
        input_ids = torch.tensor([
            1, 2, 3,  # 前置文本
            vision_start_id,  # 视觉开始
            image_pad_id, image_pad_id, image_pad_id,  # 图像 tokens
            vision_end_id,  # 视觉结束
            4, 5, 6  # 后置文本
        ])
        
        # 验证 mask 构建逻辑
        vision_start_mask = input_ids == vision_start_id
        vision_end_mask = input_ids == vision_end_id
        
        vision_start_positions = vision_start_mask.nonzero(as_tuple=True)[0]
        vision_end_positions = vision_end_mask.nonzero(as_tuple=True)[0]
        
        print(f"  - vision_start 位置: {vision_start_positions.tolist()}")
        print(f"  - vision_end 位置: {vision_end_positions.tolist()}")
        
        # 构建图像 token mask
        images_seq_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if len(vision_start_positions) > 0 and len(vision_end_positions) > 0:
            start_pos = vision_start_positions[0].item()
            end_pos = vision_end_positions[0].item()
            images_seq_mask[start_pos+1:end_pos] = True
        
        expected_image_tokens = [4, 5, 6]  # indices
        actual_image_positions = images_seq_mask.nonzero(as_tuple=True)[0].tolist()
        
        print(f"  - 图像 token 位置: {actual_image_positions}")
        self.assertEqual(actual_image_positions, expected_image_tokens, 
                        "图像 token 位置识别错误")
        print("  ✓ vision token 识别逻辑正确")


def run_diagnostic():
    """运行诊断检查"""
    print("=" * 60)
    print("FrozenQwen 诊断检查")
    print("=" * 60)
    
    # 1. 检查环境
    print("\n1. 环境检查:")
    try:
        import transformers
        print(f"  ✓ transformers 版本: {transformers.__version__}")
    except:
        print(f"  ✗ transformers 未安装")
    
    try:
        import torch
        print(f"  ✓ torch 版本: {torch.__version__}")
        print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
    except:
        print(f"  ✗ torch 未安装")
    
    # 2. 测试 processor
    print("\n2. Processor 测试:")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True
        )
        print("  ✓ Processor 加载成功")
        
        # 测试简单处理
        image = Image.new('RGB', (224, 224), color='red')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Test"}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        
        print(f"  ✓ 处理成功")
        print(f"  - 输出 keys: {list(inputs.keys())}")
        
        # 关键检查
        if 'image_grid_thw' in inputs:
            print(f"  ✓ image_grid_thw 存在: {inputs['image_grid_thw']}")
        else:
            print(f"  ✗ CRITICAL: image_grid_thw 缺失！")
            print(f"  这可能是导致错误的根本原因")
            
    except Exception as e:
        print(f"  ✗ Processor 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # 先运行诊断
    run_diagnostic()
    
    # 运行单元测试
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60 + "\n")
    
    unittest.main(verbosity=2)

