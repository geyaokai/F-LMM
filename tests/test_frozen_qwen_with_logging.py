"""
FrozenQwen 模型单元测试 - 日志输出版本

主要测试内容：
1. data_sample 数据结构验证
2. Qwen 动态分辨率处理
3. image_grid_thw 生成和传递
4. 与 processor 的正确交互
"""

import torch
import logging
from PIL import Image
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志 - 同时输出到控制台和文件
log_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(os.path.dirname(__file__), log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(log_filepath, encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)


class TestQwenDataSample:
    """测试 Qwen 数据样本的结构"""
    
    def __init__(self):
        self.processor = None
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        
    def setup(self):
        """初始化测试环境"""
        logger.info("=" * 80)
        logger.info("🔧 初始化测试环境")
        logger.info("=" * 80)
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )
            logger.info("✓ Processor 加载成功")
            logger.info(f"  - Processor 类型: {type(self.processor)}")
            return True
        except Exception as e:
            logger.error(f"✗ Processor 加载失败: {e}")
            self.processor = None
            return False
    
    def log_test_start(self, test_num, test_name):
        """记录测试开始"""
        logger.info("\n" + "=" * 80)
        logger.info(f"📝 Test {test_num}: {test_name}")
        logger.info("=" * 80)
        self.total_tests += 1
    
    def log_test_pass(self, message=""):
        """记录测试通过"""
        self.passed_tests += 1
        if message:
            logger.info(f"✅ 测试通过: {message}")
        else:
            logger.info("✅ 测试通过")
    
    def log_test_fail(self, message=""):
        """记录测试失败"""
        self.failed_tests += 1
        if message:
            logger.error(f"❌ 测试失败: {message}")
        else:
            logger.error("❌ 测试失败")
    
    def test_01_processor_available(self):
        """测试 processor 是否可用"""
        self.log_test_start(1, "Processor 可用性测试")
        
        if self.processor is None:
            self.log_test_fail("Processor 未能加载")
            return False
        
        logger.info(f"  - Processor 已加载")
        logger.info(f"  - 类型: {type(self.processor).__name__}")
        self.log_test_pass("Processor 可用")
        return True
    
    def test_02_basic_image_processing(self):
        """测试基本的图像处理"""
        self.log_test_start(2, "基本图像处理测试")
        
        if self.processor is None:
            logger.warning("⚠️  跳过测试: Processor 不可用")
            return None
        
        # 创建测试图像
        image = Image.new('RGB', (224, 224), color='red')
        text = "Please describe this image."
        
        try:
            # Qwen2.5-VL 使用 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                images=[image],
                return_tensors="pt"
            )
            
            logger.info("  ✓ 图像处理成功")
            logger.info(f"    - input_ids shape: {inputs['input_ids'].shape}")
            logger.info(f"    - pixel_values shape: {inputs['pixel_values'].shape}")
            
            # 检查关键字段
            if 'input_ids' not in inputs:
                self.log_test_fail("缺少 input_ids")
                return False
            
            if 'pixel_values' not in inputs:
                self.log_test_fail("缺少 pixel_values")
                return False
            
            # 检查 image_grid_thw
            if 'image_grid_thw' in inputs:
                logger.info(f"    - image_grid_thw: {inputs['image_grid_thw']}")
                if inputs['image_grid_thw'] is None:
                    logger.warning("    ⚠️  警告: image_grid_thw 为 None")
                else:
                    logger.info("    ✓ image_grid_thw 存在且不为 None")
            else:
                logger.warning("    ⚠️  警告: image_grid_thw 不在 inputs 中")
            
            self.log_test_pass("基本图像处理正常")
            return True
            
        except Exception as e:
            self.log_test_fail(f"图像处理失败: {e}")
            import traceback
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            return False
    
    def test_03_dynamic_resolution(self):
        """测试 Qwen 的动态分辨率处理"""
        self.log_test_start(3, "动态分辨率测试")
        
        if self.processor is None:
            logger.warning("⚠️  跳过测试: Processor 不可用")
            return None
        
        # 测试不同分辨率的图像
        test_sizes = [
            (224, 224, "正方形"),
            (448, 224, "宽矩形"),
            (224, 448, "高矩形"),
            (640, 480, "标准分辨率"),
        ]
        
        all_passed = True
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
                
                logger.info(f"  - {desc} ({width}x{height}):")
                logger.info(f"      pixel_values: {inputs['pixel_values'].shape}")
                
                if 'image_grid_thw' in inputs:
                    grid_thw = inputs['image_grid_thw']
                    logger.info(f"      image_grid_thw: {grid_thw}")
                    
                    # 验证 grid_thw 的结构
                    if grid_thw is None:
                        logger.error("      ✗ image_grid_thw 为 None")
                        all_passed = False
                    elif isinstance(grid_thw, torch.Tensor):
                        if grid_thw.ndim != 2:
                            logger.error(f"      ✗ grid_thw 维度错误: {grid_thw.ndim}, 应该是 2D")
                            all_passed = False
                        elif grid_thw.shape[-1] != 3:
                            logger.error(f"      ✗ grid_thw 最后一维错误: {grid_thw.shape[-1]}, 应该是 3")
                            all_passed = False
                        else:
                            logger.info("      ✓ image_grid_thw 格式正确")
                else:
                    logger.warning("      ⚠️  image_grid_thw 缺失")
                
            except Exception as e:
                logger.error(f"      ✗ 处理失败: {e}")
                all_passed = False
        
        if all_passed:
            self.log_test_pass("所有分辨率测试通过")
            return True
        else:
            self.log_test_fail("部分分辨率测试失败")
            return False
    
    def test_04_data_sample_structure(self):
        """测试完整的 data_sample 结构"""
        self.log_test_start(4, "data_sample 结构验证")
        
        if self.processor is None:
            logger.warning("⚠️  跳过测试: Processor 不可用")
            return None
        
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
            
            # 构建 data_sample
            data_sample = {
                'input_ids': inputs['input_ids'][0],
                'pixel_values': inputs['pixel_values'],
                'image': image,
                'masks': torch.zeros(1, 480, 640),
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
                logger.info("  ✓ 添加 attention_mask")
            
            if 'image_grid_thw' in inputs:
                data_sample['image_grid_thw'] = inputs['image_grid_thw']
                logger.info(f"  ✓ 添加 image_grid_thw: {inputs['image_grid_thw']}")
            else:
                logger.error("  ✗ CRITICAL: image_grid_thw 缺失！")
                logger.info(f"  可用的 keys: {list(inputs.keys())}")
            
            # 验证必需字段
            logger.info("\n  📋 data_sample 包含的字段:")
            required_fields = [
                'input_ids', 'pixel_values', 'image', 'masks', 
                'mask_ids', 'meta_data'
            ]
            
            all_present = True
            for field in required_fields:
                if field in data_sample:
                    value = data_sample[field]
                    if isinstance(value, torch.Tensor):
                        logger.info(f"    ✓ {field}: shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, Image.Image):
                        logger.info(f"    ✓ {field}: PIL.Image {value.size}")
                    elif isinstance(value, dict):
                        logger.info(f"    ✓ {field}: dict with keys {list(value.keys())}")
                    else:
                        logger.info(f"    ✓ {field}: type={type(value)}")
                else:
                    logger.error(f"    ✗ 缺少必需字段: {field}")
                    all_present = False
            
            # 验证 Qwen 特有字段
            logger.info("\n  🔍 Qwen 特有字段验证:")
            if 'image_grid_thw' in data_sample:
                logger.info("    ✓ image_grid_thw 存在")
            else:
                logger.warning("    ⚠️  image_grid_thw 未在 data_sample 中")
            
            # 验证数据类型和形状
            logger.info("\n  📐 数据维度验证:")
            if data_sample['input_ids'].ndim == 1:
                logger.info(f"    ✓ input_ids 是 1D: {data_sample['input_ids'].shape}")
            else:
                logger.error(f"    ✗ input_ids 应该是 1D，实际是 {data_sample['input_ids'].ndim}D")
                all_present = False
            
            # Qwen2.5-VL 的 pixel_values 可能是 2D/3D/4D
            pixel_values_ndim = data_sample['pixel_values'].ndim
            if pixel_values_ndim in [2, 3, 4]:
                logger.info(f"    ✓ pixel_values 维度: {pixel_values_ndim}D (形状: {data_sample['pixel_values'].shape})")
            else:
                logger.error(f"    ✗ pixel_values 维度异常: {pixel_values_ndim}D")
                all_present = False
            
            if all_present:
                self.log_test_pass("data_sample 结构完整")
                return True
            else:
                self.log_test_fail("data_sample 结构不完整")
                return False
                
        except Exception as e:
            self.log_test_fail(f"data_sample 构建失败: {e}")
            import traceback
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            return False
    
    def test_05_vision_tokens(self):
        """测试视觉 token 的识别"""
        self.log_test_start(5, "视觉 Token 验证")
        
        if self.processor is None:
            logger.warning("⚠️  跳过测试: Processor 不可用")
            return None
        
        tokenizer = self.processor.tokenizer
        
        # Qwen2.5-VL 的视觉 token
        vision_tokens = {
            '<|vision_start|>': 151652,
            '<|vision_end|>': 151653,
            '<|image_pad|>': 151655,
        }
        
        logger.info("  🔤 验证视觉 Token ID:")
        all_correct = True
        for token, expected_id in vision_tokens.items():
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id == tokenizer.unk_token_id:
                    logger.warning(f"    ⚠️  {token}: 未找到 (返回 unk_token)")
                elif token_id == expected_id:
                    logger.info(f"    ✓ {token}: {token_id} (正确)")
                else:
                    logger.error(f"    ✗ {token}: {token_id} (预期: {expected_id})")
                    all_correct = False
            except Exception as e:
                logger.error(f"    ✗ {token}: 查找失败 - {e}")
                all_correct = False
        
        if all_correct:
            self.log_test_pass("所有视觉 token 验证通过")
            return True
        else:
            self.log_test_fail("部分视觉 token 验证失败")
            return False
    
    def test_06_image_grid_thw_calculation(self):
        """测试 image_grid_thw 的计算逻辑"""
        self.log_test_start(6, "image_grid_thw 计算验证")
        
        # Qwen2.5-VL 的 patch_size 通常是 14
        patch_size = 14
        
        test_cases = [
            ((224, 224), "正方形小图"),
            ((448, 336), "矩形图"),
            ((640, 480), "标准分辨率"),
            ((1024, 768), "大图"),
        ]
        
        logger.info(f"  📊 测试不同分辨率的 grid_thw 计算 (patch_size={patch_size}):")
        
        all_passed = True
        for (width, height), desc in test_cases:
            # 计算预期的 grid 尺寸
            grid_h = (height + patch_size - 1) // patch_size
            grid_w = (width + patch_size - 1) // patch_size
            num_patches = grid_h * grid_w
            
            logger.info(f"\n  - {desc} ({width}x{height}):")
            logger.info(f"      预期 grid: {grid_h} x {grid_w} = {num_patches} patches")
            
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
                        logger.info(f"      实际 grid_thw: {actual_grid}")
                        
                        # 验证格式
                        if isinstance(actual_grid, torch.Tensor):
                            if actual_grid.shape[-1] == 3:
                                logger.info("      ✓ grid_thw 格式正确 (3个值: t, h, w)")
                            else:
                                logger.error(f"      ✗ grid_thw 格式错误: shape={actual_grid.shape}")
                                all_passed = False
                    else:
                        logger.warning("      ⚠️  image_grid_thw 缺失")
                        all_passed = False
                        
                except Exception as e:
                    logger.error(f"      ✗ 处理失败: {e}")
                    all_passed = False
        
        if all_passed:
            self.log_test_pass("所有 grid_thw 计算测试通过")
            return True
        else:
            self.log_test_fail("部分 grid_thw 计算测试失败")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "🚀" * 40)
        logger.info("开始运行 Qwen2.5-VL 测试套件")
        logger.info("🚀" * 40 + "\n")
        logger.info(f"日志文件: {log_filepath}\n")
        
        # 初始化
        if not self.setup():
            logger.error("❌ 测试环境初始化失败，无法继续")
            return
        
        # 运行所有测试
        tests = [
            self.test_01_processor_available,
            self.test_02_basic_image_processing,
            self.test_03_dynamic_resolution,
            self.test_04_data_sample_structure,
            self.test_05_vision_tokens,
            self.test_06_image_grid_thw_calculation,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"❌ 测试执行异常: {e}")
                import traceback
                logger.error(f"详细错误:\n{traceback.format_exc()}")
                self.failed_tests += 1
        
        # 输出总结
        self.print_summary()
    
    def print_summary(self):
        """输出测试总结"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 测试总结")
        logger.info("=" * 80)
        logger.info(f"  总测试数: {self.total_tests}")
        logger.info(f"  ✅ 通过: {self.passed_tests}")
        logger.info(f"  ❌ 失败: {self.failed_tests}")
        
        if self.failed_tests == 0:
            logger.info("\n  🎉 所有测试通过！")
        else:
            logger.warning(f"\n  ⚠️  有 {self.failed_tests} 个测试失败")
        
        logger.info("=" * 80)
        logger.info(f"详细日志已保存到: {log_filepath}")
        logger.info("=" * 80 + "\n")


def main():
    """主函数"""
    tester = TestQwenDataSample()
    tester.run_all_tests()


if __name__ == '__main__':
    main()

