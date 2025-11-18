"""
快速诊断脚本：检查 image_grid_thw 问题

主要用途：
1. 快速验证 processor 输出
2. 检查 data_sample 中是否包含 image_grid_thw
3. 提供修复建议
"""

import torch
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def diagnose_processor():
    """诊断 processor 的输出"""
    print("=" * 70)
    print("诊断 Qwen Processor 输出")
    print("=" * 70)
    
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        print("✓ Processor 加载成功\n")
    except Exception as e:
        print(f"✗ Processor 加载失败: {e}")
        return None
    
    # 测试不同的输入格式
    image = Image.new('RGB', (640, 480), color='blue')
    
    print("测试 1: 使用 messages 格式")
    print("-" * 70)
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"格式化文本: {text[:100]}...")
        
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        
        print(f"\n输出 keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: type={type(value)}, value={value}")
        
        if 'image_grid_thw' in inputs:
            print(f"\n✓ image_grid_thw 存在！")
            print(f"  值: {inputs['image_grid_thw']}")
            print(f"  形状: {inputs['image_grid_thw'].shape if isinstance(inputs['image_grid_thw'], torch.Tensor) else 'N/A'}")
        else:
            print(f"\n✗ image_grid_thw 缺失！")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试 2: 直接使用文本格式")
    print("-" * 70)
    try:
        # 尝试直接使用带图像标记的文本
        text = "<|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        
        print(f"输出 keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}")
            else:
                print(f"  - {key}: {value}")
        
        if 'image_grid_thw' in inputs:
            print(f"\n✓ image_grid_thw 存在: {inputs['image_grid_thw']}")
        else:
            print(f"\n✗ image_grid_thw 缺失")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试 3: 检查 image_processor 的行为")
    print("-" * 70)
    try:
        image_processor = processor.image_processor
        print(f"Image processor 类型: {type(image_processor)}")
        print(f"Image processor 属性:")
        
        # 检查相关属性
        attrs = ['patch_size', 'size', 'do_resize', 'do_rescale']
        for attr in attrs:
            if hasattr(image_processor, attr):
                print(f"  - {attr}: {getattr(image_processor, attr)}")
        
        # 直接处理图像
        print(f"\n直接处理图像:")
        image_inputs = image_processor(images=image, return_tensors="pt")
        print(f"输出 keys: {list(image_inputs.keys())}")
        for key, value in image_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}")
            else:
                print(f"  - {key}: {value}")
                
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    return processor


def check_data_sample_structure(processor):
    """检查 data_sample 的结构"""
    if processor is None:
        print("Processor 不可用，跳过检查")
        return
    
    print("\n" + "=" * 70)
    print("检查 data_sample 结构")
    print("=" * 70)
    
    image = Image.new('RGB', (640, 480), color='green')
    
    try:
        # 模拟 dataset 的处理流程
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Segment the object."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # 构建 data_sample
        data_sample = {
            'input_ids': inputs['input_ids'][0],  # [seq_len]
            'pixel_values': inputs['pixel_values'],  # [1, C, H, W]
            'image': image,
            'masks': torch.zeros(1, 480, 640),
            'mask_ids': torch.zeros(inputs['input_ids'].shape[1], dtype=torch.long),
            'meta_data': {
                'image_shape': {'height': 480, 'width': 640},
                'padded_shape': {'height': 480, 'width': 640},
                'padding': {'before_height': 0, 'before_width': 0}
            }
        }
        
        # 添加可选字段
        optional_fields = ['attention_mask', 'image_grid_thw']
        for field in optional_fields:
            if field in inputs:
                if field == 'attention_mask':
                    data_sample[field] = inputs[field][0]
                else:
                    data_sample[field] = inputs[field]
                print(f"✓ 添加 {field}: {data_sample[field]}")
            else:
                print(f"✗ {field} 不在 processor 输出中")
        
        print(f"\ndata_sample 包含的字段:")
        for key, value in data_sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, Image.Image):
                print(f"  - {key}: PIL.Image {value.size}")
            elif isinstance(value, dict):
                print(f"  - {key}: dict with keys {list(value.keys())}")
            else:
                print(f"  - {key}: type={type(value)}")
        
        # 检查是否满足模型要求
        print(f"\n模型要求验证:")
        required_for_model = ['input_ids', 'pixel_values', 'image_grid_thw']
        all_present = True
        for field in required_for_model:
            if field in data_sample:
                print(f"  ✓ {field}")
            else:
                print(f"  ✗ {field} - MISSING!")
                all_present = False
        
        if all_present:
            print(f"\n✓ 所有必需字段都存在")
        else:
            print(f"\n✗ 缺少必需字段，这将导致模型错误")
            print(f"\n建议修复方案:")
            print(f"1. 确保使用正确的 transformers 版本（建议 >= 4.37.0）")
            print(f"2. 在 dataset 中显式添加 image_grid_thw")
            print(f"3. 检查 processor 的配置")
        
        return data_sample
        
    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def suggest_fixes():
    """提供修复建议"""
    print("\n" + "=" * 70)
    print("修复建议")
    print("=" * 70)
    
    print("""
如果 image_grid_thw 缺失，可以尝试以下修复方法：

方法 1: 在 Dataset 中手动计算 image_grid_thw
-----------------------------------------------
在 flmm/datasets/transforms.py 或相关 dataset 文件中：

```python
def calculate_image_grid_thw(pixel_values, patch_size=14):
    \"\"\"
    计算 image_grid_thw
    
    Args:
        pixel_values: [C, H, W] 或 [1, C, H, W]
        patch_size: patch 大小，Qwen2.5-VL-3B 使用 14
    
    Returns:
        image_grid_thw: [1, 3] tensor, 格式为 [temporal, height_grids, width_grids]
    \"\"\"
    if pixel_values.dim() == 4:
        _, _, h, w = pixel_values.shape
    else:
        _, h, w = pixel_values.shape
    
    grid_h = (h + patch_size - 1) // patch_size
    grid_w = (w + patch_size - 1) // patch_size
    
    # Qwen2.5-VL 使用 [1, grid_h, grid_w] 格式
    # temporal 维度对于图像通常是 1
    return torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
```

然后在 __getitem__ 中添加：
```python
data_sample['image_grid_thw'] = calculate_image_grid_thw(data_sample['pixel_values'])
```

方法 2: 在模型的 _forward 中添加后备计算
---------------------------------------
在 flmm/models/frozen_qwen.py 的 _forward 方法中：

```python
# 在调用 qwen_model 之前
if 'image_grid_thw' not in data_sample or data_sample['image_grid_thw'] is None:
    # 计算 image_grid_thw
    pixel_values = data_sample['pixel_values']
    if pixel_values.dim() == 4:
        _, _, h, w = pixel_values.shape
    else:
        _, h, w = pixel_values.shape
    
    grid_h = (h + self.patch_size - 1) // self.patch_size
    grid_w = (w + self.patch_size - 1) // self.patch_size
    
    data_sample['image_grid_thw'] = torch.tensor(
        [[1, grid_h, grid_w]], 
        dtype=torch.long,
        device=self.qwen_model.device
    )
    print_log(f"Warning: image_grid_thw was missing, calculated as {data_sample['image_grid_thw']}")
```

方法 3: 使用更新的 transformers 版本
------------------------------------
确保安装最新版本的 transformers：
```bash
pip install --upgrade transformers>=4.37.0
```

方法 4: 检查 QwenImageProcessorWrapper
-------------------------------------
如果使用了自定义的 wrapper，确保它正确传递 image_grid_thw：

```python
class QwenImageProcessorWrapper:
    def __call__(self, *args, **kwargs):
        outputs = self.processor(*args, **kwargs)
        
        # 确保 image_grid_thw 存在
        if 'image_grid_thw' not in outputs:
            # 计算并添加
            ...
        
        return outputs
```
""")


if __name__ == '__main__':
    # 运行诊断
    processor = diagnose_processor()
    
    # 检查 data_sample
    data_sample = check_data_sample_structure(processor)
    
    # 提供修复建议
    suggest_fixes()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)

