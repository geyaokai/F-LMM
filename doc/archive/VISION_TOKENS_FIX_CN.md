# 🔧 修复：Qwen2.5-VL Vision Tokens 问题

## 📋 新问题

经过之前的修复后，训练出现了新的错误：

```
ValueError: Image features and image tokens do not match: tokens: 0, features 252
ValueError: Image features and image tokens do not match: tokens: 0, features 391
```

**这是好消息！** 之前的 `RuntimeError` 已经解决了。

## 🔍 问题分析

### 错误含义

- `tokens: 0` - `input_ids` 中**没有**图像相关的特殊 token
- `features 252/391` - 模型生成了图像特征

### 根本原因

Qwen2.5-VL 需要在 `input_ids` 中包含特殊的 **vision tokens**：
- `<|vision_start|>` (token ID: 151652)
- `<|image_pad|>` x N (token ID: 151655) - 数量取决于图像的 patches 数量
- `<|vision_end|>` (token ID: 151653)

**我们的数据管道问题**：
- 文本通过 `tokenizer.encode(text)` 单独处理
- 图像通过 `image_processor.preprocess(image)` 单独处理
- **Qwen 的 processor 需要同时接收图像和文本才能正确插入 vision tokens！**

## ✅ 解决方案

### 方案概述

1. **修改 `QwenImageProcessorWrapper`**: 让它处理文本（包含 `<image>` placeholder）
2. **提取 vision tokens**: 从 processor 生成的 `input_ids` 中提取
3. **插入到数据管道**: 在原有的 `input_ids` 中替换 `<image>` token

### 详细实现

#### 1. 修改 `flmm/datasets/qwen_image_processor.py`

**关键变更**：

```python
def preprocess(self, image, text=None):
    # 如果没有提供文本，使用 "<image>" placeholder
    if text is None or not text:
        processor_texts = ["<image>"] * len(images)
    else:
        processor_texts = texts
    
    # 同时处理图像和文本
    inputs = self.processor(
        text=processor_texts,
        images=images,
        return_tensors="pt",
        padding=False,
    )
    
    # 提取结果
    pixel_values = inputs['pixel_values']
    image_grid_thw = inputs['image_grid_thw']
    input_ids_with_vision = inputs['input_ids']  # 包含 vision tokens!
    
    # ... 返回所有内容
    result['input_ids_with_vision'] = [input_ids_np[i] for i in range(len(images))]
```

#### 2. 修改 `flmm/datasets/transforms.py` 和 `flmm/datasets/png.py`

**关键变更**：

```python
# 处理完图像后
image_data = self.image_processor.preprocess(image)

# 检查是否有 vision tokens
if 'input_ids_with_vision' in image_data:
    vision_input_ids = image_data['input_ids_with_vision'][0]
    
    # 提取 vision tokens (从 <|vision_start|> 到 <|vision_end|>)
    vision_start_id = 151652
    vision_end_id = 151653
    
    vision_start_idx = (vision_input_ids == vision_start_id).nonzero()[0]
    vision_end_idx = (vision_input_ids == vision_end_id).nonzero()[0]
    
    if len(vision_start_idx) > 0 and len(vision_end_idx) > 0:
        vision_tokens = vision_input_ids[vision_start_idx[0]:vision_end_idx[0]+1]
        
        # 在 input_ids 中找到 <image> token 并替换为 vision tokens
        image_token_positions = (input_ids == self.image_token_idx).nonzero()[0]
        
        if len(image_token_positions) > 0:
            img_pos = image_token_positions[0]
            input_ids = torch.cat([
                input_ids[:img_pos],
                vision_tokens,
                input_ids[img_pos+1:]
            ])
            
            # 同样更新 mask_ids
            vision_mask_ids = torch.full((len(vision_tokens),), -1)
            mask_ids = torch.cat([
                mask_ids[:img_pos],
                vision_mask_ids,
                mask_ids[img_pos+1:]
            ])
```

## 📊 修复效果

### 修复前

```
input_ids: [prompt_tokens] + [<image>] + [text_tokens]
                              ^^^^^^^^
                           单个 token，没有 vision 信息

模型：我看到了图像特征 (252 个)，但 input_ids 中没有对应的 vision tokens！
结果：ValueError ❌
```

### 修复后

```
input_ids: [prompt_tokens] + [<|vision_start|>] + [<|image_pad|> x N] + [<|vision_end|>] + [text_tokens]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              正确的 vision tokens (总共 N+2 个)

模型：input_ids 中有 N+2 个 vision tokens，匹配图像特征！
结果：正常处理 ✅
```

## 🎯 关键要点

1. **Qwen 的特殊性**
   - 必须在 `input_ids` 中包含 vision tokens
   - 不能只提供 `pixel_values` 和 `image_grid_thw`

2. **Processor 的正确使用**
   - 必须同时传递图像和文本（即使文本只是 "<image>"）
   - Processor 会自动生成正确的 vision tokens

3. **数据管道的适配**
   - 提取 processor 生成的 vision tokens
   - 替换原有 `input_ids` 中的 `<image>` placeholder

## 🧪 验证

现在请测试：

```bash
cd F-LMM
xtuner train configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py
```

**预期结果**：
- ✅ 不再出现 "Image features and image tokens do not match"
- ✅ 训练正常进行
- ✅ `input_ids` 包含正确的 vision tokens

## 📚 相关文件

### 修改的文件
1. `flmm/datasets/qwen_image_processor.py` - 添加text参数，提取input_ids_with_vision
2. `flmm/datasets/transforms.py` - 插入 vision tokens
3. `flmm/datasets/png.py` - 插入 vision tokens

### 相关文档
- `tests/CRITICAL_FIX_CN.md` - pixel_values 格式修复
- `tests/FIX_SUMMARY_CN.md` - 原始 image_grid_thw 修复
- `tests/QWEN_PIXEL_VALUES_FORMAT.md` - Qwen 格式说明

---

**修复时间**: 2025-11-08 05:30+  
**影响文件**: 3个  
**严重性**: 🔥 关键（blocking训练）  
**状态**: ✅ 已修复，等待验证

## 🎉 进展总结

从最初的错误到现在，我们已经修复了：
1. ✅ `image_grid_thw` 缺失问题
2. ✅ pixel_values 格式误解问题
3. ✅ vision tokens 缺失问题

现在应该可以正常训练了！🚀

