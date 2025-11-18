# 🔧 Qwen2.5-VL Messages API 修复

## 📋 问题诊断

通过调试日志，我们发现了问题的根源：

```
DEBUG: Found input_ids_with_vision in image_data
DEBUG: vision_input_ids shape: torch.Size([3])  ← 只有3个token！
DEBUG: vision_start_idx: tensor([], dtype=torch.int64), vision_end_idx: tensor([], dtype=torch.int64)
WARNING: Vision start/end tokens not found in processor output!
```

**问题**：
- `input_ids_with_vision` 只有 **3 个 token**
- 这3个token只是 `"<image>"` 的普通 tokenization
- **不包含任何 vision tokens** (vision_start, image_pad, vision_end)

## 🔍 根本原因

我们之前的代码这样调用 processor：

```python
# ❌ 错误的方式
inputs = self.processor(
    text=["<image>"],  # 只是普通文本
    images=[image],
    return_tensors="pt",
)
```

**问题**：
- Qwen2.5-VL 的 processor 把 `"<image>"` 当作**普通文本**处理
- 没有将它识别为**图像 placeholder**
- 因此不会插入 vision tokens

## ✅ 正确的解决方案

Qwen2.5-VL 需要使用 **messages API** 格式：

### 修复代码

```python
# ✅ 正确的方式
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# 使用 apply_chat_template 生成正确的 prompt
text_prompt = self.processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 然后处理
inputs = self.processor(
    text=[text_prompt],
    images=[img],
    return_tensors="pt",
    padding=False,
)
```

### 关键步骤

1. **创建 messages 格式**
   ```python
   messages = [
       {
           "role": "user",
           "content": [
               {"type": "image", "image": img},  # 指定这是图像
               {"type": "text", "text": "..."}   # 配套的文本
           ]
       }
   ]
   ```

2. **应用 chat template**
   ```python
   text_prompt = processor.apply_chat_template(messages, ...)
   ```
   这会生成包含正确 placeholder 的提示，例如：
   ```
   <|im_start|>system
   You are a helpful assistant.<|im_end|>
   <|im_start|>user
   <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>Describe this image.<|im_end|>
   <|im_start|>assistant
   ```

3. **处理图像和文本**
   ```python
   inputs = processor(text=[text_prompt], images=[img], ...)
   ```
   现在 processor 知道在哪里插入 vision tokens

## 📊 修复效果

### 修复前 ❌
```
input_ids: [token1, token2, token3]  # 只有3个 token，不含 vision tokens
                                     # "< image >"
↓
ValueError: Image features and image tokens do not match: tokens: 0, features 391
```

### 修复后 ✅
```
input_ids: [system_tokens] + [vision_start] + [image_pad x N] + [vision_end] + [text_tokens]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              正确的 vision tokens（N+2 个）

↓
模型正常处理！
```

## 🛠️ 修改的文件

### `flmm/datasets/qwen_image_processor.py`

**主要变更**：

```python
def preprocess(self, image, text=None):
    # 为每个图像创建 messages 格式
    messages_list = []
    for i, img in enumerate(images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": texts[i] or "Describe this image."}
                ]
            }
        ]
        messages_list.append(messages)
    
    # 处理每个图像
    all_input_ids = []
    all_pixel_values = []
    all_image_grid_thw = []
    
    for messages in messages_list:
        # 应用 chat template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理图像和文本
        inputs = self.processor(
            text=[text_prompt],
            images=[messages[0]["content"][0]["image"]],
            return_tensors="pt",
            padding=False,
        )
        
        all_input_ids.append(inputs['input_ids'])
        all_pixel_values.append(inputs['pixel_values'])
        if 'image_grid_thw' in inputs:
            all_image_grid_thw.append(inputs['image_grid_thw'])
    
    # 合并并返回
    input_ids_list = torch.cat(all_input_ids, ...) if len(all_input_ids) > 1 else all_input_ids[0]
    pixel_values = torch.cat(all_pixel_values, ...) if len(all_pixel_values) > 1 else all_pixel_values[0]
    ...
```

## 🎯 关键要点

1. **Messages API 是必需的**
   - Qwen2.5-VL 需要使用 messages 格式
   - 不能简单地传递 text + images

2. **apply_chat_template 是关键**
   - 它生成包含正确 vision placeholders 的文本
   - processor 根据这个文本插入 vision tokens

3. **类型标注很重要**
   - `{"type": "image", "image": img}` 告诉 processor 这是图像
   - `{"type": "text", "text": "..."}` 告诉 processor 这是文本

## 🧪 验证

现在重新运行训练：

```bash
cd /home/cvprtemp/gyk/F-LMM
xtuner train configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py
```

**预期调试输出**：
```
DEBUG: Found input_ids_with_vision in image_data
DEBUG: vision_input_ids shape: torch.Size([XXX])  ← 应该有很多 token
DEBUG: vision_start_idx: tensor([YY]), vision_end_idx: tensor([ZZ])  ← 应该找到
DEBUG: Extracted vision_tokens length: N
DEBUG: Replaced <image> token with vision tokens at position P
DEBUG: New input_ids length: XXX
```

**预期结果**：
- ✅ vision_input_ids 应该有几百个 token（不只是3个）
- ✅ 应该找到 vision_start 和 vision_end
- ✅ 应该成功替换 <image> token
- ✅ 训练正常进行，不再有 ValueError

## 📚 相关资源

- [Qwen2.5-VL 官方文档](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Transformers Vision Language Models](https://huggingface.co/docs/transformers/model_doc/qwen2_vl)

---

**修复时间**: 2025-11-08 06:30+  
**影响文件**: 1个（qwen_image_processor.py）  
**严重性**: 🔥 关键（blocking训练）  
**状态**: ✅ 已修复，等待验证

## 🎊 修复历程

从最初的问题到现在，我们已经修复了：
1. ✅ `image_grid_thw` 缺失问题
2. ✅ pixel_values 格式误解问题  
3. ✅ vision tokens 缺失问题（processor API 使用不当）

第三次修复应该是最后一次了！🚀

