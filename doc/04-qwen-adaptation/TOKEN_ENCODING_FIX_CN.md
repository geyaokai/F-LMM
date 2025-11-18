# Qwen Tokenizer `<image>` Token 编码问题及修复

## 问题发现

训练日志显示：
```
Image token ids: [27, 1805, 29], decoded: <image>
...
WARNING: No <image> token found in input_ids to replace!
ValueError: Image features and image tokens do not match: tokens: 0, features 345
```

## 根本原因

通过测试发现，Qwen tokenizer 使用**贪心合并策略**：

### 单独编码
```python
tokenizer.encode("<image>", add_special_tokens=False)
# 结果: [27, 1805, 29]  (<, image, >)
```

### 在句子中编码
```python
tokenizer.encode("<image>Please give me...", add_special_tokens=False)
# 结果: [27, 1805, 66304, ...]  (<, image, >Please, ...)
#                  ^^^^^^
#       注意：'>' 和 'Please' 被合并成单个 token '>Please' (ID: 66304)
```

**关键问题**：在实际 prompt 中，`>` 后紧跟其他文本时，tokenizer 会将它们合并，导致：
- 单独编码的 `<image>` = `[27, 1805, 29]`
- 实际句子中的 `<image>` ≠ `[27, 1805, 29]`（第三个token被合并了）

因此，手动查找替换 `[27, 1805, 29]` 序列**永远找不到**！

## 解决方案

### 旧方案（失败）
```python
# 步骤1: 单独编码 <image>
image_token_ids = tokenizer.encode("<image>", add_special_tokens=False)

# 步骤2: 在 prompt 中查找这个序列
img_pos = find_position(input_ids, image_token_ids)  # 找不到！

# 步骤3: 替换为 vision tokens
# ...（永远执行不到）
```

### 新方案（正确）
```python
# 直接使用 Qwen processor 生成的完整 input_ids
# processor 会正确处理 <image> 并插入 vision tokens
image_data = processor.preprocess(image, text="<image>Please...")

# 使用 processor 返回的完整序列（已包含 vision tokens）
input_ids_with_vision = image_data['input_ids_with_vision'][0]

# 直接使用，不需要手动查找替换
input_ids = input_ids_with_vision + caption_input_ids
```

## 修改的文件

### 1. `/data/gyk/F-LMM/flmm/datasets/png.py`

**修改前**：
- 尝试在 `self.prompt + caption_input_ids` 中查找 `[27, 1805, 29]`
- 失败，因为实际编码不匹配

**修改后**：
- 调用 `processor.preprocess(image, text=simple_prompt)` 获取完整的 input_ids
- 直接使用 `input_ids_with_vision` 替换 `self.prompt` 部分
- 拼接 caption：`vision_input_ids_with_prompt + caption_input_ids`

### 2. `/data/gyk/F-LMM/flmm/datasets/transforms.py`

**修改前**：
- 相同的查找替换逻辑
- 相同的失败原因

**修改后**：
- 使用 `processor.preprocess(image, text=prompt)` 
- 直接使用返回的完整 input_ids

## 关键代码片段

```python
# png.py 和 transforms.py 中的新逻辑
if 'input_ids_with_vision' in image_data:
    vision_input_ids_with_prompt = image_data['input_ids_with_vision'][0]
    # 转换为 list
    if isinstance(vision_input_ids_with_prompt, np.ndarray):
        vision_input_ids_with_prompt = vision_input_ids_with_prompt.tolist()
    elif isinstance(vision_input_ids_with_prompt, torch.Tensor):
        vision_input_ids_with_prompt = vision_input_ids_with_prompt.tolist()
    
    # 替换 prompt 部分，保留 caption 部分
    input_ids = vision_input_ids_with_prompt + caption_input_ids
    
    # 更新 mask_ids
    vision_prompt_len = len(vision_input_ids_with_prompt)
    mask_ids = [-1] * vision_prompt_len + mask_ids[len(self.prompt):]
else:
    # 非 Qwen 模型：使用原有逻辑
    input_ids = self.prompt + caption_input_ids
```

## 测试验证

测试脚本 `/data/gyk/F-LMM/tests/test_image_token_encoding.py` 清楚地展示了问题：

```
Test 1: Encode '<image>' alone (no special tokens)
  Token IDs: [27, 1805, 29]

Test 2: Encode in sentence (no special tokens)
  Token IDs (first 10): [27, 1805, 66304, ...]
  Looking for [27, 1805, 29] in prompt_ids...
  NOT FOUND!
    Token 0: 27 -> '<'
    Token 1: 1805 -> 'image'
    Token 2: 66304 -> '>Please'  # 注意这里！
```

## 经验教训

1. **不要假设 tokenizer 的编码是上下文无关的**
   - BPE/WordPiece tokenizer 使用贪心合并
   - 相同文本在不同上下文中可能编码不同

2. **对于视觉-语言模型，优先使用 processor 的完整输出**
   - Processor 知道如何正确插入 special tokens
   - 不要尝试"聪明"地手动替换

3. **充分测试实际编码行为**
   - 单独测试 vs 上下文测试可能有差异
   - 用真实数据验证假设

## 影响范围

- **修复**：Qwen 模型现在能正确处理 vision tokens
- **兼容性**：非 Qwen 模型保持原有逻辑不变
- **性能**：无影响，只是改变了构建 input_ids 的方式

## 日期
2025-11-08

