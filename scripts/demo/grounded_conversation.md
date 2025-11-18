# Grounded Conversation 使用指南

本文档介绍如何使用 `grounded_conversation.py` 脚本实现**交互式的视觉对话与 Grounding** 功能。该脚本可以：
- 让模型回答关于图像的问题
- 自动从模型回答中提取名词短语
- 交互式选择需要 grounding 的短语
- 生成分割 mask 并可视化

## 前置依赖

### 1. 安装 spaCy 和语言模型

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 2. 模型支持

目前主要支持 **DeepSeekVL** 模型（实现了 `answer` 和 `ground` 方法）。其他模型（LLaVA、MGM 等）可能不支持这些方法。

## 快速开始

```bash
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.

# 基本运行（使用默认参数）
python scripts/demo/grounded_conversation.py \
    configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py \
    --checkpoint checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth \
    --image data/coco/val2017/000000000632.jpg \
    --text "Where is the shampoo?"

# 使用 SAM 细化（更精确的边界）
python scripts/demo/grounded_conversation.py \
    configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py \
    --checkpoint checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth \
    --image data/coco/val2017/000000000632.jpg \
    --text "What objects are in this image?" \
    --use_sam
```

## 参数说明

- `config`（必需）：配置文件路径，建议使用 DeepSeekVL 配置
- `--checkpoint`：模型权重路径（默认：`checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth`）
- `--image`：输入图像路径（默认：`data/coco/val2017/000000000632.jpg`）
- `--text`：问题文本（默认：`"Where is the shampoo?"`）
- `--use_sam`：可选，使用 SAM 细化 mask（边界更精确）

## 代码流程详解

### 整体流程图

```
用户输入问题 
    ↓
模型生成回答文本 (output_text)
    ↓
从回答文本中提取名词短语 (spaCy)
    ↓
过滤处理 (去除停用词、去重)
    ↓
交互式询问用户 (输入 1/0)
    ↓
将选中短语转换为 token 位置
    ↓
Grounding (基于 token 位置提取注意力 → 生成 mask)
    ↓
可视化并保存结果
```

### 详细步骤说明

#### 步骤 1-2: 模型生成回答

```101:104:scripts/demo/grounded_conversation.py
    image = Image.open(args.image)
    output = model.answer(image, args.text)
    output_ids = output.pop('output_ids').cpu()
    output_text = output.pop('output_text')
```

**关键点**：模型接收用户问题（如 `"Where is the shampoo?"`），生成回答文本（`output_text`）。

**示例**：
- 用户问题：`"Where is the shampoo?"`
- 模型回答：`"The shampoo is on the dresser next to the sink."`

#### 步骤 3: 提取名词短语

```38:61:scripts/demo/grounded_conversation.py
def extract_noun_phrases(output_text):
    doc = nlp(output_text)
    noun_chunks = list(set(chunk.text for chunk in doc.noun_chunks))
    if len(noun_chunks) == 0:
        noun_chunks = [output_text]
    last_end = 0
    noun_chunks = process_noun_chunks(noun_chunks)
    noun_chunks = sorted(noun_chunks, key=lambda x: output_text.find(x))

    noun_chunks = [noun_chunk for noun_chunk in noun_chunks
                   if int(input(f'Ground {noun_chunk}?')) == 1]

    positive_ids = []
    phrases = []
    for noun_chunk in noun_chunks:
        obj_start = output_text.find(noun_chunk)
        if obj_start < last_end:
            continue
        obj_end = obj_start + len(noun_chunk)
        last_end = obj_end
        positive_ids.append((obj_start, obj_end))
        phrases.append(noun_chunk)

    return positive_ids, phrases
```

**流程**：
1. **spaCy 提取**（第 40 行）：使用 `en_core_web_sm` 模型从回答文本中提取所有名词短语
2. **过滤处理**（第 44 行）：调用 `process_noun_chunks` 去除停用词和去重
3. **交互选择**（第 47-48 行）：对每个名词短语询问用户是否要 grounding

**交互示例**：
```
Ground shampoo? 1      # 输入 1 表示要 grounding
Ground dresser? 1      # 输入 1
Ground sink? 0         # 输入 0 表示跳过
```

#### 步骤 4: 过滤处理机制

```16:35:scripts/demo/grounded_conversation.py
def process_noun_chunks(noun_chunks):
    new_noun_chunks = []
    for i in range(len(noun_chunks)):
        noun_chunk = noun_chunks[i]
        if 'image' in noun_chunk.lower():
            continue
        if noun_chunk.lower() in ['it', 'this', 'that', 'those', 'these', 'them',
                                  'he', 'she', 'you', 'i', 'they', 'me', 'her',
                                  'him', 'a', 'what', 'which', 'whose', 'who']:
            continue
        keep = True
        for j in range(len(noun_chunks)):  # de-duplicate
            if i != j and noun_chunk in noun_chunks[j]:
                if len(noun_chunk) < len(noun_chunks[j]) or i > j:
                    keep = False
                    break
        if keep:
            new_noun_chunks.append(noun_chunk)

    return new_noun_chunks
```

**过滤规则**：
- ✅ **跳过包含 "image" 的短语**：避免 grounding 图像 token
- ✅ **跳过代词和停用词**：`it`, `this`, `that`, `he`, `she` 等无具体指代
- ✅ **去重处理**：如果短语 A 包含在短语 B 中，保留更长的 B

**去重示例**：
- 输入：`["shampoo", "the shampoo", "dresser"]`
- 输出：`["the shampoo", "dresser"]`（"shampoo" 被 "the shampoo" 包含，已去除）

#### 步骤 5: Token 位置映射

```106:114:scripts/demo/grounded_conversation.py
    encoded = model.tokenizer(output_text, add_special_tokens=False, return_tensors='pt')
    assert (encoded.input_ids[0] == output_ids).all()
    offsets = encoded.encodings[0].offsets
    str_places, phrases = extract_noun_phrases(output_text)
    positive_ids = []
    for start_id, end_id in str_places:
        start_token_place = find_interval(offsets, start_id)
        end_token_place = max(start_token_place+1, find_interval(offsets, end_id))
        positive_ids.append((start_token_place, end_token_place))
```

**关键转换**：
- **输入**：字符位置 `(start_id, end_id)`，例如 `"shampoo"` 在文本中的字符位置 `(10, 17)`
- **输出**：token 位置 `(start_token_place, end_token_place)`，例如 token 索引 `(5, 6)`
- **目的**：模型内部使用 token 位置，需要将文本位置转换为 token 位置

#### 步骤 6: Grounding 与可视化

```115:129:scripts/demo/grounded_conversation.py
    with torch.no_grad():
        pred_masks, sam_pred_masks = model.ground(image=image, positive_ids=positive_ids, **output)
    if args.use_sam:
        masks = sam_pred_masks.cpu().numpy() > 0
    else:
        masks = pred_masks.cpu().numpy() > 0

    image_np = np.array(image).astype(np.float32)
    for color_id, mask in enumerate(masks):
        image_np[mask] = image_np[mask] * 0.2 + np.array(colors[color_id]).reshape((1, 1, 3)) * 0.8

    image = Image.fromarray(image_np.astype(np.uint8))
    print(output_text, flush=True)
    print(phrases, flush=True)
    os.makedirs('F-LMM/scripts/demo/results', exist_ok=True)
    image.save('F-LMM/scripts/demo/results/example.jpg')
```

**Grounding 过程**：
1. 使用 `positive_ids`（token 位置）在模型回答对应的 token 上提取注意力特征
2. 将注意力特征输入 mask head（UNet）生成初始 mask
3. 如果使用 `--use_sam`，SAM 会进一步细化 mask 边界

**可视化**：
- 每个选中的短语对应一个不同颜色的 mask
- 结果保存在 `F-LMM/scripts/demo/results/example.jpg`
- 打印模型的回答文本和选中短语列表

## 重要设计决策：关键词提取机制

### ⚠️ 关键理解：从回答中提取，而非从问题中提取

**当前实现**：
- ✅ **提取源**：模型的回答文本（`output_text`）
- ❌ **不是从**：用户的问题文本（`args.text`）

### 为什么会出现关键词不一致？

#### 案例 1: 问题与回答不一致

```
用户问题："Where is the shampoo?"
模型回答："I don't see shampoo, but I can see some dresses on the rack."
提取结果：["dresses", "rack"]  ← 没有 "shampoo"！
```

**原因**：
- 模型回答中没有提到 "shampoo"
- 代码从回答文本中提取，所以提取到的是 "dresses" 和 "rack"

#### 案例 2: 回答包含额外信息

```
用户问题："Where is the shampoo?"
模型回答："The shampoo is on the dresser next to the sink."
提取结果：["shampoo", "dresser", "sink"]
```

**原因**：
- 回答中提到了多个名词，都会被提取
- 包括问题中的 "shampoo"，也包括回答中新增的 "dresser" 和 "sink"

### 这种设计的原因

**设计意图**：Grounding 模型回答中提到的对象，而不是直接 grounding 问题中的关键词。

**优点**：
- 如果模型回答了其他相关对象，可以一并 grounding
- 更符合对话的自然流程

**缺点**：
- 如果模型回答错误或不相关，提取的关键词也会偏离
- 如果用户只想 grounding 问题中的关键词，当前无法直接实现

## 解决方案建议

如果需要基于问题中的关键词进行 grounding，可以考虑以下修改方案：

### 方案 A: 同时从问题和回答中提取（优先级：问题 > 回答）

```python
# 提取问题中的关键词（优先级高）
question_phrases = extract_noun_phrases(args.text)
# 提取回答中的关键词
answer_phrases = extract_noun_phrases(output_text)
# 合并，问题中的关键词优先
all_phrases = question_phrases + [p for p in answer_phrases if p not in question_phrases]
```

### 方案 B: 基于问题提取，回答仅作补充

```python
# 主要从问题中提取
main_phrases = extract_noun_phrases(args.text)
# 从回答中提取额外的（问题中没有的）作为候选
additional_phrases = [p for p in extract_noun_phrases(output_text) 
                      if p not in main_phrases]
# 让用户选择是否要 grounding 额外短语
```

### 方案 C: 用户直接指定要 grounding 的短语

```python
# 显示所有候选短语
question_candidates = extract_noun_phrases(args.text)
answer_candidates = extract_noun_phrases(output_text)
all_candidates = list(set(question_candidates + answer_candidates))

# 让用户直接输入要 grounding 的短语
print(f"Candidates: {all_candidates}")
user_input = input("Enter phrases to ground (comma-separated): ")
user_phrases = [p.strip() for p in user_input.split(',')]
```

### 方案 D: 添加命令行参数直接指定

```python
parser.add_argument('--ground_phrases', type=str, default=None,
                    help='Comma-separated phrases to ground directly')
# 如果指定了，直接使用，否则使用自动提取
if args.ground_phrases:
    phrases = [p.strip() for p in args.ground_phrases.split(',')]
else:
    phrases = extract_noun_phrases(output_text)
```

## 常见问题 (FAQ)

### Q1: 为什么提取的关键词和问题不一致？

**A**: 因为代码从模型的**回答文本**中提取关键词，而不是从问题中提取。如果模型的回答与问题不一致，提取的关键词也会不一致。

**解决方法**：运行前先打印 `output_text` 确认模型回答内容：
```python
print(output_text)  # 在 extract_noun_phrases 调用前添加
```

### Q2: 如何确保 grounding 问题中的关键词？

**A**: 目前代码不支持直接从问题中提取。可以：
1. 修改代码实现方案 A 或 B（见上）
2. 确保问题中的关键词在模型回答中出现（通过改进 prompt）
3. 手动在交互中选择包含问题关键词的短语（如果它在回答中）

### Q3: 模型回答为空或没有名词短语怎么办？

**A**: 代码第 41-42 行有处理：
```python
if len(noun_chunks) == 0:
    noun_chunks = [output_text]  # 使用整个回答文本
```
会使用整个回答文本作为候选。

### Q4: 可以批量处理多张图像吗？

**A**: 当前脚本不支持批量处理。需要：
1. 编写循环脚本调用该脚本
2. 或者修改代码支持批量输入

### Q5: 输出的结果在哪里？

**A**: 结果保存在 `F-LMM/scripts/demo/results/example.jpg`（第 130 行）。

## 实际运行示例

假设运行：
```bash
export PATHONPATH=.
python scripts/demo/grounded_conversation.py \
    configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py \
    --checkpoint checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth \
    --image data/coco/val2017/000000000632.jpg \
    --text "Where is the shampoo?"
```

**可能的交互过程**：
```
[模型加载中...]
[生成回答中...]

模型回答: "I can see a bottle of shampoo on the dresser, next to some other items."

Ground bottle? 0        # 输入 0 跳过
Ground shampoo? 1      # 输入 1 选择 grounding
Ground dresser? 1      # 输入 1 选择 grounding
Ground items? 0        # 输入 0 跳过

[生成 mask 中...]
[可视化完成]

输出文本: I can see a bottle of shampoo on the dresser, next to some other items.
选中短语: ['shampoo', 'dresser']
结果已保存到: F-LMM/scripts/demo/results/example.jpg
```

## 代码位置参考

- **主脚本**：`scripts/demo/grounded_conversation.py`
- **模型实现**：`flmm/models/frozen_deepseek_vl.py`（`answer` 和 `ground` 方法）
- **配置文件示例**：`configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`
