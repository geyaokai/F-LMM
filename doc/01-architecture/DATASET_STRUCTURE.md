# F-LMM 数据集结构详解

本文档详细解释 F-LMM 训练数据集的加载过程和 `data_sample` 中各字段的含义。

## 目录

- [数据集概述](#数据集概述)
- [数据加载流程](#数据加载流程)
- [data_sample 字段详解](#datasample-字段详解)
- [数据流程总结](#数据流程总结)
- [训练时的使用](#训练时的使用)

---

## 数据集概述

F-LMM 使用 **PNG (Panoptic Narrative Grounding)** 数据集进行训练，这是一个将自然语言描述与图像中的分割区域关联起来的数据集。

### 数据集配置

数据集配置在 `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py` 中定义：

```python
datasets_list = [
    dict(type=PNGDataset,
         json_file='data/coco/annotations/png_coco_train2017.json',
         panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
         panoptic_png_path='data/coco/annotations/panoptic_train2017',
         tokenizer=tokenizer,
         image_processor=image_processor,
         prompt_template=prompt_template,
         local_path='data/coco/train2017',
         prompt=prompt,
         image_token=image_token),
    # ... 其他数据集（RefCOCO 系列）
]
```

### 数据集大小

- **训练集大小**: 190,157 个样本
- **每个样本**: 包含一张图像和多个文本描述的分割区域（segments）

---

## 数据加载流程

### 1. 从配置构建数据集

```python
from mmengine.config import Config
from xtuner.registry import BUILDER

cfg = Config.fromfile('configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py')
dataset = BUILDER.build(cfg.train_dataloader.dataset)
```

**注意**: `cfg.train_dataloader.dataset` 是一个配置字典（ConfigDict），需要使用 `BUILDER.build()` 构建为实际的数据集对象。

### 2. 数据集初始化过程

`PNGDataset` 在初始化时会：

1. 加载 PNG 标注 JSON 文件（包含图像 ID 和 segments）
2. 加载 COCO Panoptic 标注（包含分割信息）
3. 初始化 tokenizer 和 image_processor
4. 构建 prompt 模板的 token IDs

### 3. 单个样本加载流程

当调用 `dataset[index]` 时，`__getitem__` 方法会：

1. 读取原始 PNG 数据样本
2. Tokenize prompt 和所有 segments 的文本
3. 构建 mask_ids（标记每个 token 属于哪个 mask）
4. 加载图像并进行预处理
5. 从 panoptic 分割图提取 mask
6. 调整 mask 尺寸并添加 padding
7. 构建训练标签

---

## data_sample 字段详解

### 字段概览

一个 `data_sample` 包含以下 12 个字段：

```python
{
    'input_ids': Tensor,        # Token 化的输入序列
    'mask_ids': Tensor,         # 每个 token 对应的 mask 索引
    'pixel_values': Tensor,    # 预处理后的图像张量
    'padded_masks': Tensor,    # 填充到模型输入尺寸的 mask
    'masks': Tensor,           # 调整到图像处理尺寸的 mask
    'gt_masks': Tensor,        # 原始图像尺寸的 ground truth mask
    'image_sizes': Tensor,     # 原始图像尺寸 [height, width]
    'mask_infos': list,        # Mask 的元信息
    'image': PIL.Image,        # 原始 PIL 图像对象
    'file_name': str,          # 图像文件名
    'meta_data': dict,         # 图像处理元数据
    'labels': Tensor           # 训练标签（用于语言模型）
}
```

---

### 1. `input_ids` - Token 化的输入序列

**形状**: `torch.Size([608])`, `dtype: torch.int64`

**生成过程**（`png.py:159`）:
```python
input_ids = self.prompt + caption_input_ids
input_ids = torch.tensor(input_ids, dtype=torch.long)
```

**含义**:
- `self.prompt`: 模板编码，包含：
  - Prompt 模板: `"User: {input}\n\nAssistant:"`
  - 图像占位符: `<image_placeholder>` × 576 次
  - 固定描述: `"Please give me a description of the image."`
- `caption_input_ids`: PNG 数据集中所有 segment 的 utterance 的 token IDs
- 总长度 = prompt 长度 + caption 长度

**示例结构**:
```
[prompt_tokens..., image_token×576, "Please", "give", "me", ..., 
 segment1_words..., segment2_words...]
```

---

### 2. `mask_ids` - 每个 token 对应的 mask 索引

**形状**: `torch.Size([608])`, `dtype: torch.int64`

**生成过程**（`png.py:118-129`）:
```python
mask_ids = [-1]*len(self.prompt)  # prompt 部分都是 -1（不关联任何 mask）

for segment in data_sample['segments']:
    segment_input_ids = self.tokenizer.encode(segment['utterance'], ...)
    if len(segment['segment_ids']) == 0:
        mask_ids += [-1] * len(segment_input_ids)  # 没有对应 mask
    else:
        mask_ids += [mask_cnt] * len(segment_input_ids)  # 关联到第 mask_cnt 个 mask
        mask_cnt += 1
```

**含义**:
- `-1`: 该 token 不关联任何 mask（prompt 部分或无 mask 的 segment）
- `0, 1, 2, ...`: 该 token 属于第 n 个 mask（从 0 开始）
- 每个有对应 mask 的 segment 的所有 token 都被标记为相同的 mask 索引

**用途**: 训练时确定哪些 token 需要预测对应的 mask

**示例**:
```python
# 假设 input_ids 长度为 608
# mask_ids[0:32] = [-1, -1, ..., -1]  # prompt 部分
# mask_ids[32:100] = [0, 0, ..., 0]    # 第一个 segment 的 tokens
# mask_ids[100:200] = [1, 1, ..., 1]   # 第二个 segment 的 tokens
```

---

### 3. `pixel_values` - 预处理后的图像张量

**形状**: `torch.Size([3, 384, 384])`, `dtype: torch.float32`

**生成过程**（`png.py:163-168`）:
```python
image = self.read_image(image_info['file_name'])  # 原始 PIL Image
image_data = self.image_processor.preprocess(image)  # DeepSeekVL 的图像处理器
pixel_values = image_data['pixel_values'][0]  # 提取处理后的图像
pixel_values = torch.from_numpy(pixel_values)  # 转为 tensor
```

**含义**:
- 原始图像被处理为固定尺寸: **384×384**
- 已经归一化（通常在 [0, 1] 或标准化后的范围）
- 通道顺序: RGB（3 通道）
- 用于输入视觉编码器（Vision Transformer）

**预处理步骤**:
1. 图像 resize 到固定尺寸
2. 归一化处理
3. 转换为 tensor 格式

---

### 4. `masks` - 调整到图像处理尺寸的 mask

**形状**: `torch.Size([2, 255, 384])`, `dtype: torch.uint8`

**生成过程**（`png.py:150-157, 171, 175`）:
```python
# 1. 从 panoptic 分割图加载原始 mask
segm_map = self._load_segm(os.path.join(self.panoptic_png_path, segm_file))

# 2. 根据 segment_ids 提取 mask
masks = []
for mask_segment_ids_ in mask_segment_ids:
    mask = 0
    for segment_id in mask_segment_ids_:
        mask += (segm_map == int(segment_id)).astype(np.uint8)
    masks.append(np.clip(mask, a_max=1, a_min=0))

masks = torch.from_numpy(np.stack(masks))

# 3. 调整到图像处理后的尺寸（255×384）
h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
masks = F.interpolate(masks[None], size=(h, w))[0]
```

**含义**:
- `2`: 该样本有 2 个 mask（对应 2 个 segment）
- `255×384`: 与 `meta_data['image_shape']` 一致，是图像预处理后的实际尺寸
- `uint8`: 0/1 二值 mask（0=背景，1=前景）

**注意**: 这里的尺寸是图像处理后的尺寸，不是原始图像尺寸，也不是模型输入尺寸。

---

### 5. `padded_masks` - 填充到模型输入尺寸的 mask

**形状**: `torch.Size([2, 384, 384])`, `dtype: torch.uint8`

**生成过程**（`png.py:177-183`）:
```python
p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']  # 384×384
padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
padding = meta_data['padding']

# 将 masks 放到正确位置（考虑 padding）
padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                padding['before_width']:p_w-padding['after_width']] = masks
```

**含义**:
- 模型输入需要固定尺寸（384×384），但图像预处理后的实际尺寸是 255×384
- 在高度方向添加 padding（上下各 padding），得到 384×384
- 训练时使用 `padded_masks` 与模型输出对齐（模型输出也是 384×384）

**Padding 信息**:
```python
padding = {
    'before_height': 64,   # 上方 padding
    'after_height': 65,    # 下方 padding
    'before_width': 0,     # 左侧 padding
    'after_width': 0       # 右侧 padding
}
```

**示例**:
- 原始图像: 640×426
- 预处理后: 384×255（保持宽高比）
- Padding 后: 384×384（固定尺寸）

---

### 6. `gt_masks` - 原始图像尺寸的 ground truth mask

**形状**: `torch.Size([2, 426, 640])`, `dtype: torch.uint8`

**生成过程**（`png.py:174`）:
```python
gt_masks = masks.clone()  # 在 interpolate 之前克隆
```

**含义**:
- `426×640`: 原始图像尺寸（高度×宽度）
- 保留原始分辨率，用于评估时计算 IoU 等指标
- 训练时主要使用 `padded_masks`，评估时可能需要 resize 回原尺寸

**用途**:
- 训练时: 不使用（使用 `padded_masks`）
- 评估时: 计算真实 IoU、准确率等指标

---

### 7. `image_sizes` - 原始图像尺寸

**形状**: `torch.Size([2])`, `dtype: torch.int64`

**生成过程**（`png.py:199`）:
```python
image_sizes=torch.tensor(image_data['image_sizes'][0])
```

**含义**:
- `[height, width]`: 原始图像的尺寸
- 示例中为 `[426, 640]`（高度=426，宽度=640）

**用途**: 
- 在评估时将预测的 mask resize 回原始尺寸
- 计算真实的像素级指标

---

### 8. `mask_infos` - Mask 的元信息

**类型**: `list of length 2`

**生成过程**（`png.py:131-139`）:
```python
for segment in data_sample['segments']:
    if not segment['plural']:
        segment_id = int(segment['segment_ids'][0])
        isthing = self.coco.cats[annotations[segment_id]['category_id']]['isthing']
    else:
        isthing = 1
    
    mask_infos.append(dict(
        plural=segment['plural'],      # 是否为复数（多个对象）
        isthing=isthing > 0            # 是否为 thing（物体）还是 stuff（背景）
    ))
```

**含义**:
- `plural`: `True/False`，表示该 mask 是否对应多个 segment（复数形式）
- `isthing`: `True/False`，区分 thing（可数物体，如"人"、"车"）和 stuff（不可数区域，如"天空"、"草地"）

**示例**:
```python
mask_infos = [
    {'plural': False, 'isthing': True},   # 第一个 mask: 单数物体
    {'plural': True, 'isthing': True}    # 第二个 mask: 复数物体
]
```

---

### 9. `image` - 原始 PIL 图像对象

**类型**: `PIL.JpegImagePlugin.JpegImageFile`

**生成过程**（`png.py:163`）:
```python
image = self.read_image(image_info['file_name'])
```

**含义**:
- 原始未处理的 PIL Image 对象
- 可用于可视化、调试或保存原始图像

**示例**:
```python
image = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x426>
```

---

### 10. `file_name` - 图像文件名

**类型**: `str`

**示例**: `"000000226461.jpg"`

**含义**: COCO 数据集的图像文件名，用于标识和调试

---

### 11. `meta_data` - 图像处理元数据

**类型**: `dict`

**内容**:
```python
{
    'padding': {
        'before_height': 64,   # 上方 padding 像素数
        'after_height': 65,    # 下方 padding 像素数
        'before_width': 0,     # 左侧 padding 像素数
        'after_width': 0       # 右侧 padding 像素数
    },
    'image_shape': {
        'height': 255,         # 图像预处理后的实际高度
        'width': 384           # 图像预处理后的实际宽度
    },
    'padded_shape': {
        'height': 384,         # Padding 后的高度（模型输入尺寸）
        'width': 384          # Padding 后的宽度（模型输入尺寸）
    }
}
```

**含义**:
- `image_shape`: 图像预处理后的实际尺寸（255×384）
- `padded_shape`: 填充后的尺寸（384×384），用于模型输入
- `padding`: 各方向的 padding 像素数，用于将模型输出从 `padded_shape` 映射回 `image_shape`

**用途**: 
- 训练时: 确定 mask 的放置位置
- 评估时: 将预测结果从 padded 尺寸映射回原始尺寸

---

### 12. `labels` - 训练标签（用于语言模型）

**形状**: `torch.Size([608])`, `dtype: torch.int64`

**生成过程**（`png.py:186-188`）:
```python
prompt_len = len(self.prompt)
labels = torch.ones_like(input_ids) * IGNORE_INDEX  # 全部设为忽略
labels[prompt_len:] = input_ids[prompt_len:]  # prompt 之后的部分作为标签
```

**含义**:
- `IGNORE_INDEX`（通常为 -100）: 该位置不计算损失
- Prompt 部分（前 `prompt_len` 个 token）: 设为 `IGNORE_INDEX`（不预测 prompt）
- Caption 部分（prompt 之后）: 设为对应的 token ID（预测 caption）

**用途**: 
- 语言建模损失（如果需要）: 只对 caption 部分计算损失
- Mask 预测: 主要使用 `mask_ids` 和 `padded_masks`

**示例**:
```python
# 假设 prompt_len = 32
labels[0:32] = [-100, -100, ..., -100]  # prompt 部分，忽略
labels[32:] = input_ids[32:]             # caption 部分，用于预测
```

---

## 数据流程总结

### 完整的数据加载流程

```
原始 PNG 数据样本
  ↓
1. 读取 PNG JSON 数据（包含 image_id 和 segments）
  ↓
2. Tokenize prompt + captions → input_ids
  ↓
3. 构建 mask_ids（标记每个 token 属于哪个 mask）
  ↓
4. 加载原始图像（PIL Image）
  ↓
5. 图像预处理（resize、归一化）→ pixel_values (384×384)
  ↓
6. 从 panoptic 分割图提取 mask → 原始尺寸 mask
  ↓
7. 调整 mask 到图像处理尺寸 → masks (255×384)
  ↓
8. Padding mask → padded_masks (384×384)
  ↓
9. 保存原始尺寸 → gt_masks (426×640)
  ↓
10. 构建 labels（用于语言建模）
  ↓
输出 data_sample 字典（12 个字段）
```

### 尺寸变化流程

```
原始图像: 640×426
  ↓ [图像预处理]
处理尺寸: 384×255 (保持宽高比)
  ↓ [Padding]
模型输入: 384×384 (固定尺寸)
  ↓ [模型处理]
模型输出: 384×384
  ↓ [移除 Padding]
预测结果: 384×255
  ↓ [Resize 回原尺寸]
最终评估: 640×426
```

---

## 训练时的使用

### 模型输入

```python
# 主要输入
input_ids        # Token 序列，形状: [seq_len]
pixel_values     # 图像张量，形状: [3, 384, 384]
mask_ids         # Mask 索引，形状: [seq_len]
```

### 训练目标

```python
# Mask 预测目标
padded_masks     # Ground truth mask，形状: [num_masks, 384, 384]

# 语言建模目标（可选）
labels           # Token 标签，形状: [seq_len]
```

### 损失计算

1. **Mask 预测损失**:
   - 模型输出: `pred_masks` (形状: [num_masks, 384, 384])
   - Ground truth: `padded_masks` (形状: [num_masks, 384, 384])
   - 损失函数: Dice Loss + CrossEntropy Loss

2. **语言建模损失**（如果需要）:
   - 模型输出: `logits` (形状: [seq_len, vocab_size])
   - Ground truth: `labels` (形状: [seq_len])
   - 只对 `labels != IGNORE_INDEX` 的位置计算损失

### 评估时

```python
# 评估指标计算
gt_masks         # 原始尺寸的 ground truth，形状: [num_masks, H, W]
image_sizes      # 原始图像尺寸 [height, width]
meta_data        # 用于将预测结果映射回原始尺寸
```

---

## 相关文件

- **数据集实现**: `flmm/datasets/png.py`
- **数据转换**: `flmm/datasets/transforms.py`
- **配置文件**: `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`

---

## 注意事项

1. **数据集大小**: 训练集包含 190,157 个样本，如果 mask_cnt=0（没有有效 mask），会随机选择一个新样本。

2. **Mask 数量**: 每个样本可能有不同数量的 mask（segments），`padded_masks` 的第一维是 `mask_cnt`。

3. **尺寸对齐**: 
   - 训练时使用 `padded_masks`（384×384）与模型输出对齐
   - 评估时使用 `gt_masks`（原始尺寸）计算真实指标

4. **Token 与 Mask 的对应**: `mask_ids` 建立了 `input_ids` 中每个 token 与 mask 的对应关系，只有 `mask_ids != -1` 的 token 需要预测 mask。

5. **Padding 处理**: 
   - 图像预处理后尺寸可能不是 384×384（保持宽高比）
   - 通过 padding 统一到 384×384
   - `meta_data['padding']` 记录了 padding 的位置和大小

---

## 参考

- PNG 数据集: [BCV-Uniandes/PNG](https://github.com/BCV-Uniandes/PNG)
- COCO Panoptic: [COCO Dataset](https://cocodataset.org/)
- F-LMM 论文: [F-LMM: Grounding Frozen Large Multimodal Models](https://arxiv.org/abs/2406.05821)

