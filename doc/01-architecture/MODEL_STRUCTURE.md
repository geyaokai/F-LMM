# F-LMM 模型结构详解

本文档详细解释 F-LMM 模型的架构、实现细节和不同模型之间的对比。

## 目录

- [模型架构概述](#模型架构概述)
- [1. 如何冻结基础 LMM 参数](#1-如何冻结基础-lmm-参数)
- [2. 如何提取 Hidden States 和 Attentions](#2-如何提取-hidden-states-和-attentions)
- [3. 注意力如何映射到图像空间](#3-注意力如何映射到图像空间)
- [4. U-Net 和 SAM 的集成](#4-u-net-和-sam-的集成)
- [5. 损失计算方式](#5-损失计算方式)
- [图像分辨率与 Token 数量处理](#图像分辨率与-token-数量处理)
- [不同模型的对比](#不同模型的对比)

---

## 模型架构概述

F-LMM 的核心思想是**冻结大型多模态模型（LMM）的参数**，只训练一个轻量级的 mask head（U-Net）来执行 grounding 任务。

### 整体架构

```
输入 (图像 + 文本)
  ↓
[冻结的 LMM] (DeepSeekVL/LLaVA/MGM 等)
  ├─ 视觉编码器 (冻结)
  ├─ 语言模型 (冻结)
  └─ 输出: hidden_states + attentions
  ↓
[注意力提取与处理]
  ├─ 提取图像 token 的注意力权重
  ├─ 重塑为图像空间维度 (H×W)
  └─ 加权融合多层注意力
  ↓
[U-Net Mask Head] (可训练)
  └─ 输出: 粗粒度 mask
  ↓
[SAM Refiner] (可选，冻结图像编码器)
  └─ 输出: 精细 mask
  ↓
损失计算 (Dice Loss + CrossEntropy Loss)
```

---

## 1. 如何冻结基础 LMM 参数

### 1.1 冻结方法

所有 Frozen 模型都使用相同的方式冻结基础 LMM：

#### DeepSeekVL (`frozen_deepseek_vl.py:24`)
```python
def __init__(self, model, ...):
    with LoadWoInit():  # 不初始化权重（从预训练加载）
        self.deepseek_vl = BUILDER.build(model)
    self.deepseek_vl.requires_grad_(False)  # 冻结所有参数
```

#### LLaVA (`frozen_llava.py:22`)
```python
def __init__(self, model, ...):
    self.llava = BUILDER.build(model)
    self.llava.requires_grad_(False)  # 冻结所有参数
```

#### MGM (`frozen_mgm.py:107`)
```python
def _init_mgm_model(self, model):
    # ... 加载模型 ...
    self.mgm.requires_grad_(False)  # 冻结所有参数
```

### 1.2 训练模式控制

在训练时，确保 LMM 始终处于 eval 模式：

```python
def train(self, mode=True):
    super().train(mode=mode)
    self.deepseek_vl.train(mode=False)  # LMM 始终 eval 模式
    self.training = mode
    return self
```

**关键点**:
- `requires_grad_(False)`: 禁用梯度计算，节省显存和计算
- `train(mode=False)`: 禁用 BatchNorm/Dropout 等训练时的行为
- 只有 `mask_head`、`text_proj`、`text_layer_weights` 是可训练的

---

## 2. 如何提取 Hidden States 和 Attentions

### 2.1 前向传播获取输出

#### DeepSeekVL (`frozen_deepseek_vl.py:112-118`)
```python
with torch.no_grad():  # 不需要梯度
    outputs = self.deepseek_vl.language_model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,  # 输出 hidden states
        output_attentions=True,      # 输出 attention weights
        return_dict=True,
        use_cache=False)
```

#### LLaVA (`frozen_llava.py:110-114`)
```python
with torch.no_grad():
    outputs = self.llava(**inputs,
                         attention_mask=attention_mask,
                         output_hidden_states=True,
                         output_attentions=True)
```

#### MGM (`frozen_mgm.py:224-232`)
```python
with torch.no_grad():
    outputs = self.mgm(input_ids=input_ids,
                       mask_ids=mask_ids,
                       images=image_tensor,
                       images_aux=image_tensor_aux,
                       output_hidden_states=True,
                       output_attentions=True,
                       return_dict=True,
                       use_cache=False)
```

### 2.2 提取 Hidden States

**目标**: 提取文本 token 的 hidden states，用于生成文本嵌入（text embeddings）

#### DeepSeekVL (`frozen_deepseek_vl.py:124-126`)
```python
# 获取所有层的 hidden states
hidden_states = outputs.hidden_states[-self.deepseek_vl.config.language_config.num_hidden_layers:]
# 堆叠: [num_layers, batch_size, seq_len, dim] -> [num_layers, seq_len, dim]
hidden_states = torch.stack([hs[0] for hs in hidden_states])

# 使用可学习的层权重进行加权融合
text_layer_weights = self.get_text_layer_weights()  # Softmax 归一化
hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)
# 结果: [seq_len, dim]
```

**层权重的作用**:
- 不同层可能包含不同层次的信息（浅层：语法，深层：语义）
- 通过学习权重，模型可以自适应地选择最有用的层

#### LLaVA (`frozen_llava.py:118-123`)
```python
hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]
hidden_states = torch.stack([hs[0] for hs in hidden_states])
hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)
```

**相同模式**: 所有模型都使用相同的加权融合策略

### 2.3 提取 Attentions

**目标**: 提取文本 token 对图像 token 的注意力权重

#### DeepSeekVL (`frozen_deepseek_vl.py:122-123`)
```python
# 找到图像 token 的位置
images_seq_mask = input_ids == self.image_token_idx  # [seq_len]

# 提取注意力: 只保留对图像 token 的注意力
attentions = [attn[0, ..., images_seq_mask[0]] for attn in outputs.attentions]
# attn 形状: [batch, num_heads, seq_len, seq_len]
# 提取后: [num_heads, seq_len, num_image_tokens]
```

#### LLaVA (`frozen_llava.py:116-117`)
```python
# LLaVA 模型返回了 image_to_overwrite 索引
attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
              for attn in outputs.attentions]
```

#### MGM (`frozen_mgm.py:238-239`)
```python
# MGM 模型返回了 image_places 索引
attentions = [attn[0, ..., outputs.image_places[0]]
              for attn in outputs.attentions]
```

**关键差异**:
- **DeepSeekVL**: 手动查找图像 token 位置 (`images_seq_mask`)
- **LLaVA**: 使用模型返回的 `image_to_overwrite`
- **MGM**: 使用模型返回的 `image_places`

---

## 3. 注意力如何映射到图像空间

### 3.1 重塑注意力为空间维度

注意力权重原本是 `[num_heads, seq_len, num_image_tokens]`，需要重塑为 `[num_heads, seq_len, H, W]`。

#### DeepSeekVL (`frozen_deepseek_vl.py:123`)
```python
# clip_shape = 24 (DeepSeekVL 使用 24×24 的图像特征图)
self.clip_shape = 24
attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) 
              for attn in attentions]
# 结果: [num_heads, seq_len, 24, 24]
```

**原理**:
- DeepSeekVL 的图像被处理为 24×24 个 patch
- 每个 patch 对应一个图像 token
- 因此 `num_image_tokens = 24 × 24 = 576`

#### LLaVA (`frozen_llava.py:128-130`)
```python
# LLaVA 的图像特征图尺寸取决于图像大小和 patch_size
padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size
# 例如: 384 // 14 = 27 (CLIP ViT-L/14)

attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) 
              for attn in attentions]
```

#### MGM (`frozen_mgm.py:171-202`)
```python
def _process_attention(self, attention2image):
    num_heads, seq_len, image_len = attention2image.shape
    single_image_len = self.clip_shape[0] * self.clip_shape[1]
    image_grid = getattr(self.mgm.config, 'image_grid', 1)
    
    if image_grid == 1:
        # 单图像: 直接重塑
        return attention2image.view(num_heads, seq_len, *self.clip_shape)
    else:
        # 多图像网格: 需要处理全局图像和局部图像
        # ... 复杂的重塑逻辑 ...
        return attention2hd_image
```

**MGM 的特殊处理**:
- 支持图像网格（image_grid > 1），例如 2×2 网格
- 需要合并全局图像和局部图像的注意力

### 3.2 按 Mask 分组注意力

对于每个 mask，提取对应文本 token 的注意力：

#### DeepSeekVL (`frozen_deepseek_vl.py:131-143`)
```python
masks = data_sample['masks']
mask_attentions = []
text_embeds = []

for mask_id in range(len(masks)):
    # 找到属于当前 mask 的 token
    matched = mask_ids == mask_id
    assert matched.sum() > 0
    
    # 合并所有层的注意力 (逐层 merge，然后 concat)
    mask_attentions.append(
        torch.cat([
            self.apply_merge(attn[:, matched], dim=1)  # 合并 heads
            for attn in attentions  # 遍历所有层
        ])
    )
    # 结果: [num_layers * num_heads, H, W]
    
    # 提取对应的 hidden states
    text_embeds.append(self.text_proj(hidden_states[matched]))
```

**Merge 操作** (`frozen_deepseek_vl.py:40-46`):
```python
def apply_merge(self, x, dim=1):
    if self.merge == 'mean':
        return x.mean(dim=dim)  # 平均合并多个 head
    elif self.merge == 'max':
        return x.max(dim=dim).values  # 最大池化
```

**过程**:
1. 对于每个 layer: `[num_heads, num_matched_tokens, H, W]` → `[num_matched_tokens, H, W]` (merge heads)
2. 对于每个 token: 取平均值/最大值 → `[H, W]`
3. 拼接所有层: `[num_layers, H, W]` → `[num_layers * num_heads, H, W]`

**最终形状**: `[num_masks, num_layers * num_heads, H, W]`

---

## 4. U-Net 和 SAM 的集成

### 4.1 U-Net Mask Head

U-Net 负责将注意力图转换为粗粒度的 mask。

#### U-Net 结构 (`mask_decoder.py:20-59`)
```python
class UNetHead(UNet):
    def __init__(self, upsample_input=None, normalize_input=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = nn.Conv2d(self.base_channels, 1, kernel_size=1)
        # 输出通道为 1（二值 mask）
        
    def forward(self, x):
        # x: [batch, num_layers * num_heads, H, W]
        
        # 1. 归一化输入（可选）
        if self.normalize_input:
            x_sum = x.sum((-2, -1), keepdims=True).clamp(min=1e-12)
            x = x / x_sum
        
        # 2. 上采样输入（可选）
        if self.upsample_input is not None:
            scale_factor = max(1.0, self.upsample_input / max(h, w))
            x = F.interpolate(x.float(), scale_factor=scale_factor, mode='bilinear')
        
        # 3. Padding 到可整除的尺寸
        padded_x = x.new_zeros(*x.shape[:2], padded_h, padded_w)
        padded_x[..., :h, :w] = x
        
        # 4. U-Net 前向传播
        x = super().forward(padded_x)[-1][..., :h, :w]
        
        # 5. 输出 mask logits
        return self.conv_seg(x)  # [batch, 1, H', W']
```

**U-Net 配置** (从配置文件):
```python
unet = dict(
    type=UNetHead,
    normalize_input=True,      # 归一化注意力图
    upsample_input=64,         # 上采样到 64×64
    in_channels=2048,          # num_layers * num_heads
    base_channels=64,
    num_stages=4,
    # ... 其他配置
)
```

**输入输出**:
- 输入: `[num_masks, num_layers * num_heads, H, W]` (例如 `[2, 2048, 24, 24]`)
- 输出: `[num_masks, 1, H', W']` (例如 `[2, 1, 64, 64]`)

### 4.2 SAM Refiner

SAM 用于细化 U-Net 输出的粗粒度 mask。

#### SAM 包装器 (`mask_refiner.py:24-125`)
```python
class SAMWrapper(nn.Module):
    def __init__(self, model_name, checkpoint, use_text=True, use_mask=True, use_box=True):
        super().__init__()
        self.model = sam_model_registry[model_name](checkpoint=checkpoint)
        self.model.image_encoder.requires_grad_(False)  # 冻结图像编码器
        self.text_proj = ...  # 文本嵌入投影层
        
    def forward(self, image, pred_masks, text_embeds):
        # 1. 编码图像
        image_embedding, original_image_size, input_size = self.encode_image(image)
        
        # 2. 生成 prompt mask (从 U-Net 输出)
        prompt_masks = self.generate_prompt_masks(pred_masks, input_size)
        
        # 3. 处理每个 mask
        for prompt_mask, pred_mask, text_embed in zip(prompt_masks, pred_masks, text_embeds):
            # 3.1 生成 bounding box
            if pred_mask.sum() > 0:
                box = mask2box(pred_mask.float().cpu().numpy())
            else:
                box = [0.0, 0.0, w, h]
            
            # 3.2 编码 prompt (box + mask + text)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=prompt_mask if self.use_mask else None,
            )
            
            # 3.3 添加文本嵌入
            if self.use_text:
                sparse_embeddings = torch.cat([
                    sparse_embeddings,
                    text_embed[None]  # 文本嵌入作为 prompt
                ], dim=1)
            
            # 3.4 SAM 解码器生成精细 mask
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            
            # 3.5 后处理到原始尺寸
            sam_mask = self.model.postprocess_masks(
                low_res_masks, input_size, original_image_size)
        
        return torch.stack(sam_masks)
```

**SAM 的输入**:
1. **图像嵌入**: SAM 图像编码器的输出
2. **Box prompt**: 从 U-Net mask 提取的 bounding box
3. **Mask prompt**: U-Net 输出的粗粒度 mask（resize 到 256×256）
4. **Text prompt**: 文本嵌入（通过 `text_proj` 投影）

**SAM 的输出**: 精细的 mask，分辨率更高（通常是原始图像尺寸）

### 4.3 集成流程

```python
# 1. U-Net 生成粗粒度 mask
pred_masks = self.mask_head(mask_attentions)[:, 0]  # [num_masks, H, W]

# 2. 移除 padding（如果使用）
pred_masks = pred_masks[:, before_height:before_height + mask_h, 
                        before_width:before_width + mask_w]

# 3. SAM 细化
sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)
# 结果: [num_masks, H_orig, W_orig]
```

---

## 5. 损失计算方式

### 5.1 损失函数

F-LMM 使用两种损失函数：

#### Dice Loss (`frozen_deepseek_vl.py:69-71`)
```python
loss_dice = self.loss_dice(
    pred_masks.view(mask_cnt, -1), 
    gt_masks.view(mask_cnt, -1),
    avg_factor=mask_cnt)
```

**Dice Loss 公式**:
```
Dice Loss = 1 - (2 * intersection + eps) / (union + eps)
```

**特点**:
- 对类别不平衡友好（mask 通常只占图像的一小部分）
- 关注重叠区域

#### CrossEntropy Loss (`frozen_deepseek_vl.py:72-75`)
```python
loss_mask = self.loss_mask(
    pred_masks.view(-1),
    gt_masks.view(-1),
    avg_factor=pred_masks.numel())
```

**CrossEntropy Loss**:
- 使用 sigmoid + BCE Loss
- 对每个像素独立计算损失

### 5.2 总损失

#### 训练时 (`frozen_deepseek_vl.py:175-225`)
```python
def compute_loss(self, data):
    loss_dice = 0
    loss_mask = 0
    sam_loss_dice = 0
    sam_loss_mask = 0
    
    for data_sample in data:
        forward_output = self._forward(data_sample)
        pred_masks = forward_output['pred_masks']      # U-Net 输出
        sam_pred_masks = forward_output['sam_pred_masks']  # SAM 输出
        
        # U-Net 损失
        loss_dice_, loss_mask_, ... = self._compute(pred_masks, gt_masks)
        loss_dice += loss_dice_ * mask_cnt
        loss_mask += loss_mask_ * mask_cnt
        
        # SAM 损失
        sam_loss_dice_, sam_loss_mask_, ... = self._compute(sam_pred_masks, sam_gt_masks)
        sam_loss_dice += sam_loss_dice_ * mask_cnt
        sam_loss_mask += sam_loss_mask_ * mask_cnt
    
    # 平均损失
    loss_dict = {
        'loss_mask': loss_mask / mask_cnts,
        'loss_dice': loss_dice / mask_cnts,
        'sam_loss_mask': sam_loss_mask / mask_cnts,
        'sam_loss_dice': sam_loss_dice / mask_cnts,
        # ... 其他指标
    }
    return loss_dict
```

**总损失**:
```
Total Loss = loss_mask + loss_dice + sam_loss_mask + sam_loss_dice
```

**注意**: 
- U-Net 和 SAM 的损失都需要计算
- 最终使用的 mask 是 SAM 的输出（更精细）
- U-Net 损失起到监督作用，帮助 U-Net 学习

---

## 图像分辨率与 Token 数量处理

### 核心问题

在使用 `<image_placeholder>` token 时，存在一个关键问题：**如何确定图像 token 的数量？**

### DeepSeekVL: 固定 Token 数量

**设计**:
- 固定使用 **576 个** `<image_placeholder>` token
- 对应固定的 **24×24 = 576** 个图像 patch

**实现**:
```python
# 配置文件中硬编码
prompt = '<image_placeholder>'*576

# 模型代码中硬编码
self.clip_shape = 24  # 24×24 = 576 个 patch（固定）
```

**图像预处理**:
```python
# VLMImageProcessor 将所有图像处理成固定尺寸
def resize(self, pil_img: Image):
    # 1. 保持宽高比，resize 到 image_size (通常是 384)
    size = [int(height / max_size * self.image_size), 
            int(width / max_size * self.image_size)]
    
    # 2. Pad 成正方形 [image_size, image_size]
    pil_img, meta = expand2square(pil_img, self.background_color)
    
    # 3. Vision Model 输出固定的 24×24 = 576 个 patch
    return [3, 384, 384]  # 固定尺寸
```

**特点**:
- ✅ **优势**: 实现简单，训练稳定，batch 内序列长度一致
- ❌ **限制**: 不支持动态分辨率（固定 576 个 token）
- ✅ **实际**: 训练时所有图像都被预处理成相同尺寸（384×384），因此可以正常工作

**设计权衡**:
- 虽然不能处理任意分辨率，但训练数据都是统一尺寸
- 固定 token 数量简化了注意力重塑和训练流程

### LLaVA: 动态 Token 数量

**设计**:
- 根据实际图像大小**动态计算**图像 token 数量
- Token 数量 = `(padded_h // patch_size) × (padded_w // patch_size)`

**实现**:
```python
# 动态计算特征图尺寸
padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
llava_h = padded_h // self.patch_size  # 例如: 384 // 14 = 27
llava_w = padded_w // self.patch_size
num_image_tokens = llava_h * llava_w  # 动态计算

# 注意力重塑
attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) 
              for attn in attentions]
```

**特点**:
- ✅ **优势**: 支持不同分辨率，更灵活
- ❌ **限制**: 实现复杂，batch 内序列长度可能不一致

### MGM: 支持图像网格和高分辨率

**设计**:
- 支持图像网格（image_grid > 1），例如 2×2 网格
- 可以处理高分辨率图像（通过图像网格分割）

**实现**:
```python
def _process_attention(self, attention2image):
    image_grid = getattr(self.mgm.config, 'image_grid', 1)
    
    if image_grid == 1:
        # 单图像: 直接重塑
        return attention2image.view(num_heads, seq_len, *self.clip_shape)
    else:
        # 多图像网格: 处理全局图像 + 局部图像网格
        # ... 复杂的重塑逻辑 ...
        return attention2hd_image
```

**特点**:
- ✅ **优势**: 支持高分辨率，通过图像网格处理大图像
- ❌ **限制**: 实现最复杂

### 总结对比

| 模型 | Token 数量 | 分辨率支持 | 实现复杂度 | 训练稳定性 |
|------|-----------|-----------|-----------|-----------|
| **DeepSeekVL** | 固定 576 | 固定 384×384 | 简单 | 高（统一序列长度） |
| **LLaVA** | 动态计算 | 可变（预处理后） | 中等 | 中等（需处理不同长度） |
| **MGM** | 动态计算 | 可变（支持网格） | 复杂 | 中等 |

### 为什么 DeepSeekVL 选择固定 Token？

1. **简化实现**: 固定 token 数量使注意力重塑简单直接
2. **训练稳定性**: Batch 内序列长度一致，训练更稳定
3. **性能优化**: 不需要动态分配内存，性能更好
4. **实际需求**: 训练数据已统一尺寸，固定 token 满足需求

### 如果真需要动态分辨率怎么办？

如果需要处理不同分辨率，可以考虑：

**方案 1**: 动态计算 token 数量
```python
num_image_tokens = vision_model.get_num_patches(image_size)
prompt = '<image_placeholder>' * num_image_tokens
```

**方案 2**: 使用 LLaVA 或 MGM 的实现方式
- LLaVA: 动态计算特征图尺寸
- MGM: 支持图像网格和高分辨率

**方案 3**: 保持固定 token，但用 padding
- 小图像用零填充（会浪费计算资源）

---

## 不同模型的对比

### 模型架构对比

| 特性 | DeepSeekVL | LLaVA | MGM |
|------|-----------|-------|-----|
| **图像特征尺寸** | 24×24 (固定) | 27×27 (384×384, patch_size=14) | 可变 (取决于配置) |
| **图像 token 数量** | 576 | ~729 | 可变 |
| **注意力提取** | `images_seq_mask` | `image_to_overwrite` | `image_places` |
| **图像处理** | `prepare_inputs_embeds` | 直接输入 | `_process_image` (支持网格) |
| **特殊处理** | 图像占位符 × 576 | 支持多分辨率 | 支持图像网格 |

### 注意力映射对比

#### DeepSeekVL
```python
# 固定 24×24 特征图
attentions = [attn.view(*attn.shape[:-1], 24, 24) for attn in attentions]
```

#### LLaVA
```python
# 动态计算特征图尺寸
llava_h = padded_h // self.patch_size  # 例如: 384 // 14 = 27
llava_w = padded_w // self.patch_size
attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
```

#### MGM
```python
# 支持图像网格，需要特殊处理
def _process_attention(self, attention2image):
    # 处理全局图像 + 局部图像网格
    # ... 复杂的重塑逻辑 ...
```

### Hidden States 提取对比

**相同点**:
- 所有模型都使用相同的加权融合策略
- 都使用可学习的 `text_layer_weights`
- 都提取最后 N 层的 hidden states

**不同点**:
- **DeepSeekVL**: 从 `language_model` 输出提取
- **LLaVA**: 从模型直接输出提取
- **MGM**: 从模型输出提取，需要额外的图像处理

### 文本嵌入投影对比

所有模型都使用类似的投影层：

```python
# DeepSeekVL
self.text_proj = nn.Linear(
    self.deepseek_vl.config.language_config.hidden_size,
    self.sam.model.prompt_encoder.embed_dim)

# LLaVA
self.text_proj = nn.Linear(
    self.llava.config.text_config.hidden_size,
    self.sam.model.prompt_encoder.embed_dim)

# MGM
self.text_proj = nn.Linear(
    self.mgm.config.hidden_size,
    self.sam.model.prompt_encoder.embed_dim)
```

**目的**: 将文本 hidden states 投影到 SAM prompt encoder 的嵌入空间

---

## 关键设计模式

### 1. 冻结 + 提取模式

```
冻结 LMM → 提取中间表示 → 轻量级 head → 输出
```

**优势**:
- 保留 LMM 的视觉-语言理解能力
- 只需训练少量参数（U-Net + 投影层）
- 训练效率高，显存占用小

### 2. 多层注意力融合

```
Layer 1 Attention → Merge Heads → Concat
Layer 2 Attention → Merge Heads → Concat
...
Layer N Attention → Merge Heads → Concat
  ↓
[num_layers * num_heads, H, W] → U-Net
```

**优势**:
- 利用不同层的信息（浅层：细节，深层：语义）
- 提供丰富的特征表示

### 3. 两阶段细化

```
注意力图 → U-Net (粗粒度) → SAM (精细)
```

**优势**:
- U-Net: 快速生成初始 mask
- SAM: 利用强大的分割能力细化边界

---

## 总结

F-LMM 的核心创新在于：

1. **冻结策略**: 完全冻结 LMM，保留其理解能力
2. **注意力利用**: 将文本对图像的注意力作为空间定位信号
3. **轻量级设计**: 只训练 U-Net 和少量投影层
4. **两阶段细化**: U-Net + SAM 的组合

这种设计使得 F-LMM 能够在保持 LMM 能力的同时，高效地学习 grounding 任务。

---

## 相关文件

- **DeepSeekVL**: `flmm/models/frozen_deepseek_vl.py`
- **LLaVA**: `flmm/models/frozen_llava.py`
- **MGM**: `flmm/models/frozen_mgm.py`
- **U-Net**: `flmm/models/mask_head/mask_decoder.py`
- **SAM**: `flmm/models/mask_head/mask_refiner.py`

---

## 参考

- F-LMM 论文: [F-LMM: Grounding Frozen Large Multimodal Models](https://arxiv.org/abs/2406.05821)
- SAM: [Segment Anything](https://segment-anything.com/)
- U-Net: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

