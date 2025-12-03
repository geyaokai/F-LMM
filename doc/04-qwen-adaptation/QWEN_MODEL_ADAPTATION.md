# Qwen2.5-VL 适配总览

> 本文整合原 `QWEN_MODEL_ADAPTATION.md` 与 `QWEN_VS_DEEPSEEK_LLAVA.md`，聚焦在关键张量维度、数据流、以及与其他底座的差异。阅读下文即可掌握在 F-LMM 中驱动 `FrozenQwenSAM` 的最小要点。

## 1. 目标速览
- **兼容统一 Processor**：通过 `QwenImageProcessorWrapper`（`flmm/datasets/qwen_image_processor.py`）让 Qwen 的 messages API 输出版与 RefCOCO2PNG 管线对齐。
- **明确动态分辨率逻辑**：`image_grid_thw=[grid_t, grid_h, grid_w]` 与 `merge_size` 一起定义了注意力要 reshape 的空间维度。
- **串联粗到细的掩码生产链**：`pixel_values/input_ids → Qwen → mask_head → SAMWrapper`，其中 `text_embeds` 已包含视觉上下文。
- **理解关键张量/形状**：下文提供的速查表与数据流图是排障的第一参考。

## 2. 数据流与关键张量

### 2.1 数据流图
```text
image + text (RefCOCO/PNG)
        │  (RefCOCO2PNG + QwenImageProcessorWrapper)
        ├─ input_ids [seq_len]
        ├─ mask_ids [seq_len]
        ├─ pixel_values [grid_t, vit_dim]
        ├─ image_grid_thw [1,3]  (grid_t, grid_h, grid_w)
        ├─ meta_data (padding / scale info)
        └─ masks / gt_masks
                │
                ▼
       FrozenQwenSAM._forward (flmm/models/frozen_qwen.py)
        1.  qwen_model(input_ids, pixel_values, image_grid_thw)
            ├─ hidden_states [num_layers, seq_len, hidden_size]
            └─ attentions [num_layers, num_heads, seq_len, seq_len]
        2.  Reshape attentions → [num_layers, num_heads, seq_len, qwen_h, qwen_w]
        3.  按 mask_ids 聚合 → mask_attentions[num_masks, num_layers*num_heads, qwen_h, qwen_w]
        4.  text_proj(hidden_states per mask) → text_embeds[num_tokens_i, sam_dim]
        5.  mask_head(mask_attentions) → pred_masks[num_masks, H_pad, W_pad]
        6.  去 padding → pred_masks[num_masks, H_img, W_img]
        7.  SAMWrapper(image, pred_masks, text_embeds)
            └─ sam_pred_masks[num_masks, H_img, W_img]
```

### 2.2 关键张量速查表
| 张量 | 形状（单样本） | 产出位置 | 备注 |
| --- | --- | --- | --- |
| `input_ids` | `[seq_len]`，dataloader 后 `[B, seq_len]` | `RefCOCO2PNG.transform_concat` | 已含 `<|vision_start|> ... <|vision_end|>`；mask_ids 中 `-1` 表示非目标 token。 |
| `mask_ids` | `[seq_len]` | 同上 | 每个对象的 token 编号；用于聚合注意力与 hidden states。 |
| `pixel_values` | `[grid_t × grid_h × grid_w, channel × temporal_patch_size(2) × patch_size(14) × patch_size(14)]` | `QwenImageProcessorWrapper` | 每个 patch 展平为一行，图像场景 `grid_t=1`，总 token 数等于 `grid_h × grid_w//mergesize(2)**2`。 |
| `image_grid_thw` | `[1, 3]` (grid_t, grid_h, grid_w) | 同上 | `qwen_h = grid_h / merge_size`，`qwen_w = grid_w / merge_size`；若缺失则回退 `meta_data`。 |
| `images_seq_mask` | `[seq_len]` bool | `_prepare_inputs` | 标记 `<|vision_start|>` 与 `<|vision_end|>` 之间的 token，用于裁剪注意力的最后一维。 |
| `attentions` | `list[num_layers]`，每项形如 `[num_heads, seq_len, num_image_tokens]` | Qwen 输出 | `num_image_tokens = grid_t × grid_h × grid_w//4`，图像用例中 `grid_t=1`；先按 `images_seq_mask` 选出视觉 token，再 reshape 为 `[num_heads, seq_len, qwen_h, qwen_w]`。 |
| `mask_attentions` | `[num_masks, num_layers*num_heads, qwen_h, qwen_w]` | `FrozenQwenSAM._forward` | 对属于同一 mask 的 token 做 `apply_merge`（mean/max），再按层 concat 成 U-Net 输入通道。 |
| `hidden_states` | `[seq_len, hidden_size]` | Qwen 输出 | 取最后 `num_layers` 层，经 `text_layer_weights` 加权；维度与语言模型隐藏维一致。 |
| `text_embeds` | `[tokens_i, sam_prompt_dim]` | `text_proj(hidden_states[matched])` | tokens_i 为该 mask 的文本 token 数；包含已融合视觉上下文的多模态表示。 |
| `pred_masks` | `[num_masks, H_pad, W_pad]` → `[num_masks, H_img, W_img]` | `mask_head` → padding 裁剪 | H/W 与 `meta_data[image_shape]` 对齐；在送入 SAM 前保持 logits。 |
| `prompt_masks` | `[num_masks, 1, 256, 256]` | `SAMWrapper.generate_prompt_masks` | 由 `pred_masks` 缩放/Pad 获取；用于 SAM prompt encoder。 |
| `sam_pred_masks` | `[num_masks, H_img, W_img]` | `SAMWrapper.forward` | 默认单通道；`multimask_output=True` 时取 IoU 最优候选。 |

## 3. 各阶段细节

### 3.1 数据集与 Processor
a. **QwenImageProcessorWrapper**（`flmm/datasets/qwen_image_processor.py`）
   - 使用 messages API 构造 `<|vision_start|><|image_pad|>...`，返回 `pixel_values`、`image_grid_thw`、`input_ids_with_vision`。
   - 将 `image_grid_thw=[t,h,w]` 写回 `result_dict`，供前向 reshape 注意力。
   - 自动 strip 用户 prompt 中的 `<image>` 占位符，避免重复插入。

b. **RefCOCO2PNG**（`flmm/datasets/transforms.py`）
   - 拼接模型 prompt + caption token，生成 `mask_ids`（`-1` 表示上下文）。
   - 将 wrapper 产出的 `pixel_values`（2D）转成 tensor，随 batch 堆叠。
   - 复制 `meta_data`（记录 padding、原图尺寸）用于 mask 对齐。

### 3.2 FrozenQwenSAM 前向
1. `_prepare_inputs` 读取 `input_ids`，定位 `<|vision_start|>`/`<|vision_end|>`，构造 `images_seq_mask`。
2. `qwen_model(**model_kwargs)` 输入：`input_ids`/`pixel_values`/`image_grid_thw` (+ 可选 `attention_mask`)；输出 hidden states 与 attentions。
3. `image_grid_thw` + `merge_size` 推出 `qwen_h = grid_h/merge_size`、`qwen_w = grid_w/merge_size`，用于将 `[... , grid_t]` reshape 成二维空间。
4. `mask_ids` 分桶：
   - 对每个 mask，`attn[:, matched]` 先在 token 维做 mean/max（`apply_merge`），再在层维 concat，得到 `[num_layers*num_heads, qwen_h, qwen_w]`。
   - `hidden_states[matched]` 经 `text_proj` → `text_embeds`，此时 embedding 已含视觉上下文。

### 3.3 Mask 生成 & SAM 细化
1. `mask_head(mask_attentions)` 输出 `[num_masks, 1, H_pad, W_pad]`，移除通道得 `pred_masks`（logits）。
2. `meta_data` 记录的 padding/scale 让 `pred_masks` 裁剪回原图尺度 (`mask_h × mask_w`)；同时保存 resized `mask_attentions` 做可视化或蒸馏。
3. `SAMWrapper`（`flmm/models/mask_head/mask_refiner.py`）
   - `pred_masks.sigmoid()` → resize → binarize 产生 box/mask prompt。
   - `text_embeds` 逐 mask 作为额外稀疏提示拼进 SAM prompt encoder。
   - `sam_model.mask_decoder` 输出的候选与粗 mask IoU 进行比对（若 `multimask_output=True`）。

## 4. 与 DeepSeek-VL / LLaVA 的差异（压缩版）
| 维度 | Qwen2.5-VL | DeepSeek-VL | LLaVA-Next |
| --- | --- | --- | --- |
| Processor | **统一式**（必须 text+image，一次性返回 vision tokens） | SigLIP 图像编码，分离式 | CLIP + projector，分离式 |
| 视觉 token 数 | 动态 (`grid_h × grid_w`)，需 `image_grid_thw` + `merge_size` | 固定 24×24 patch | coarse/fine 双尺度，由模型返回 shapes |
| pixel_values | `[grid_t, vit_dim]` | `[1, 3, H, W]` 经卷积编码 | `[2, 3, H, W]`（coarse/fine） |
| 注意力 reshape | 依赖 `grid_h/merge_size` 推出 `(qwen_h, qwen_w)` | 恒定 24×24 | 根据模型自带 `image_feature_shapes` |
| 文本占位符 | `<|vision_start|>`/`<|image_pad|>`/`<|vision_end|>` | `<image_placeholder>` 铺满 | `<image>` 单 token |
| Fallback | Wrapper 自动回退到 image_processor 并推断 grid | 无特殊处理 | 依赖 HuggingFace 实现 |

## 5. 排障与验证建议
- **核查 `image_grid_thw`：** 日志若出现 `Image token count …`，说明 grid 推断不一致。先确认数据集中是否正确写入 `[grid_t, grid_h, grid_w]`。
- **检查 `mask_ids` 对齐：** 若某个对象没有匹配 token，会在 `_forward` 中触发断言 `Mask {id} has no corresponding tokens`。
- **确认注意力可用：** 当 transformers 启用 SDPA 且未返回 `attentions` 时，需要在配置里强制 `output_attentions=True` 并禁用部分优化。
- **验证 `text_embeds`：** 由于来自融合后的 hidden states，常用方法是抽查 `text_embeds.norm()` 或可视化与 SAM 结果的相关性，确保 projector 训练稳定。
- **可视化链路：** `mask_attentions`（裁剪后尺寸）与 `sam_pred_masks` 可直接存盘比对粗细化效果，定位问题更快。

## 6. 参考文件
- `flmm/datasets/qwen_image_processor.py`：统一 Processor wrapper，返回图像相关张量。
- `flmm/datasets/transforms.py`：RefCOCO2PNG 入口，将 wrapper 结果注入数据样本。
- `flmm/models/frozen_qwen.py`：`FrozenQwenSAM` 核心逻辑，含注意力 reshape、mask 汇聚、`text_proj`。
- `flmm/models/mask_head/mask_refiner.py`：SAMWrapper，负责 box/mask prompt 生成与最终 refinement。
- `F-LMM/tests/debug_frozen_qwen_forward.py`：用于验证输入/输出形状的调试脚本，可在新样本上打印上述张量。

> 如需进一步细节（安装、版本、测试脚本等），请结合主仓 README 及 `configs/qwen` 下的示例配置。
