# Qwen Visual CoT：Re-sampling 方案设计稿

> 面向实现前的工程说明，帮助快速落地 “缓存 ViT token → ROI 重采样 → 再回答” 的 Visual Chain-of-Thought。只保留关键点，调试细节由你手动确认，避免为未知情况写冗余防呆代码。

---

## 1. 目标与约束

- **目标**：无需重新编码原图，仅利用 `FrozenQwenSAM.answer()` 时缓存的视觉 token，在选中 ROI 后构造新的上下文 token，让 Qwen 再次回答或描述 ROI。
- **约束**：
  - 仍然使用一张 GPU，禁止重新跑 `processor(image)`。
  - `answer()` 内部必须返回可复用的视觉特征。
  - 接口风格与现有 `answer/ground` 一致，CLI/脚本可以直接调用。

---

## 2. 需要新增的缓存

| 字段 | 来源 | 用途 | Debug 事项 |
|------|------|------|------------|
| `vision_tokens` | Qwen ViT 输出 `[grid_t × grid_h × grid_w, vision_hidden_size]` | ROI 重采样的原素材 | HuggingFace 默认输出里拿不到，需要在 `self.qwen_model.visual` 上挂 hook 把 `last_hidden_state` 抓出来。 |
| `vision_grid` (`grid_t`, `grid_h`, `grid_w`, `merge_size`) | 已有 `image_grid_thw` | 像素 → patch 索引映射 | 记住 `grid_t` 是时间维，图像恒为 1，不要再假设 `grid_h × grid_w == grid_t`。 |
| `pixel_offsets` | `meta_data` + padding | ROI → 真实像素坐标 | 现有逻辑可复用，注意 ROI 若超出图像需 clip。 |

实现建议：在 `FrozenQwenSAM.answer()` 中把 vision 编码器的输出 `vision_output.last_hidden_state` 存入 `answer_output['vision_tokens']`，shape `[grid_h*grid_w, vision_hidden_size]`。

---

## 3. ROI → patch 索引

1. `ground()` 已经算出 mask/bbox，得到 `bbox = (x0, y0, x1, y1)`。
2. 利用 `image_grid_thw`：
   - `patch_size = self.patch_size`
   - `qwen_h = grid_h // merge_size`
   - `qwen_w = grid_w // merge_size`
3. ROI 到 patch：  
   ```
   col_start = floor(x0 / (patch_size * merge_size))
   col_end   = ceil(x1 / (patch_size * merge_size))
   row_start = floor(y0 / (patch_size * merge_size))
   row_end   = ceil(y1 / (patch_size * merge_size))
   ```
4. clamp 到 `[0, qwen_w)` / `[0, qwen_h)`。

Debug：手动 print 若 ROI 太小导致 `row_end == row_start`，直接扩张 1 个 patch；无需写复杂 fallback。

---

## 4. 重采样逻辑

- **抽取 token**：在 `vision_tokens` 里按 `(row, col)` 转线性索引 `idx = row * qwen_w + col`，得到子集 `[num_roi_tokens, vision_hidden_size]`。
- **Projector**：
  - 引入 `self.resample_proj = nn.Linear(vision_hidden_size, language_hidden_size)`（可与 `text_proj` 共用或单独初始化）。
  - 输出 `roi_context = resample_proj(roi_tokens)`，作为额外 context token。
- **拼接方式**：
 1. 重新构造 prompt：`<|image_pad|>` 标记仍由 processor 负责，额外在文本末尾附加 “Here are ROI tokens” 类似提示。
 2. 将 `roi_context` 插入到 `input_embeds` 的合适位置（例如 `<vision_end>` 之后），记得同步扩展 `attention_mask`。
  3. 调用 `qwen_model` 的 `generate` 或 `forward`，并关闭重新编码。

注意：HuggingFace 的 Qwen 接口允许传 `inputs_embeds`，可参考 `DeepSeek.visual_cot_v1` 里的 `prepare_inputs_embeds` 用法。需要你手动测试 `resample_proj` 输出的 dtype/device 与语言模型一致。

---

## 5. API 设计

- `def visual_cot_resample(self, image, question, bbox, use_sam=True, max_new_tokens=64, extra_prompt=''):`
  1. 调用 `answer()` 得到 `vision_tokens`（若已有缓存可传入）。
  2. 根据 `bbox` 计算 patch index → 生成 `roi_context`。
  3. 构造新的 prompt 并生成回答。
  4. 返回 `{thought, roi_bbox, roi_context, answer_text}`。
- CLI：`cot resample <idx> <question>`：idx 取自最近 `ground` 的 mask/bbox。

---

## 6. 待你确认的细节（别写过度防御）

1. **Qwen 是否暴露 vision token**： `self.qwen_model.visual` 上挂 hook 把 `last_hidden_state` 抓出来
2. **Projector 初始化**：是否重用 `self.text_proj`？建议新建一层并复用 `nn.init.xavier_uniform_`，但具体数值你自行调试。
3. **Prompt 拼接**：`processor.apply_chat_template` 是否支持纯文本+自定义 context token？可以先用最朴素的方式：继续使用 `<image_placeholder>`，但在 `inputs_embeds` 中把 ROI context 追加到文本前，必要时手动调用 `self.qwen_model.prepare_inputs_embeds`。
4. **性能**：ROI token 数量可能很大（大框覆盖 200+ patch）。需要你手动测显存，必要时限制 `row_end-row_start` 最大值或做 stride>1 采样。
5. **多 ROI**：当前设计一次只处理一个 ROI。未来如需多个，可在 `roi_context` 前插入 `[BOS_i]` 标记区分。暂时不实现，代码里不要提前做复杂循环。

---

## 7. 实现顺序建议

1. `answer()` 中缓存 `vision_tokens` 并打印 shape。
2. 写一个小脚本：  
   - 手动指定 `bbox`  
   - 取出 patch embedding → projector → 用 `torch.randn_like` 替代语言 context  
   - 确认能通过 `qwen_model(inputs_embeds=...)` 正常运行。
3. 接好 CLI：`cot resample <mask_idx> <question>`，输出 thought + 新回答。
4. 最后再考虑训练/调优 projector。

---

该方案的重点是“只要 ROI → patch 映射正确，剩余部分都是 deterministic 的线性代数”。调试时宁可 `assert` 失败，也不要写过多兜底逻辑，以免掩盖错误。实现完成后，再根据需要添加更友好的提示。 
