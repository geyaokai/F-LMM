# Web Demo 端到端逻辑说明

本文串联前端、后端与模型侧（`frozen_qwen.py`）的主要流程，便于排查「回答对但找不到 / 代词不接上下文」等问题。

## 组件与入口
- 前端：`scripts/demo/web/frontend/src/App.tsx`  
  - 左侧对话区；右侧图像展示、文件/URL 上传。  
  - 调用 `/session/create` → `/load_image` → `/ask` → `/ground`。
- 后端：`scripts/demo/web/backend/main.py`  
  - 基于 FastAPI，维护 SessionStore；串行执行模型（`backend.model_lock`）。  
  - 关键 handler：`ask` 使用 `pipeline_default_ask`；`ground` 复用 `handle_ground`。
- CLI 调试：`scripts/demo/interact.py`  
  - 与后端共享同一流水线逻辑（见下）。便于本地复现。
- 核心模型：`flmm/models/frozen_qwen.py`  
  - `answer`：文本生成，同时抓取 hidden_states / attention_maps / vision_tokens。  
  - `ground`：用回答里短语对应的 token span 的 cross-attention → mask head/SAM。  
  - `visual_cot_resample`：用已缓存 vision tokens + ROI bbox 做二次回答（ROI re-answer）。

## 会话与上下文
- `SessionState.history` 保存最近 `max_history_turns` 轮的 user/assistant 对；`--reset-history` 或 `/clear` 清空。  
- `model.answer` 会把 history 作为对话上下文与系统提示拼接，生成当前回答。  
- ROI re-answer（`visual_cot_resample`）仅使用当前问题 + ROI，占用缓存的 vision tokens；不再带历史。

## 默认问答流水线（`pipeline_default_ask`）
1. 调 `model.answer(image, question, history, max_new_tokens)` 得到 `answer_text`、`hidden_states`、`attention_maps`、`meta_data`、`vision_tokens`。
2. 抽短语：`extract_phrases_via_model`（小 prompt）→ `build_phrase_candidates`（字符/Token 对齐）。
3. 自动 ground：按前 `auto_topk` 个短语调用 `perform_ground`（封装 `model.ground`）。  
   - 仅能 ground “回答里出现的短语”——因为 token span 来自回答文本。  
4. ROI re-answer：若有有效 bbox，调用 `visual_cot_resample` 对 ROI 再答一次，作为最终答复。  
5. 历史追加：当前问题 + 最终答案写入 `history`，用于下轮代词/多轮语境。

## Prompt 与生成参数
- 系统提示：`cfg.prompt_template['SYSTEM']`；后端可通过环境变量 `FLMM_WEB_EXTRA_PROMPT` 追加用户问题尾部。  
- 反垃圾提示：在 `_build_conversation`、`visual_cot_resample` 中追加“不要输出 addCriterion 等无关字符”。  
- 生成：`do_sample=False`，可按需加 `repetition_penalty` / `no_repeat_ngram_size` 以抑制 “the the…” 回环。

## Tokenizer / Padding 关键设置
- `frozen_qwen.py` 在加载时强制 `tokenizer.padding_side = 'left'`，`pad_token` 回退到 EOS，避免批量/单样本 padding 方向错配导致截断或乱码。  
- `visual_cot_resample` 通过 `<|vision_start|> ... <|vision_end|>` 占位，将 ROI tokens 注入 `inputs_embeds` 实现局部重答。

## 常见现象与定位
- 代词问法未触发 ROI re-answer：如果本轮回答未显式提及目标名词，短语列表为空 → 不会 ground → 不会 ROI re-answer；可通过改写提示或加入“复用上一轮 bbox”兜底（尚未默认启用）。
- 输出重复 “the the…”：缺少重复惩罚导致的生成退化；可在 `model.answer` / `visual_cot_resample` 的 `generate` 参数中增加 `repetition_penalty`，或前端改写问句更具体。
- “addCriterion” 乱码：通常由 padding_side 错、batch 不一致或 transformers 版本；当前代码已锁定 left padding 和 4.51.3。

## 相关文件索引
- 交互入口：`scripts/demo/interact.py`  
  - `pipeline_default_ask`、`perform_ground`、`handle_ground`。  
- 后端路由：`scripts/demo/web/backend/main.py`  
  - `/ask` 调 `pipeline_default_ask`，`/ground` 调 `handle_ground`。  
- 模型实现：`flmm/models/frozen_qwen.py`  
  - `answer` / `ground` / `visual_cot_resample`，以及 tokenizer 设置。  
- 前端调用：`scripts/demo/web/frontend/src/App.tsx`  
  - Session 管理、提问、ground 触发及结果展示。

## 可选改进（未默认启用）
- 代词兜底：若本轮无短语可用，自动复用上一轮的 bbox 再跑 `visual_cot_resample`。  
- 生成去重复：在 `generate` 里统一加 `repetition_penalty=1.1~1.2`、`no_repeat_ngram_size=3`。  
- 问题改写：前端对 “it/they/that” 自动替换为上一轮已命名目标，提升短语显式度。

（本文件仅记录逻辑，不改动代码行为。） 
