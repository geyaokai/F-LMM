# 00-getting-started / TODO

> 聚焦任务：让 **Qwen-FLMM** 在保持底座冻结的前提下，具备“先回答、再聚焦、可回流 Visual CoT”的交互体验，用最小成本验证 Qwen 适配是否成功。


## 🎯 当前核心任务：Qwen-FLMM 互动 Demo

**目标**：构建一个统一的 `interact.py` 会话脚本（后续可复用到 Web），用户可以：
1. `load` 图像后开始会话；
2. `ask` 常规问题，默认只走 Qwen 原生回答；
3. 需要定位时再 `ground` 关键词，得到掩码与可视化；
4. 如有需要，触发 Visual Chain-of-Thought（裁剪 ROI → 再次提问）以降低幻觉、提升准确度。

---

### 🧠 设计主线（Answer / Ground / Visual CoT）
- **Answer-only 默认路径**：`answer()` 快速返回文本（可携带结构化短语/char span），不必每次都跑掩码。
- **Ground 作为外挂工具**：`ground()` 接受用户选定的关键词或 token span，调用 mask head + SAM，输出掩码、可视化和可选的 ROI 裁剪。
- **Visual CoT 回流**：`refine()` 或 `inspect()` 命令把 ROI 拼入新的 prompt，再次调用大模型，展示“掩码→裁剪→再推理”的增益。
- **SessionController**：缓存当前图像的 embeddings、最近一次回答的 hidden states / attention maps / meta，支撑多轮 `ask→ground→refine`。

---

### ✅ MVP 范围（必须完成）
- [ ] 新增 `scripts/demo/interact.py`，参数支持 `config`、`checkpoint`、`--image`、`--max_new_tokens`、`--no_sam`，并提供 `load/ask/ground/inspect/exit` 命令。
- [ ] `FrozenQwenSAM` 暴露新的轻量接口：  
  - `answer(image, question, return_spans=True)` → `{text, spans, output_ids, hidden_states, attention_maps, meta}`；  
  - `ground(spans | keywords, cached_state, use_sam=True)` → `{pred_masks, sam_masks, viz_path, roi_images}`。
- [ ] 交互循环：载入一次模型与图像即可多轮 `ask`；模型返回的 `spans`（JSON）直接用于 CLI 0/1 选择，不依赖 spaCy。
- [ ] 每次 grounding 自动保存叠加掩码的 PNG/JPEG 到 `scripts/demo/results/qwen/<timestamp>/round_<idx>.png`，如有 ROI 裁剪也单独存盘。
- [ ] Visual CoT 示例命令：`inspect <span_id>` 触发 ROI → `answer`（第二轮），CLI 打印“裁剪提示 + 新回答”，并标记显存/耗时。
- [ ] 健壮性：正确清洗 `<|vision_start|>` 等特殊 token；无效命令或不存在图像路径需给出提示但不中断会话；`exit` 释放 GPU。

---

### 🌱 进阶体验（第二阶段）
- [ ] 多图会话：`load <image_path>` 切换图像但保留模型缓存，必要时复用 Processor 结果减少重复编解码。
- [ ] Web UI（Gradio/Streamlit）：展示原图、历史问答、候选短语、点击后的掩码与 ROI；CLI/Web 共享同一后端 Session。
- [ ] Visual CoT 自动化：支持 `--cot {v1|v2|v3}`，直接调用 `FrozenQwenSAM.visual_cot_*`，演示“回答→聚焦→再回答”的全流程。
- [ ] 日志增强：记录每轮 `prompt/answer/选词/IoU/ROI 尺寸/二次回答差异` 等，便于分析聚焦是否带来收益。

---

### 🧱 前置与依赖
- 保证 Qwen 推理环境：`transformers>=4.46`、`qwen-vl-utils`、`numpy<2`，可用 `setup_qwen_env_py310.sh` 一键配置后 `python -c "import transformers, qwen_vl_utils"` 验证。
- 全流程依赖配置驱动：`Config.fromfile` → `BUILDER.build` 来构造 tokenizer/processor/模型，避免与训练代码漂移。
- `data_sample['image_grid_thw']` 必须一路传递；缺失时要抛出明确的报错并提示如何修复。
- SAM 权重与 checkpoints 默认放在 `checkpoints/` 下，与 demo 参数保持一致。

---

### 🧪 验收标准
- `python scripts/demo/interact.py configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --checkpoint ... --image demo.jpg`：  
  1. `ask Where is the shampoo?` → 返回文本和候选短语列表；  
  2. `ground 0` → 保存可视化与 ROI，CLI 打印路径；  
  3. `inspect 0` → 使用 ROI 触发二次回答；  
  4. `exit` 时打印显存峰值与总耗时。
- 至少验证 Qwen2.5-VL-3B & 7B 两个配置，CLI 不需要改代码即可切换。
- Visual CoT 命令与 `scripts/visual_cot/visual_cot_inference.py` 的接口保持一致，避免割裂。
- 文档同步更新：`doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md` 或 `scripts/demo/grounded_conversation.md` 新增 “Qwen 互动 Demo” 章节，给出完整命令与示例输出。

---

### 🧰 开发流程与测试规划
1. **阶段 0｜环境体检**  
   - [ ] 运行 `bash setup_qwen_env_py310.sh`（或 QUICKSTART 等价脚本）；  
   - [ ] `python scripts/test_qwen_config.py --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py`。
2. **阶段 1｜模型接口对齐**  
   - [ ] 用 `tests/debug_frozen_qwen_forward.py` 打印 `_forward` 输出尺寸，确认 `image_grid_thw` / `mask_attentions` 正确；  
   - [ ] 编写最小脚本依次调用 `answer()`、`ground()`，断言字段齐全且 spans 与 tokenizer offset 对齐；  
   - [ ] 构造缺失 `image_grid_thw` 的样本，验证会抛出可读错误。
3. **阶段 2｜交互 Demo 联调**  
   - [ ] 通过 here-doc 自动化两轮 `ask→ground`，检查 `scripts/demo/results/qwen/` 是否生成对应图像；  
   - [ ] 切换 `--no_sam` 与默认模式，验证掩码差异；  
   - [ ] 随机输入非法命令/路径，确保 CLI 友好提示。
4. **阶段 3｜Visual CoT 验证**  
   - [ ] 对同一问题分别跑“仅回答”与“回答+ground+inspect”，比较输出差异或 hallucination 率；  
   - [ ] 运行 `python scripts/visual_cot/visual_cot_inference.py ... --discard_sam`，确保 `visual_cot_v*` 在新接口下仍可用；  
   - [ ] 记录 `load_model`、单轮 QA、ground、inspect 的耗时与显存，方便日后优化。
5. **阶段 4｜文档 & 发布**  
   - [ ] 更新文档并截图示例交互；  
   - [ ] `python -m scripts.demo.interact --help` 输出涵盖所有命令；  
   - [ ] 提供清理脚本（如 `python scripts/demo/cleanup_results.py`）避免测试产物污染仓库。

---

### 🧾 代码规范约束
- 默认科研代码风格：注释只保留必要的信息，禁止堆砌解释。
- 不写额外的 `try/except` 或复杂 `if/else` 做“防呆”，关键分支直接用 `assert` 约束张量形状/字段齐备。
- 允许抛出断言让调试暴露真实问题，宁愿失败也不吞掉错误。
- 接口命名保持简洁：`answer / ground / inspect` 等；参数顺序与类型保持一致，避免隐藏状态修改。
- 新文件默认 ASCII，log/print 信息保持精简但可定位（显存、路径、命令等必要信息）。

---

> 完成上述条目后，Qwen-FLMM 将具备“默认回答 + 按需 Ground + Visual CoT” 的完整闭环，既可展示注意力与掩码，也能真实提升图文理解的可靠度。若需新增任务，请继续在本页补充。 
