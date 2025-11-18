# Qwen2.5-VL 与 DeepSeek-VL / LLaVA 适配差异说明

> 本文基于原项目 [F-LMM](https://github.com/wusize/F-LMM) 的主干实现，总结了为支持 Qwen2.5-VL 所做的核心改动，并与既有的 DeepSeek-VL、LLaVA 路径进行逐项对比，方便后续维护与迁移。

## 1. 总览对比

| 模块 | Qwen2.5-VL (`flmm/models/frozen_qwen.py`) | DeepSeek-VL (`flmm/models/frozen_deepseek_vl.py`) | LLaVA-Next (`flmm/models/frozen_llava_next.py`) |
| --- | --- | --- | --- |
| 视觉 patch 大小 | 读取 `processor.image_processor.patch_size`（默认 14），支持动态分辨率 | 固定 `patch_size=16` / `clip_shape=24`，默认 576 patch | 读取模型 `vision_config.patch_size`，分 coarse / fine feature |
| 图像 token 个数 | `image_grid_thw` × `merge_size` 自动推断；fallback 也会对齐 | 训练配置中直接铺满 576 个 `<image_placeholder>` | 模型前向返回 fine/coarse 图像特征及 `image_feature_shapes` |
| `merge_size` | 运行时读取 `processor.image_processor.merge_size`（默认 2），影响 token 展开与位置编码 | 未使用；依赖 SigLIP 编码固定网格 | 无对应概念；依赖 CLIP ViT & Llava projector |
| 回退策略 | 当 `HF processor.__call__` 失败时，自动使用 `image_processor` 推算 `image_grid_thw` 并重新构造文本 | 无特殊回退，输入需符合 DeepSeek 预处理格式 | 直接调用 LLaVA 模型封装，不展示 fallback |
| 训练日志 | 关注动态分辨率警告；新增 `Merge size: {self.merge_size}` 等调试输出 | 主要打印 `<image_placeholder>` ID 与注意力信息 | 依赖 LLaVA 原生日志，关注 coarse/fine 对齐 |

## 2. 模型封装层差异

### 2.1 Qwen2.5-VL
- 入口：`flmm/models/frozen_qwen.py`。
- 关键改动：
  - 初始化阶段读取 `patch_size` 与 `merge_size`，在 `_forward` 中严格校验 `image_grid_thw` 与视觉 token 数量。（相关逻辑集中在 `L92-207`）
  - 若数据集中缺失 `image_grid_thw`，可依据原图尺寸自动推算并打印 `Calculated image_grid_thw (aligned to merge_size)`。（`L279-323`）
  - 在重塑注意力图时，优先使用 `model_kwargs['image_grid_thw']` 推导视觉网格，只有在缺失时才回退到 `meta_data`。（`L352-383`）
  - 注意力重塑的最后一步需要把一维权重恢复成二维网格，因此会显式推导 Qwen 视觉 token 对应的 `qwen_h` / `qwen_w`（参见 `L377-431`）。推导顺序为：
    1. 若 `model_kwargs` 提供了 `image_grid_thw`，先读取其中的 height / width，除以 `merge_size` 得到 `qwen_h`、`qwen_w`，并计算期望的 token 数；
    2. 若缺失，则从 `meta_data` 中的 `padded_shape` 或 `image_shape` 回退，按照 `(尺寸 // patch_size // merge_size)` 计算网格大小；
    3. 如果仍无法获知，就对实际的 `num_image_tokens` 尝试开方取整，寻找一对可整除的 h / w 作为最终网格；
    最终将注意力张量 reshape 成 `[num_heads, seq_len, qwen_h, qwen_w]`，实现跨动态分辨率的可视化与后续 mask 加权。
    > 只有当 image_grid_thw 缺失（例如外部自定义数据没带上该字段）时，代码才会回退到 meta_data 或简单按 token 数量反推。你训练过程中没有看到 “Warning: Image token count … != …” 等提示，也说明第一条分支生效。如果想再确认，可以在日志搜索 “Calculated image_grid_thw” 或 “Inferred spatial dimensions” 等关键字；若没有出现，就说明一直是通过 image_grid_thw 直接推导的。
### 2.2 DeepSeek-VL
- 入口：`flmm/models/frozen_deepseek_vl.py`。
- 特点：
  - 假设视觉输入恒定为 `24×24` patch，故 `_forward` 里直接将注意力 reshape 到固定空间尺度。（`L168-170`）
  - 训练时需要提前铺满 576 个 `<image_placeholder>`，且不支持动态 patch grid。
  - 不额外读取 `merge_size`，也无 `image_grid_thw` 的动态推断逻辑。

### 2.3 LLaVA-Next
- 入口：`flmm/models/frozen_llava_next.py`。
- 特点：
  - 模型前向会同时返回 coarse/fine 图像特征与 `image_feature_shapes`，因此 `_forward` 里将注意力拆成两个尺度再拼接。（`L115-152`）
  - 由于 LLaVA 进行多尺度融合，输入期望包含 coarse (CLIP 原始 patch grid) 与 fine (高分辨率) 两部分视觉 token。
  - 项目侧不需要额外的 `merge_size` 或 `image_grid_thw` 校正。

## 3. 数据管线与 Token 展开

### 3.1 QwenImageProcessorWrapper
- 入口：`flmm/datasets/qwen_image_processor.py`。
- 主要职责：
  - 对 `transformers.Qwen2_5_VLProcessor` 做适配，支持 `processor.__call__` 与fallback 双路线。
  - 统一返回：`pixel_values`、`meta_datas`、`image_grid_thw`、`input_ids_with_vision` 等字段；fallback 与正常路径输出保持一致。
  - fallback 时会依据原图尺寸推导 `image_grid_thw`，并依据 `merge_size` 重新展开 `<|image_pad|>`，保证视觉 token 数与 `pixel_values` 匹配。
  - 新增文本清洗，自动去除 `<|image_pad|>` / `<image>` 等显式占位符，避免与 `apply_chat_template` 二次插入造成重复。（`L100-133`、`L152-205`）

### 3.2 RefCOCO2PNG 数据变换
- 入口：`flmm/datasets/transforms.py`。
- 修改点：
  - 当检测到 `image_processor` 类型为 Qwen Processor 时，自动替换为 `QwenImageProcessorWrapper` 并记录日志。（`L86-91`）
  - Qwen 分支直接使用 `input_ids_with_vision` 作为 Prompt + vision token，不再手动替换 `<image>`，避免 tokenizer 贪心合并导致错误。（`L185-204`）
  - 如果可用，则将 `image_grid_thw` 写回 `result_dict`，供 `FrozenQwen` 在前向中使用。（`L245-247`）
- DeepSeek / LLaVA 分支仍沿用既有逻辑：
  - DeepSeek 依赖 `<image_placeholder>` 铺满文本，`RefCOCO2PNG` 会将该 token 标记到 `mask_ids` 中。
  - LLaVA 通过模型 forward 自动替换视觉 token，不依赖 `image_grid_thw`。

## 4. 文档与配置补充
- 新增 `doc/04-qwen-adaptation/` 下多篇说明文档：
  - `QWEN_MODEL_ADAPTATION.md`：概述项目集成流程与注意事项。
  - `TOKEN_ENCODING_FIX_CN.md`：记录 tokenizer 贪心合并导致 `<image>` 映射不稳定的问题及解决方案。
  - `QWEN_VS_DEEPSEEK_LLAVA.md`（即本文）：提供 Qwen/DeepSeek/LLaVA 三者差异对比。
- Qwen 专用文档：
  - `configs/qwen/README.md` / `configs/qwen/QUICKSTART.md`：补充动态分辨率训练说明、常见警告解释与调试技巧。

## 5. 训练与排障提示
- **动态分辨率警告**：若日志出现 `Image token count X != Y`，通常是 `image_grid_thw` 或 fallback 推算失配。现有实现会输出调整后的网格尺寸，可结合 `meta_data` 追踪。
- **Processor fallback 告警**：当 Hugging Face processor 报 `index 1 is out of bounds` 时，wrapper 会自动改用 `image_processor` 并打印提示，属预期行为，只要后续无 `ValueError` 即可。
- **与旧模型共存**：DeepSeek-VL 与 LLaVA 的路径未做破坏式修改，仍可按原配置运行。
