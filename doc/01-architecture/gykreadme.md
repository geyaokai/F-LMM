# F-LMM 项目说明（gykreadme）

本文档面向研发与复现实验使用者，系统性介绍 F-LMM 项目的目录结构与文件作用，其中数据集部分点到为止，重点详细说明 F-LMM 模型设计与关键实现。

## 顶层结构概览

```
F-LMM/
├── checkpoints/           # 预训练与已训好的权重存放目录
├── configs/               # 训练/评测配置（按底座模型分类）
├── data/                  # 数据目录（COCO/PNG/RefCOCO/VisCoT 等）
├── deepseek_vl/           # DeepSeek-VL 相关适配与封装
├── flmm/                  # 项目核心：数据处理、模型、Runner 等
├── hpt/                   # HPT 模型相关文件（SigLIP 等）
├── images/                # 论文与示意图
├── llava/                 # LLaVA 与 LLaVA-Next 适配代码
├── mgm/                   # MiniGemini (MGM) 适配与推理工具
├── scripts/               # 训练/评测/可视化脚本与 Demo
├── segment_anything/      # SAM 模型与工具（Mask 生成/细化）
├── the/                   # 结果/临时输出示例目录
├── dataoverview.ipynb     # 数据概览 Notebook
├── README.md              # 官方简要 README
├── requirements.txt       # 依赖清单
└── LICENSE                # 许可证
```

## 目录与文件作用说明

- checkpoints/
  - 存放已下载或训练出的权重文件。
  - 示例：`sam_vit_l_0b3195.pth`（SAM 预训练），`frozen_*_unet_sam_l_refcoco_png.pth`（各底座 LMM 的 F-LMM 训练权重）。

- configs/
  - 各底座大模型的训练/评测配置，基于 Xtuner 与 MMEngine 配置体系。
  - 子目录：
    - deepseek_vl/：DeepSeek-VL 相关配置。
    - hpt/：HPT（HyperGAI/HPT-Air）相关配置。
    - llava/：LLaVA-1.5 等配置。
    - llava_next/：LLaVA-Next 系列配置。
    - mgm/：MiniGemini（Vicuna/Gemma 等）配置。

- data/
  - 数据根目录，结构与官方 README 基本一致。
  - 子目录：
    - coco/：COCO 与 PNG、RefCOCO 系列标注与图像。
    - cot/：VisCoT 视觉链路推理相关测试数据。
  - 提示：数据准备可按 `README.md` 的 Data Preparation 进行，本文不展开数据细节。

- deepseek_vl/
  - DeepSeek-VL 的模型与工具适配封装。
  - `models/`、`utils/`：与 DeepSeek-VL 的输入输出对齐、隐藏状态/注意力导出等支持。

- flmm/（核心模块）
  - `datasets/`：输入预处理、对齐与拼接、mask/文本耦合等数据管线。
    - `hpt_processors.py`：HPT 数据处理器。
    - `llava_processors.py`、`llava_next_processors.py`：LLaVA(LLaVA-Next) 数据处理器。
    - `pad2square_processor.py`：将图像 pad 到方形，便于 patch/grid 对齐。
    - `png.py`：PNG/RefCOCO 任务样本组织。
    - `transforms.py`：常用图像/张量变换。
  - `models/`：F-LMM 核心“冻结大模型 + 分割头(SAM+UNet)”架构实现。
    - `frozen_llava.py`：基于 LLaVA 的冻结推理与注意力到 mask 的映射。
    - `frozen_llava_next.py`：LLaVA-Next 版本，支持 coarse/fine 双尺度拼接。
    - `frozen_mgm.py`：MiniGemini（多图/高分辨率 grid + global）的冻结推理与映射。
    - `frozen_hpt.py`、`frozen_deepseek_vl.py`：HPT、DeepSeek-VL 版本适配（结构与上类似）。
    - `mask_head/`：分割头实现（UNet 解码器与细化模块）。
      - `mask_decoder.py`：`UNetHead` 解码头，接受由注意力/特征拼接形成的“注意力地图堆叠”并输出二值 mask。
      - `mask_refiner.py`：mask 细化器（如需进一步边界修正）。
  - `runner.py`：自定义 Runner，兼容 Xtuner/MMEngine 的训练流程，定制加载/保存 checkpoint 行为（仅保存可训练参数等）。
  - `utils.py`：通用工具，如 `compute_mask_IoU`、`multi_apply`。

- hpt/
  - `modeling_siglip.py`：HPT 模型（SigLIP）适配/封装，供 `flmm/models/frozen_hpt.py` 使用。

- llava/
  - `modeling_llava.py`、`modeling_llava_next.py`：对接 HuggingFace LLaVA/LLaVA-Next，实现输出 hidden states/attentions 与图像 token mapping。

- mgm/
  - MiniGemini 适配与对话/推理工具。
  - `constants.py`、`conversation.py`、`mm_utils.py`、`model/`、`utils.py`：构建与大模型交互的“多图/多尺度”输入形式，提供注意力与 hidden states。

- scripts/
  - 训练与评测脚本、Demo 与可视化工具。
  - 常用：
    - `multiprocess_eval_png.py`：PNG 多进程评测入口（Accelerate）。
    - `multiprocess_eval_refcoco.py`：RefCOCO 系列评测入口。
    - `demo/`：推理演示（多进程 PNG 推理、grounded 对话可视化等）。
    - `visual_cot/`：VisCoT 推理与 GPT 评测脚本。

- segment_anything/
  - 引入 Meta 的 SAM（Segment Anything Model）与其工具：
    - `build_sam.py`：构建 SAM 模型。
    - `automatic_mask_generator.py`、`predictor.py`、`modeling/`、`utils/`：用于根据 F-LMM 提供的初始 mask 或提示进一步细化分割。

- the/
  - 结果输出示例目录（如可视化、JSON 推理结果等）。

- 其余文件：
  - `README.md`：官方简要说明（依赖、数据准备、训练/测试命令、权重链接）。
  - `requirements.txt`：项目依赖（transformers=4.39.1 等版本建议）。
  - `dataoverview.ipynb`：数据概览 Notebook。

## F-LMM 模型设计与关键实现（重点）

F-LMM（Grounding Frozen Large Multimodal Models）的核心思想：
- 冻结大多模态大模型（LLaVA / LLaVA-Next / DeepSeek-VL / MiniGemini / HPT 等）的参数，仅作为“语义与视觉对齐特征/注意力”提供器；
- 将这些注意力/隐藏状态经过“层加权与通道聚合”后，映射为与图像空间对齐的注意力地图堆叠；
- 使用轻量的分割头（UNet 解码器 + 可选 SAM 细化）进行像素级分割学习；
- 训练时仅优化分割头（以及极少量可学习权重，如文本层权重），以较低成本在多数据集上获得强分割泛化能力。

### 1) 冻结底座 LMM 的统一范式（Frozen 模型族）

对应代码位于 `flmm/models/`：
- `FrozenLlava(FrozenLlavaSAM)`：见 `frozen_llava.py`
- `FrozenLlavaNext(FrozenLlavaNextSAM)`：见 `frozen_llava_next.py`
- `FrozenMGM(FrozenMGMSAM)`：见 `frozen_mgm.py`
- `FrozenDeepSeekVL(FrozenDeepSeekVLSAM)`、`FrozenHPT(FrozenHPTSAM)`：与上类似（通过各自适配模块构建）。

通用结构要点：
- 冻结 LMM（`self.llava` / `self.mgm` 等）参数：`requires_grad_(False)`；
- 导出文本隐藏层 `hidden_states` 与跨模态注意力 `attentions`，仅进行推理前向；
- 可学习文本层权重 `text_layer_weights`（shape: num_layers），以 softmax 归一后对不同层的 hidden states 加权求和，得到文本 token 表征；
- 将与图像 token 对齐的注意力张量，按 mask 语义位置聚合并“合并维度”（mean/max），再堆叠成“注意力地图集”作为分割头输入；
- LLaVA-Next、MGM 等会区分 coarse/fine 或 grid/global 等多尺度特征，最终在空间上对齐并拼接通道；
- 文本特征通过 `text_proj` 投影到 SAM Prompt Encoder 的维度，配合初始 mask 作为 SAM 的提示，输出更精细的 `sam_pred_masks`。

训练/推理接口：
- `forward(..., mode='loss'|'predict'|'tensor')`：与 MMEngine 训练流程对齐；
- `compute_loss`：对 `pred_masks` 与 `sam_pred_masks` 分别计算 `dice` 与 `bce`（`loss_mask`）等，统计 `accuracy`/`aIoU`；
- `predict`：仅返回 `sam_pred_masks`，便于评测与部署。

关键张量匹配：
- 文本 token 与 `mask_ids` 对齐：仅取与某一目标短语/实体相对应的 token 索引（`mask_ids == mask_id`）进行注意力/隐藏状态聚合；
- 图像空间对齐：根据视觉塔 patch size 或多尺度形状，重排 `attentions` 到 `[H, W]` 空间，对齐原图 pad/crop 后的有效区域；
- 多尺度拼接（LLaVA-Next、MGM）：将 coarse 与 fine（或 grid 与 global）的注意力分别上采样至同一分辨率后在通道维拼接，输入分割头。

### 2) 分割头（mask_head：UNetHead + SAM 细化）

核心在 `flmm/models/mask_head/mask_decoder.py`：
- `UNetHead(UNet)`：继承自 MMSeg 的 UNet，实现：
  - 按需对输入注意力图进行归一化/上采样；
  - 动态 padding 以适配 2^(num_stages-1) 的下采样栈深度，确保卷积/上采样后尺寸还原；
  - 最终 `1x1 conv` 输出单通道 mask logit；
- `mask_refiner.py`（可选）：进一步边界/细节修正；
- SAM 细化：各 `*SAM` 类在 `_forward` 中调用 `self.sam(image, pred_masks, text_embeds)`，利用 SAM 的 Prompt Encoder + Mask Decoder 进一步提升细粒度边界质量。

训练目标：
- 使用 `loss_dice` + `loss_mask`（BCE/CE）共同监督，指标统计包含 `accuracy` 和 `aIoU`；
- 同时对 `pred_masks`（UNetHead 输出）与 `sam_pred_masks`（SAM 细化输出）计算并汇总损失（权重在配置中控制）。

### 3) 关键实现细节与易错点

- 层加权：`text_layer_weights` 使用 softmax 归一，避免任意层被过度放大，训练中可自适应选择更有用的语义层；
- 合并策略：`merge` 支持 `mean` 与 `max`，控制多头/多层注意力在聚合时的鲁棒性与尖锐度；
- 图像尺寸与 pad 对齐：注意 UNet 输入 padding 与原图 `padded_shape`/`padding` 元信息还原，确保裁剪回原始 ROI；
- 多尺度注意力重排：LLaVA-Next 与 MGM 对粗/细/全局/网格注意力有不同拼接顺序，需与各自 `image_feature_shapes`/`image_places` 严格匹配；
- 冻结底座：务必确保大模型处于 eval/frozen 状态，避免显存暴涨与训练不稳定；
- 版本兼容：`transformers` 建议固定在 `4.39.1`，更高版本可能导致性能或行为差异。

## 训练与评测（概览）

- 训练（示例，DeepSeek-VL）：
  - `xtuner train configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py --deepspeed deepspeed_zero2`

- 评测（示例，PNG）：
  - `accelerate launch scripts/multiprocess_eval_png.py <config.py> --checkpoint <ckpt_path>`

详细命令行示例、可用权重与数据准备，请参考根目录 `README.md`。

## 常见问答（FAQ）

- Q: 一定要用 SAM 吗？
  - A: UNetHead 的输出已可用，SAM 作为可选的细化模块，通常能明显提升边界与薄物体性能。

- Q: 更换底座 LMM 的工作量大吗？
  - A: 需要提供：输入预处理器（datasets/processors）、模型前向适配（导出 hidden_states/attentions 与 image token 对齐信息）、以及 `frozen_*.py` 中的拼接/重排逻辑与 `mask_head` 通道设置。

- Q: 训练显存如何估算？
  - A: 主要看底座 LMM 的显存占用（尽管冻结但仍需前向）与注意力堆叠后的通道数、UNet 深度。可通过减小图像尺寸、采用梯度累积与 Deepspeed ZeRO-2 缓解。

## 参考

- 论文：F-LMM: Grounding Frozen Large Multimodal Models（arXiv:2406.05821）
- 依赖：Xtuner、MMEngine、MMSegmentation、MMDetection、Transformers 4.39.1、Accelerate
- 上游模型：LLaVA、DeepSeek-VL、MiniGemini、HPT、SAM


