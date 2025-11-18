# F-LMM 快速开始

> 🚀 **欢迎使用 F-LMM 项目！**  
> 本文档帮助你快速了解项目并开始使用

---

## 📖 项目简介

**F-LMM (Grounding Frozen Large Multimodal Models)** 是一个基于冻结大模型的视觉定位与分割框架。

### 核心特点

- 🔒 **冻结底座**：冻结大多模态模型（LLaVA、Qwen、DeepSeek-VL 等），仅训练轻量分割头
- 🎯 **语义定位**：通过文本引导的注意力机制实现精确分割
- 🔧 **易于适配**：支持多种底座模型，易于扩展新模型
- 📊 **高效训练**：只需训练少量参数即可获得强大的分割能力

---

## 🏗️ 系统架构（简化）

```
输入：图像 + 文本描述
    ↓
冻结的大模型（LLaVA/Qwen/DeepSeek等）
    ↓ 提取
注意力图 + Hidden States
    ↓ 聚合
UNet Mask Head
    ↓ 生成
粗粒度 Mask
    ↓ 细化
SAM（Segment Anything Model）
    ↓
精细分割结果
```

---

## 📋 前置要求

### 硬件要求
- **GPU**：建议 2x A100/A800 80GB（可使用 DeepSpeed ZeRO-2 优化）
- **内存**：至少 32GB RAM
- **磁盘**：至少 200GB 用于数据和模型

### 软件依赖
- Python 3.10+
- PyTorch 2.2+
- CUDA 11.8+
- 详见 `requirements.txt`

---

## ⚡ 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/your-org/F-LMM.git
cd F-LMM
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. 准备数据

详见项目根目录的 `README.md` 或 [`../01-architecture/DATASET_STRUCTURE.md`](../01-architecture/DATASET_STRUCTURE.md)

### 4. 下载预训练权重

```bash
# SAM 权重
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoints/sam_vit_l_0b3195.pth

# 底座模型会自动从 HuggingFace 下载
```

---

## 🎯 第一次运行

### 运行测试（推荐）

```bash
cd tests
python test_frozen_qwen.py
```

### 运行训练（DeepSeek-VL 示例）

```bash
export PYTHONPATH=.
NPROC_PER_NODE=2 xtuner train \
  configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py \
  --deepspeed deepspeed_zero2
```

### 运行推理（Demo）

```bash
cd scripts/demo
python grounded_conversation_demo.py --image <image_path> --text "a red car"
```

---

## 📚 学习路径

### 初学者路径（推荐）

1. ✅ **阅读本文档** - 了解基本概念
2. 📖 **阅读 [`../01-architecture/gykreadme.md`](../01-architecture/gykreadme.md)** - 深入理解架构
3. 🧪 **运行测试** - 验证环境配置
4. 🎯 **查看 [`todo.md`](./todo.md)** - 制定学习计划
5. 📝 **阅读 [`../02-training/train.md`](../02-training/train.md)** - 开始训练

### 进阶路径

1. 深入研究模型结构：[`../01-architecture/MODEL_STRUCTURE.md`](../01-architecture/MODEL_STRUCTURE.md)
2. 理解训练流程：[`../02-training/RUNNER_AND_TRAINING.md`](../02-training/RUNNER_AND_TRAINING.md)
3. 适配新模型：[`../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)

---

## 🔗 重要链接

### 本地文档
- **架构说明**：[`../01-architecture/`](../01-architecture/)
- **训练指南**：[`../02-training/`](../02-training/)
- **测试文档**：[`../../tests/README.md`](../../tests/README.md)
- **任务清单**：[`todo.md`](./todo.md)

### 外部资源
- **论文**：[arXiv:2406.05821](https://arxiv.org/abs/2406.05821)
- **GitHub**：[F-LMM Repository](https://github.com/your-org/F-LMM)
- **HuggingFace Models**：
  - [LLaVA](https://huggingface.co/llava-hf)
  - [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
  - [DeepSeek-VL](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)

---

## ❓ 常见问题

### Q: 训练需要多久？
A: 在 2x A800 上，RefCOCO 数据集训练约需 8-12 小时（8 epochs）

### Q: 可以用更小的 GPU 吗？
A: 可以，但需要：
- 减小 batch size
- 使用梯度累积
- 启用 DeepSpeed ZeRO-3

### Q: 如何适配新的底座模型？
A: 参考 [`../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)

### Q: 遇到错误怎么办？
A: 查看 [`../05-troubleshooting/`](../05-troubleshooting/) 或运行诊断工具：
```bash
cd tests
python diagnose_image_grid_thw.py
```

---

## 📞 获取帮助

1. **查看文档**：[`../README.md`](../README.md) - 完整索引
2. **运行测试**：`python tests/test_frozen_qwen.py`
3. **查看示例**：`scripts/demo/`
4. **问题排查**：[`../05-troubleshooting/`](../05-troubleshooting/)

---

## ✨ 下一步

- [ ] 浏览 [`todo.md`](./todo.md) 了解学习任务
- [ ] 阅读 [`../01-architecture/gykreadme.md`](../01-architecture/gykreadme.md) 深入理解
- [ ] 运行第一个训练实验
- [ ] 探索其他文档

---

**最后更新**：2025-11-09  
**维护者**：AI Assistant


