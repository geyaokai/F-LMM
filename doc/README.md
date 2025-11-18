# F-LMM 项目文档导航

> 📚 **欢迎使用 F-LMM 项目文档**  
> 本文档提供清晰的导航，帮助你快速找到所需信息

---

## 📖 文档分类

### 🚀 [00-getting-started](./00-getting-started/) - 快速开始

**适合新手和快速上手使用**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`README.md`](./00-getting-started/README.md) | 项目总览与快速开始 | 安装、运行、基本概念 |
| [`todo.md`](./00-getting-started/todo.md) | 任务清单与学习路径 | 目标、进度 |

**推荐阅读顺序**：
1. README.md - 了解项目概况
2. todo.md - 明确学习目标

---

### 🏗️ [01-architecture](./01-architecture/) - 架构设计

**深入理解项目架构**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`gykreadme.md`](./01-architecture/gykreadme.md) | 项目详细说明（核心文档） | 目录结构、设计思想 |
| [`MODEL_STRUCTURE.md`](./01-architecture/MODEL_STRUCTURE.md) | 模型结构详解 | Frozen LMM、Mask Head、SAM |
| [`DATASET_STRUCTURE.md`](./01-architecture/DATASET_STRUCTURE.md) | 数据集结构说明 | PNG、RefCOCO、预处理 |

**推荐阅读顺序**：
1. gykreadme.md - 全局视角
2. MODEL_STRUCTURE.md - 模型细节
3. DATASET_STRUCTURE.md - 数据处理

---

### 🎯 [02-training](./02-training/) - 训练指南

**训练相关的所有文档**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`train.md`](./02-training/train.md) | 训练基础指南 | 命令、参数、配置 |
| [`RUNNER_AND_TRAINING.md`](./02-training/RUNNER_AND_TRAINING.md) | Runner 与训练流程详解 | 深度原理、流程图 |
| [`TRAINING_CONCEPTS_AND_RESUME.md`](./02-training/TRAINING_CONCEPTS_AND_RESUME.md) | 训练概念与恢复 | Checkpoint、Resume |
| [`TRAINING_DIRECTORIES.md`](./02-training/TRAINING_DIRECTORIES.md) | 训练目录结构 | 工作目录、日志 |

**推荐阅读顺序**：
1. train.md - 快速开始训练
2. RUNNER_AND_TRAINING.md - 理解训练流程
3. TRAINING_CONCEPTS_AND_RESUME.md - 高级功能
4. TRAINING_DIRECTORIES.md - 文件组织

---

### 🧪 [03-testing](./03-testing/) - 测试文档

**测试相关说明（链接到 tests/ 目录）**

| 文档 | 说明 | 位置 |
|------|------|------|
| **测试代码** | 单元测试和验证脚本 | [`../tests/`](../tests/) |
| **测试索引** | 测试文档完整索引 | [`../tests/README.md`](../tests/README.md) |

**主要测试脚本**：
- `test_frozen_qwen.py` - Qwen 模型完整测试
- `diagnose_image_grid_thw.py` - 图像网格诊断
- `verify_data_pipeline.py` - 数据管道验证

---

### 🔧 [04-qwen-adaptation](./04-qwen-adaptation/) - Qwen 模型适配

**Qwen2.5-VL 和 Qwen3-VL 适配文档**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`QWEN_MODEL_ADAPTATION.md`](./04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md) | Qwen 模型适配指南 | 接口、差异、实现 |
| [`TOKEN_ENCODING_FIX_CN.md`](./04-qwen-adaptation/TOKEN_ENCODING_FIX_CN.md) | Token 编码问题修复 | 图像 token、编码 |

**推荐阅读顺序**：
1. QWEN_MODEL_ADAPTATION.md - 适配总览
2. TOKEN_ENCODING_FIX_CN.md - 常见问题

---

### 🔍 [05-troubleshooting](./05-troubleshooting/) - 故障排除

**常见问题和解决方案**

| 类型 | 说明 |
|------|------|
| **训练问题** | 警告、错误、性能问题 |
| **数据问题** | 格式、加载、预处理 |
| **模型问题** | 适配、接口、维度 |

**常见问题快速链接**：
- 图像 token 数量不匹配警告 → 查看本次对话记录
- image_grid_thw 相关错误 → 查看 tests/README.md

---

### 📦 [archive](./archive/) - 历史文档

**已过期或仅供参考的历史文档**

这里存放：
- 修复过程记录
- 版本对比文档
- 临时调试文档
- 历史总结

**注意**：这些文档可能已过时，仅作为历史参考。

---

## 🎯 按场景查找

### 场景 1：我是新手，想快速了解项目
👉 按顺序阅读：
1. [`00-getting-started/README.md`](./00-getting-started/README.md)
2. [`01-architecture/gykreadme.md`](./01-architecture/gykreadme.md)
3. [`02-training/train.md`](./02-training/train.md)

### 场景 2：我要运行训练
👉 查看：
1. [`02-training/train.md`](./02-training/train.md) - 基础命令
2. [`02-training/TRAINING_CONCEPTS_AND_RESUME.md`](./02-training/TRAINING_CONCEPTS_AND_RESUME.md) - 高级功能

### 场景 3：我要适配 Qwen 模型
👉 查看：
1. [`04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](./04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)
2. [`../tests/README.md`](../tests/README.md) - 运行测试
3. [`01-architecture/MODEL_STRUCTURE.md`](./01-architecture/MODEL_STRUCTURE.md) - 理解模型结构

### 场景 4：我遇到了错误
👉 查看：
1. [`05-troubleshooting/`](./05-troubleshooting/) - 常见问题
2. [`../tests/README.md`](../tests/README.md) - 运行诊断工具
3. 相关模块的详细文档

### 场景 5：我要理解数据处理
👉 查看：
1. [`01-architecture/DATASET_STRUCTURE.md`](./01-architecture/DATASET_STRUCTURE.md)
2. 源码：`flmm/datasets/`

### 场景 6：我要理解训练流程
👉 查看：
1. [`02-training/RUNNER_AND_TRAINING.md`](./02-training/RUNNER_AND_TRAINING.md)
2. [`02-training/TRAINING_CONCEPTS_AND_RESUME.md`](./02-training/TRAINING_CONCEPTS_AND_RESUME.md)

---

## 📊 文档概览

```
doc/
├── README.md                           # 本文档（总索引）
│
├── 00-getting-started/                 # 快速开始
│   ├── README.md                       # 项目总览
│   └── todo.md                         # 任务清单
│
├── 01-architecture/                    # 架构设计
│   ├── gykreadme.md                   # 项目详细说明（重要）
│   ├── MODEL_STRUCTURE.md             # 模型结构
│   └── DATASET_STRUCTURE.md           # 数据集结构
│
├── 02-training/                        # 训练指南
│   ├── train.md                       # 训练基础
│   ├── RUNNER_AND_TRAINING.md         # Runner 详解
│   ├── TRAINING_CONCEPTS_AND_RESUME.md # 训练概念
│   └── TRAINING_DIRECTORIES.md        # 目录结构
│
├── 03-testing/                         # 测试（链接到 tests/）
│   └── README.md → ../tests/README.md
│
├── 04-qwen-adaptation/                 # Qwen 适配
│   ├── QWEN_MODEL_ADAPTATION.md       # 适配指南
│   └── TOKEN_ENCODING_FIX_CN.md       # Token 修复
│
├── 05-troubleshooting/                 # 故障排除
│   └── README.md                      # 常见问题
│
└── archive/                            # 历史文档
    └── (从 tests/ 移动过来的修复记录等)
```

---

## 🔗 相关资源

### 代码目录
- **模型实现**：[`../flmm/models/`](../flmm/models/)
- **数据处理**：[`../flmm/datasets/`](../flmm/datasets/)
- **配置文件**：[`../configs/`](../configs/)
- **测试代码**：[`../tests/`](../tests/)

### 外部资源
- **论文**：[F-LMM: Grounding Frozen Large Multimodal Models](https://arxiv.org/abs/2406.05821)
- **HuggingFace Models**：
  - [LLaVA](https://huggingface.co/llava-hf)
  - [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
  - [DeepSeek-VL](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)

---

## ✨ 文档维护

### 更新记录
- **2025-11-09**：重新组织文档结构，创建分类目录和统一索引
- **2025-11-08**：添加 Qwen 测试和修复文档

### 贡献指南
添加新文档时，请：
1. 确定文档所属分类
2. 放入对应的目录
3. 更新本 README.md 的索引

### 文档规范
- **文件命名**：使用有意义的英文名称或中文拼音
- **中英文**：技术文档优先英文，说明文档可用中文
- **链接**：使用相对路径
- **格式**：Markdown，遵循统一格式

---

**最后更新**：2025-11-09  
**维护者**：AI Assistant & Contributors


