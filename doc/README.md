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
| [`RESAMPLE_VISUAL_COT.md`](./04-qwen-adaptation/RESAMPLE_VISUAL_COT.md) | ROI / Visual CoT 重推理实现 | ROI、重推理、纠错 |

**推荐阅读顺序**：
1. QWEN_MODEL_ADAPTATION.md - 适配总览
2. RESAMPLE_VISUAL_COT.md - 理解 ROI 重推理

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

### 🧭 [06-explainable-framework](./06-explainable-framework/) - 可解释框架推进

**面向课题推进的规划与实验入口**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`README.md`](./06-explainable-framework/README.md) | 本专题入口与常用命令 | demo、命令、导航 |
| [`DEMO_USAGE_AND_CODE_GUIDE.md`](./06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md) | demo 使用说明与代码主逻辑 | interact、stability、worker |
| [`ATTENTION_DIRECTIONS.md`](./06-explainable-framework/ATTENTION_DIRECTIONS.md) | attention 方向与 `images_seq_mask` 说明 | token-to-region、region-to-token |
| [`ROADMAP.md`](./06-explainable-framework/ROADMAP.md) | 三层技术路线的工程化推进计划 | grounding、解释、纠错、VCD |
| [`STABILITY_MANIFEST_GUIDE.md`](./06-explainable-framework/STABILITY_MANIFEST_GUIDE.md) | 第一阶段样例集制定指南 | manifest、样例、失败类型 |

**推荐阅读顺序**：
1. README.md - 先看命令入口和文档导航
2. DEMO_USAGE_AND_CODE_GUIDE.md - 再看 demo 怎么跑、代码怎么走
3. ATTENTION_DIRECTIONS.md - 再看 attention 的方向解释
4. ROADMAP.md - 最后看推进顺序
5. STABILITY_MANIFEST_GUIDE.md - 按统一格式整理固定样例

---

### 🌐 [webdemo](./webdemo/) - Web Demo 与前后端对接

**面向后端启动、前端接手与 explainable UI 对接**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`README.md`](./webdemo/README.md) | Web demo 端到端逻辑说明 | session、backend、frontend、模型复用 |
| [`FRONTEND_AGENT_HANDOFF.md`](./webdemo/FRONTEND_AGENT_HANDOFF.md) | 给前端开发者 / agent 的交接文档 | API、tasks、静态资源、上线边界 |
| [`../scripts/demo/web/backend/README.md`](../scripts/demo/web/backend/README.md) | 后端与 worker 启动命令 | uvicorn、worker、环境变量 |

**推荐阅读顺序**：
1. FRONTEND_AGENT_HANDOFF.md - 先明确前端应该接什么
2. webdemo/README.md - 再看后端和模型链路
3. scripts/demo/web/backend/README.md - 最后按命令启动服务

---

### 🗂️ [07-execution-plan](./07-execution-plan/) - 执行文档与交付顺序

**面向“新对话直接执行”的项目级文档**

| 文档 | 说明 | 关键词 |
|------|------|--------|
| [`README.md`](./07-execution-plan/README.md) | runtime、直连接入、纠错层的统一执行文档 | runtime、deploy、verification、correction |
| [`../agent.md`](../agent.md) | 给后续 agent / subagent 的仓库级协作说明 | agent、review、commit、self-check |

**推荐阅读顺序**：
1. `../agent.md` - 先看新对话中的工作约束
2. `README.md` - 再看项目级执行顺序和验收标准

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

### 场景 7：我要推进“可解释 + 纠错”课题
👉 查看：
1. [`06-explainable-framework/README.md`](./06-explainable-framework/README.md)
2. [`06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`](./06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md)
3. [`06-explainable-framework/ROADMAP.md`](./06-explainable-framework/ROADMAP.md)
4. `scripts/demo/stability_eval.py` - 第一阶段稳定性基线脚本

### 场景 8：我要接前端、启动后端、做 explainable Web UI
👉 查看：
1. [`webdemo/FRONTEND_AGENT_HANDOFF.md`](./webdemo/FRONTEND_AGENT_HANDOFF.md)
2. [`webdemo/README.md`](./webdemo/README.md)
3. [`../scripts/demo/web/backend/README.md`](../scripts/demo/web/backend/README.md)
4. [`06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`](./06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md)

### 场景 9：我要把 runtime 独立出去，并为新对话准备可执行规范
👉 查看：
1. [`../agent.md`](../agent.md)
2. [`07-execution-plan/README.md`](./07-execution-plan/README.md)
3. [`webdemo/FRONTEND_AGENT_HANDOFF.md`](./webdemo/FRONTEND_AGENT_HANDOFF.md)
4. [`06-explainable-framework/ROADMAP.md`](./06-explainable-framework/ROADMAP.md)

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
│   └── RESAMPLE_VISUAL_COT.md         # ROI / Visual CoT 重推理
│
├── 05-troubleshooting/                 # 故障排除
│   └── README.md                      # 常见问题
│
├── 06-explainable-framework/           # 可解释框架推进
│   ├── README.md                      # 本专题入口
│   ├── DEMO_USAGE_AND_CODE_GUIDE.md   # demo 使用与代码主逻辑
│   ├── ATTENTION_DIRECTIONS.md        # attention 方向说明
│   └── ROADMAP.md                     # 研究路线与实施计划
│
├── 07-execution-plan/                 # 执行文档与交付顺序
│   └── README.md                      # runtime / 直连接入 / 纠错层执行文档
│
├── webdemo/                            # Web demo 与前后端对接
│   ├── README.md                      # 端到端逻辑说明
│   └── FRONTEND_AGENT_HANDOFF.md      # 给前端开发者的交接文档
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
- **2026-03-23**：新增 runtime / 直连接入 / 纠错层执行文档与 `agent.md`

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

**最后更新**：2026-03-23
**维护者**：AI Assistant & Contributors
