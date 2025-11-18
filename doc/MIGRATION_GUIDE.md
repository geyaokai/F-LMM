# 文档迁移指南

> 📦 **文档结构已重新组织！**  
> 本指南帮助你快速找到文档的新位置

---

## 🎯 迁移概述

**日期**：2025-11-09  
**原因**：文档分散、分类不清、tests 目录混杂了太多文档  
**结果**：清晰的分类结构、统一的索引、干净的 tests 目录

---

## 📋 文档位置对照表

### 从 `doc/` 根目录移动的文档

| 旧位置 | 新位置 | 分类 |
|--------|--------|------|
| `doc/todo.md` | `doc/00-getting-started/todo.md` | 快速开始 |
| `doc/gykreadme.md` | `doc/01-architecture/gykreadme.md` | 架构设计 |
| `doc/MODEL_STRUCTURE.md` | `doc/01-architecture/MODEL_STRUCTURE.md` | 架构设计 |
| `doc/DATASET_STRUCTURE.md` | `doc/01-architecture/DATASET_STRUCTURE.md` | 架构设计 |
| `doc/train.md` | `doc/02-training/train.md` | 训练指南 |
| `doc/RUNNER_AND_TRAINING.md` | `doc/02-training/RUNNER_AND_TRAINING.md` | 训练指南 |
| `doc/TRAINING_CONCEPTS_AND_RESUME.md` | `doc/02-training/TRAINING_CONCEPTS_AND_RESUME.md` | 训练指南 |
| `doc/TRAINING_DIRECTORIES.md` | `doc/02-training/TRAINING_DIRECTORIES.md` | 训练指南 |
| `doc/QWEN_MODEL_ADAPTATION.md` | `doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md` | Qwen 适配 |
| `doc/TOKEN_ENCODING_FIX_CN.md` | `doc/04-qwen-adaptation/TOKEN_ENCODING_FIX_CN.md` | Qwen 适配 |

### 从 `tests/` 移动到 `doc/archive/` 的文档

| 旧位置 | 新位置 | 说明 |
|--------|--------|------|
| `tests/CRITICAL_FIX_CN.md` | `doc/archive/CRITICAL_FIX_CN.md` | 修复记录归档 |
| `tests/FIX_SUMMARY_CN.md` | `doc/archive/FIX_SUMMARY_CN.md` | 修复总结归档 |
| `tests/FINAL_SUMMARY_CN.md` | `doc/archive/FINAL_SUMMARY_CN.md` | 最终总结归档 |
| `tests/BEFORE_AFTER_CN.md` | `doc/archive/BEFORE_AFTER_CN.md` | 对比记录归档 |
| `tests/MESSAGES_API_FIX_CN.md` | `doc/archive/MESSAGES_API_FIX_CN.md` | API 修复归档 |
| `tests/VISION_TOKENS_FIX_CN.md` | `doc/archive/VISION_TOKENS_FIX_CN.md` | Token 修复归档 |
| `tests/VERSION_COMPARISON_CN.md` | `doc/archive/VERSION_COMPARISON_CN.md` | 版本对比归档 |
| `tests/LOG_VERSION_README_CN.md` | `doc/archive/LOG_VERSION_README_CN.md` | 日志说明归档 |
| `tests/SUMMARY_CN.md` | `doc/archive/SUMMARY_CN.md` | 总结归档 |
| `tests/EXECUTION_SUMMARY_CN.md` | `doc/archive/EXECUTION_SUMMARY_CN.md` | 执行总结归档 |
| `tests/QUICK_REFERENCE_CN.md` | `doc/archive/QUICK_REFERENCE_CN.md` | 快速参考归档 |
| `tests/INDEX_CN.md` | `doc/archive/INDEX_CN.md` | 旧索引归档 |
| `tests/README_QWEN_TESTS.md` | `doc/archive/README_QWEN_TESTS.md` | 旧测试文档归档 |
| `tests/QWEN_PIXEL_VALUES_FORMAT.md` | `doc/archive/QWEN_PIXEL_VALUES_FORMAT.md` | 格式说明归档 |

---

## 🗂️ 新的目录结构

```
F-LMM/
├── doc/                                    # 📚 所有项目文档
│   ├── README.md                          # 📖 主索引（从这里开始！）
│   ├── MIGRATION_GUIDE.md                 # 📦 本文档
│   │
│   ├── 00-getting-started/                # 🚀 快速开始
│   │   ├── README.md                      # 项目简介和快速安装
│   │   └── todo.md                        # 学习任务清单
│   │
│   ├── 01-architecture/                   # 🏗️ 架构设计
│   │   ├── gykreadme.md                  # 项目详细说明（重要！）
│   │   ├── MODEL_STRUCTURE.md            # 模型结构详解
│   │   └── DATASET_STRUCTURE.md          # 数据集结构说明
│   │
│   ├── 02-training/                       # 🎯 训练指南
│   │   ├── train.md                      # 训练基础
│   │   ├── RUNNER_AND_TRAINING.md        # Runner 详解
│   │   ├── TRAINING_CONCEPTS_AND_RESUME.md # 训练概念
│   │   └── TRAINING_DIRECTORIES.md       # 目录结构
│   │
│   ├── 03-testing/                        # 🧪 测试（链接）
│   │   └── README.md → ../../tests/README.md
│   │
│   ├── 04-qwen-adaptation/                # 🔧 Qwen 适配
│   │   ├── QWEN_MODEL_ADAPTATION.md      # 适配指南
│   │   └── TOKEN_ENCODING_FIX_CN.md      # Token 修复
│   │
│   ├── 05-troubleshooting/                # 🔍 故障排除
│   │   └── README.md                     # 常见问题与解决
│   │
│   └── archive/                           # 📦 历史文档
│       ├── README.md                     # 归档说明
│       └── (14 个历史文档)
│
└── tests/                                 # 🧪 测试代码（现在干净了！）
    ├── README.md                         # 测试文档索引
    ├── test_frozen_qwen.py              # 主测试套件
    ├── diagnose_image_grid_thw.py       # 诊断工具
    ├── verify_data_pipeline.py          # 数据验证
    ├── test_*.py                        # 其他测试
    ├── run_tests.sh                     # 运行脚本
    └── test_results_*.log               # 测试日志
```

---

## 🔍 如何查找文档

### 方法 1：从主索引开始（推荐）

```bash
cat doc/README.md
```

主索引提供：
- 📚 完整的文档分类
- 🎯 按场景查找
- 📊 文档概览
- 🎓 学习路径

### 方法 2：根据需求查找

| 你的需求 | 应该查看 |
|----------|----------|
| 🆕 我是新手 | `doc/00-getting-started/README.md` |
| 🏗️ 理解架构 | `doc/01-architecture/gykreadme.md` |
| 🎯 开始训练 | `doc/02-training/train.md` |
| 🧪 运行测试 | `tests/README.md` |
| 🔧 适配 Qwen | `doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md` |
| 🐛 遇到问题 | `doc/05-troubleshooting/README.md` |
| 📜 查看历史 | `doc/archive/README.md` |

### 方法 3：按分类浏览

```bash
# 查看所有分类
ls doc/

# 查看某个分类的内容
ls doc/01-architecture/

# 阅读某个文档
cat doc/01-architecture/gykreadme.md
```

---

## 📝 重要变更说明

### ✅ 新增内容

1. **主索引**：`doc/README.md` - 统一的导航中心
2. **快速开始**：`doc/00-getting-started/README.md` - 新手指南
3. **故障排除**：`doc/05-troubleshooting/README.md` - 常见问题集
4. **测试索引**：`tests/README.md` - 测试文档重写
5. **归档说明**：`doc/archive/README.md` - 历史文档索引

### 📦 归档内容

14 个历史文档已移至 `doc/archive/`，包括：
- 修复记录（CRITICAL_FIX、FIX_SUMMARY 等）
- 版本对比（VERSION_COMPARISON 等）
- 历史总结（SUMMARY、EXECUTION_SUMMARY 等）

**原因**：
- 这些文档记录了已完成的修复
- 内容已整合到主文档中
- 保留供历史参考

### 🗑️ 删除内容

无。所有文档都保留，只是重新组织。

---

## 🔗 常用文档快速链接

### 📖 必读文档
1. [主索引](./README.md) - 从这里开始
2. [项目详细说明](./01-architecture/gykreadme.md) - 深入理解
3. [训练指南](./02-training/train.md) - 开始训练

### 🧪 测试相关
- [测试索引](../tests/README.md)
- [Qwen 测试](../tests/test_frozen_qwen.py)
- [诊断工具](../tests/diagnose_image_grid_thw.py)

### 🔧 Qwen 相关
- [适配指南](./04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)
- [Token 修复](./04-qwen-adaptation/TOKEN_ENCODING_FIX_CN.md)
- [故障排除](./05-troubleshooting/README.md)

### 📜 历史文档
- [归档索引](./archive/README.md)

---

## 💡 使用建议

### 对于新用户

1. ✅ **先读主索引**：`doc/README.md`
2. ✅ **再读快速开始**：`doc/00-getting-started/README.md`
3. ✅ **根据需求深入**：选择相应的分类目录

### 对于老用户

1. 📌 **书签更新**：更新你的书签到新位置
2. 📌 **习惯调整**：使用主索引作为导航中心
3. 📌 **历史参考**：需要查看历史修复记录时到 `archive/`

### 对于开发者

1. 🔧 **添加文档**：放入正确的分类目录
2. 🔧 **更新索引**：在 `doc/README.md` 中添加链接
3. 🔧 **保持整洁**：测试代码归 `tests/`，文档归 `doc/`

---

## 🎓 学习路径（更新）

### 新手路径
```
doc/README.md
  ↓
doc/00-getting-started/README.md
  ↓
doc/01-architecture/gykreadme.md
  ↓
doc/02-training/train.md
  ↓
tests/README.md (运行测试)
```

### Qwen 适配路径
```
doc/README.md
  ↓
doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md
  ↓
tests/test_frozen_qwen.py (运行测试)
  ↓
doc/05-troubleshooting/README.md (遇到问题时)
  ↓
doc/archive/ (查看历史修复)
```

---

## ❓ 常见问题

### Q: 为什么要重新组织？
A: 原来的结构存在以下问题：
- tests 目录混杂了太多文档
- 文档分类不清晰
- 缺少统一的导航
- 历史文档和当前文档混在一起

### Q: 旧的文档还在吗？
A: 是的！所有文档都保留，只是移到了更合适的位置。

### Q: 我的书签/链接失效了怎么办？
A: 查看本文档的对照表，找到新位置并更新书签。

### Q: 为什么有些文档在 archive？
A: 这些是历史修复记录和已过期的文档，内容已整合到主文档中，保留供参考。

### Q: 测试目录为什么这么少文档了？
A: 测试目录应该只包含测试代码和测试结果，文档已移到 doc/ 目录。

### Q: 我应该从哪里开始？
A: 从 `doc/README.md` 开始，它会引导你到正确的地方。

---

## 📞 反馈

如果你：
- 找不到某个文档
- 觉得分类不合理
- 有改进建议

请提出 Issue 或 PR！

---

**迁移完成日期**：2025-11-09  
**维护者**：AI Assistant


