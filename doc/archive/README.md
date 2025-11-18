# 历史文档归档

> 📦 **本目录包含历史修复记录和已过期的文档**  
> 这些文档保留作为参考，但可能已经过时

---

## 📋 归档内容说明

### 🔧 Qwen 模型修复记录

这些文档记录了 Qwen2.5-VL 模型适配过程中的问题修复：

| 文档 | 内容 | 归档原因 |
|------|------|----------|
| `CRITICAL_FIX_CN.md` | 关键修复记录 | 问题已修复，保留供参考 |
| `FIX_SUMMARY_CN.md` | 修复总结 | 问题已修复，保留供参考 |
| `FINAL_SUMMARY_CN.md` | 最终总结 | 问题已修复，保留供参考 |
| `BEFORE_AFTER_CN.md` | 修复前后对比 | 问题已修复，保留供参考 |

### 📚 版本对比文档

| 文档 | 内容 | 归档原因 |
|------|------|----------|
| `VERSION_COMPARISON_CN.md` | 版本对比 | 临时文档，已整合到主文档 |
| `LOG_VERSION_README_CN.md` | 日志版本说明 | 临时文档，已整合到主文档 |

### 📖 总结性文档

这些文档是测试和修复过程的总结，现已被更系统的文档取代：

| 文档 | 内容 | 替代文档 |
|------|------|----------|
| `SUMMARY_CN.md` | 中文总结 | [`../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md) |
| `EXECUTION_SUMMARY_CN.md` | 执行总结 | [`../../tests/README.md`](../../tests/README.md) |
| `QUICK_REFERENCE_CN.md` | 快速参考 | [`../05-troubleshooting/README.md`](../05-troubleshooting/README.md) |
| `INDEX_CN.md` | 索引文档 | [`../README.md`](../README.md) |
| `README_QWEN_TESTS.md` | Qwen 测试说明 | [`../../tests/README.md`](../../tests/README.md) |

### 🔬 技术细节文档

| 文档 | 内容 | 状态 |
|------|------|------|
| `QWEN_PIXEL_VALUES_FORMAT.md` | pixel_values 格式说明 | 仍然有效，但已整合到适配文档 |
| `MESSAGES_API_FIX_CN.md` | Messages API 修复 | 问题已修复 |
| `VISION_TOKENS_FIX_CN.md` | Vision tokens 修复 | 问题已修复 |

---

## 🎯 如何使用归档文档

### ✅ 适合查看的场景

1. **研究历史问题**：了解之前遇到的问题和解决过程
2. **深入技术细节**：某些归档文档包含详细的技术分析
3. **对比修复前后**：了解代码演进过程
4. **学习调试过程**：看看问题是如何定位和解决的

### ⚠️ 注意事项

- **可能已过时**：这些文档的内容可能不适用于当前版本
- **已有替代**：大部分内容已整合到主文档中
- **仅供参考**：不建议作为主要学习材料

---

## 📚 推荐阅读

### 如果你想了解 Qwen 适配
👉 请阅读：[`../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)

### 如果你想运行测试
👉 请阅读：[`../../tests/README.md`](../../tests/README.md)

### 如果你遇到问题
👉 请阅读：[`../05-troubleshooting/README.md`](../05-troubleshooting/README.md)

### 如果你想了解项目整体
👉 请阅读：[`../README.md`](../README.md)

---

## 🔍 归档文档索引

### 修复记录类
- `CRITICAL_FIX_CN.md` - 关键修复
- `FIX_SUMMARY_CN.md` - 修复总结  
- `FINAL_SUMMARY_CN.md` - 最终总结
- `BEFORE_AFTER_CN.md` - 修复对比
- `MESSAGES_API_FIX_CN.md` - API 修复
- `VISION_TOKENS_FIX_CN.md` - Token 修复

### 版本对比类
- `VERSION_COMPARISON_CN.md` - 版本对比
- `LOG_VERSION_README_CN.md` - 日志版本说明

### 总结索引类
- `SUMMARY_CN.md` - 中文总结
- `EXECUTION_SUMMARY_CN.md` - 执行总结
- `QUICK_REFERENCE_CN.md` - 快速参考
- `INDEX_CN.md` - 原索引文档
- `README_QWEN_TESTS.md` - 测试说明

### 技术细节类
- `QWEN_PIXEL_VALUES_FORMAT.md` - Pixel values 格式

---

## 📈 文档演进历史

### 2025-11-08
- 创建了大量修复和测试文档
- 记录了 Qwen2.5-VL 适配过程
- 完成了 image_grid_thw 问题修复

### 2025-11-09  
- **重新组织文档结构**
- 将修复记录归档
- 创建系统化的文档索引
- 整合分散的信息到主文档

---

## 💡 为什么归档？

1. **保持清晰**：测试目录应该只包含测试代码
2. **避免混淆**：避免过时文档误导新用户
3. **保留历史**：记录问题解决过程，供参考
4. **便于维护**：集中管理文档，易于更新

---

## ✨ 贡献

如果发现归档文档中的信息仍然有用但缺失在主文档中，请：

1. 提取有价值的内容
2. 更新到相应的主文档
3. 提交 Pull Request

---

**归档日期**：2025-11-09  
**维护者**：AI Assistant


