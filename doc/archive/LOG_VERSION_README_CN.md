# 日志输出版本测试说明

## 📝 版本说明

这是 Qwen2.5-VL 测试套件的**日志输出版本**，相比原版有以下改进：

### 原版 vs 日志版本

| 特性 | 原版 (test_frozen_qwen.py) | 日志版本 (test_frozen_qwen_with_logging.py) |
|------|---------------------------|----------------------------------------|
| 输出格式 | unittest 标准格式 | 结构化日志输出 |
| 时间戳 | 无 | 每条日志带时间戳 |
| 日志文件 | 无 | 自动保存到文件 |
| 详细程度 | 中等 | 非常详细 |
| 易读性 | 标准 | 更友好（带表情符号） |
| 调试信息 | 简单 | 详细（包含堆栈追踪） |

## 🚀 快速开始

### 方法 1：直接运行 Python 脚本
```bash
cd /home/cvprtemp/gyk/F-LMM/tests
python test_frozen_qwen_with_logging.py
```

### 方法 2：使用 Shell 脚本
```bash
cd /home/cvprtemp/gyk/F-LMM/tests
chmod +x run_tests_with_logging.sh
./run_tests_with_logging.sh
```

## 📊 日志输出示例

### 控制台输出
```
================================================================================
🔧 初始化测试环境
================================================================================
✓ Processor 加载成功
  - Processor 类型: <class 'transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor'>

================================================================================
📝 Test 1: Processor 可用性测试
================================================================================
  - Processor 已加载
  - 类型: Qwen2VLProcessor
✅ 测试通过: Processor 可用

================================================================================
📝 Test 2: 基本图像处理测试
================================================================================
  ✓ 图像处理成功
    - input_ids shape: torch.Size([1, 90])
    - pixel_values shape: torch.Size([256, 1176])
    - image_grid_thw: tensor([[ 1, 16, 16]])
    ✓ image_grid_thw 存在且不为 None
✅ 测试通过: 基本图像处理正常

...

================================================================================
📊 测试总结
================================================================================
  总测试数: 6
  ✅ 通过: 6
  ❌ 失败: 0

  🎉 所有测试通过！
================================================================================
详细日志已保存到: /path/to/test_results_20251108_123456.log
================================================================================
```

### 日志文件内容
日志文件会保存所有输出，格式为：
```
10:30:45 - INFO - ================================================================================
10:30:45 - INFO - 🔧 初始化测试环境
10:30:45 - INFO - ================================================================================
10:30:47 - INFO - ✓ Processor 加载成功
10:30:47 - INFO -   - Processor 类型: <class 'transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor'>
...
```

## 📂 文件说明

### 主要文件
```
tests/
├── test_frozen_qwen.py              # 原版（unittest 格式）
├── test_frozen_qwen_with_logging.py # 日志版本（新）
├── run_tests.sh                      # 原版运行脚本
├── run_tests_with_logging.sh        # 日志版本运行脚本（新）
└── test_results_YYYYMMDD_HHMMSS.log # 自动生成的日志文件
```

### 自动生成的日志文件
- **命名格式**：`test_results_YYYYMMDD_HHMMSS.log`
- **位置**：`tests/` 目录下
- **内容**：完整的测试输出，包括所有日志信息
- **编码**：UTF-8

## 🎨 日志特性

### 1. 结构化输出
- 使用分隔线（`====`）清晰划分各个部分
- 使用表情符号增强可读性
- 统一的缩进格式

### 2. 详细信息
每个测试都包含：
- 测试编号和名称
- 详细的执行步骤
- 数据形状和类型
- 通过/失败状态
- 错误堆栈追踪（如果失败）

### 3. 时间戳
每条日志都带有精确到秒的时间戳，方便追踪执行时间。

### 4. 多级日志
- `INFO`: 正常信息
- `WARNING`: 警告信息（非致命）
- `ERROR`: 错误信息（测试失败）

## 🔍 测试内容

### Test 1: Processor 可用性
- 验证 Processor 正确加载
- 输出 Processor 类型信息

### Test 2: 基本图像处理
- 测试 224×224 图像
- 验证 input_ids, pixel_values
- 检查 image_grid_thw 存在性

### Test 3: 动态分辨率
- 测试 4 种不同分辨率
- 验证每种分辨率的 grid_thw
- 检查格式正确性

### Test 4: data_sample 结构
- 验证所有必需字段
- 检查数据类型和维度
- 验证 Qwen 特有字段

### Test 5: 视觉 Token
- 验证 3 个视觉 token ID
- `<|vision_start|>`: 151652
- `<|vision_end|>`: 151653
- `<|image_pad|>`: 151655

### Test 6: grid_thw 计算
- 测试 4 种分辨率的计算
- 验证计算公式
- 对比预期和实际值

## 📋 输出解释

### 测试状态符号
- ✅ `测试通过`: 测试成功
- ❌ `测试失败`: 测试失败
- ⚠️  `警告`: 非致命问题
- ✓ `成功`: 某个步骤成功
- ✗ `失败`: 某个步骤失败

### 表情符号含义
- 🔧 初始化
- 📝 测试开始
- ✅ 测试通过
- ❌ 测试失败
- ⚠️  警告
- 📊 统计信息
- 🎉 全部通过
- 💾 文件保存
- 📋 列表
- 🔍 检查
- 📐 维度
- 🔤 Token
- 🚀 开始

## 💡 使用建议

### 1. 日常测试
使用日志版本，输出更详细，便于调试：
```bash
python test_frozen_qwen_with_logging.py
```

### 2. CI/CD 集成
使用原版，输出标准，易于解析：
```bash
python test_frozen_qwen.py
```

### 3. 查看历史日志
```bash
# 列出所有日志文件
ls -lt test_results_*.log

# 查看最新日志
cat $(ls -t test_results_*.log | head -1)

# 搜索错误
grep -i "error\|fail" test_results_*.log
```

### 4. 清理旧日志
```bash
# 保留最近 5 个日志
ls -t test_results_*.log | tail -n +6 | xargs rm -f
```

## 🔧 自定义配置

### 修改日志格式
编辑 `test_frozen_qwen_with_logging.py` 的 logging 配置：

```python
logging.basicConfig(
    level=logging.INFO,  # 可改为 DEBUG, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s',  # 自定义格式
    datefmt='%H:%M:%S',  # 时间格式
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(log_filepath, encoding='utf-8')  # 文件输出
    ]
)
```

### 修改日志级别
- `DEBUG`: 显示所有信息（包括调试）
- `INFO`: 正常信息（默认）
- `WARNING`: 仅警告和错误
- `ERROR`: 仅错误

## ❓ 常见问题

### Q1: 日志文件在哪里？
**A**: 在 `tests/` 目录下，文件名格式为 `test_results_YYYYMMDD_HHMMSS.log`

### Q2: 如何只输出到控制台？
**A**: 修改 `handlers` 配置，只保留 `StreamHandler()`

### Q3: 日志文件太多怎么办？
**A**: 定期清理旧日志，或修改代码使用固定文件名（会覆盖）

### Q4: 如何增加更多测试？
**A**: 在 `TestQwenDataSample` 类中添加新的 `test_XX_*` 方法，并在 `run_all_tests` 中调用

## 🎯 与原版对比

### 何时使用原版（unittest）？
- CI/CD 自动化测试
- 需要标准 TAP/JUnit 输出
- 集成到测试框架中

### 何时使用日志版本？
- 手动运行调试
- 需要详细输出
- 保存测试记录
- 演示测试过程

## 📞 获取帮助

如有问题，请参考：
- 原版文档：`README_QWEN_TESTS.md`
- 快速参考：`QUICK_REFERENCE_CN.md`
- 文档索引：`INDEX_CN.md`

---

**创建时间**: 2025-11-08  
**版本**: 1.0  
**状态**: ✅ 可用

