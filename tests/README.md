# F-LMM 测试套件

> 🧪 **测试和验证工具**  
> 本目录包含用于验证和诊断 F-LMM 模型的测试脚本

---

## 📋 测试文件概览

### 🔬 核心测试脚本

| 文件 | 用途 | 运行时间 |
|------|------|----------|
| [`test_frozen_qwen.py`](./test_frozen_qwen.py) | Qwen 模型完整测试套件（8个测试） | ~2-3 分钟 |
| [`diagnose_image_grid_thw.py`](./diagnose_image_grid_thw.py) | image_grid_thw 问题诊断 | ~30 秒 |
| [`compare_generate_teacher_forced_ground.py`](./compare_generate_teacher_forced_ground.py) | 对比 generate-cache 与 teacher-forced grounding | ~1-2 分钟 |
| [`verify_data_pipeline.py`](./verify_data_pipeline.py) | 数据管道验证 | ~1 分钟 |
| [`test_image_token_encoding.py`](./test_image_token_encoding.py) | 图像 token 编码测试 | ~30 秒 |
| [`test_token_matching.py`](./test_token_matching.py) | Token 匹配测试 | ~20 秒 |

### 🚀 运行脚本

| 文件 | 用途 |
|------|------|
| [`run_tests.sh`](./run_tests.sh) | 运行所有测试（一键执行） |
| [`run_verify.sh`](./run_verify.sh) | 快速验证（核心测试） |
| [`run_tests_with_logging.sh`](./run_tests_with_logging.sh) | 带日志的测试运行 |

### 📝 测试日志

测试结果会自动保存在：
- `test_results_*.log` - 测试运行日志

---

## 🎯 快速开始

### 方法 1：一键运行所有测试

```bash
cd tests
./run_tests.sh
```

### 方法 2：运行特定测试

```bash
# Qwen 模型完整测试
python test_frozen_qwen.py

# 快速诊断
python diagnose_image_grid_thw.py

# 对比在线 grounding 和 teacher-forced grounding
python compare_generate_teacher_forced_ground.py \
  --image data/custom/shampoo_room.png \
  --question "Where is the shampoo?" \
  --phrase shampoo \
  --device cuda:0 \
  --device-map none

# 验证数据管道
python verify_data_pipeline.py
```

### 方法 3：运行单个测试用例

```bash
# 只运行特定的测试
python test_frozen_qwen.py TestQwenModel.test_image_grid_thw_calculation
```

---

## 🧪 测试说明

### 1. Qwen 模型完整测试 (`test_frozen_qwen.py`)

**测试内容**：
- ✅ Processor 加载和配置
- ✅ 模型加载和初始化
- ✅ image_grid_thw 计算正确性
- ✅ 图像预处理流程
- ✅ Token 编码和解码
- ✅ 前向传播
- ✅ 注意力提取
- ✅ 完整推理流程

**运行**：
```bash
python test_frozen_qwen.py
```

**预期输出**：
```
test_attention_extraction (__main__.TestQwenModel) ... ok
test_forward_pass (__main__.TestQwenModel) ... ok
test_image_grid_thw_calculation (__main__.TestQwenModel) ... ok
test_image_preprocessing (__main__.TestQwenModel) ... ok
test_inference_end_to_end (__main__.TestQwenModel) ... ok
test_model_loading (__main__.TestQwenModel) ... ok
test_processor_loading (__main__.TestQwenModel) ... ok
test_token_encoding (__main__.TestQwenModel) ... ok

Ran 8 tests in XXs
OK
```

---

### 2. image_grid_thw 诊断 (`diagnose_image_grid_thw.py`)

**用途**：专门诊断 `image_grid_thw` 相关问题

**测试场景**：
1. 标准尺寸图像 (224x224)
2. 非方形图像 (640x480)  
3. 不同 patch size (14, 16)

**运行**：
```bash
python diagnose_image_grid_thw.py
```

**预期输出**：
```
=== Test 1: Standard Image ===
✓ Image grid thw: [1, 16, 16]
✓ Token count: 256

=== Test 2: Non-square Image ===
✓ Image grid thw: [1, 46, 34]
✓ Token count: 1564

✓ All tests passed!
```

---

### 3. 数据管道验证 (`verify_data_pipeline.py`)

**用途**：验证数据加载和预处理流程

**测试内容**：
- 数据集加载
- 图像预处理
- Token 化
- Batch 构建

**运行**：
```bash
python verify_data_pipeline.py
```

---

### 4. generate-cache / teacher-forced grounding 对比 (`compare_generate_teacher_forced_ground.py`)

**用途**：定位在线 demo grounding 变差到底是：

- `answer()` 返回的生成态 attention cache 本身更散
- 还是后续 phrase span / mask head / SAM / bbox 后处理出了问题

**它会做什么**：

1. 先正常跑一遍 `model.answer(...)`
2. 选定一个 answer phrase
3. 用当前在线 cache 做一次 grounding
4. 再把同一个 answer 作为 teacher-forced assistant 文本喂回模型做 full forward
5. 对同一个 phrase 再做一次 grounding
6. 把两条路径的 heatmap / mask / overlay / bbox / IoU 一起落盘

**运行**：

```bash
python compare_generate_teacher_forced_ground.py \
  --config ../configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint ../checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --image ../data/custom/shampoo_room.png \
  --question "Where is the shampoo?" \
  --phrase shampoo \
  --device cuda:0 \
  --device-map none
```

**主要输出**：

- `generate/heat_overlay.png`
- `teacher_forced/heat_overlay.png`
- `generate/final_overlay.png`
- `teacher_forced/final_overlay.png`
- `comparison_panel.png`
- `report.json`

---

### 4. 图像 Token 编码测试 (`test_image_token_encoding.py`)

**用途**：测试图像 token 的编码和定位

**运行**：
```bash
python test_image_token_encoding.py
```

---

### 5. Token 匹配测试 (`test_token_matching.py`)

**用途**：测试文本 token 与 mask 的匹配

**运行**：
```bash
python test_token_matching.py
```

---

## 🐛 常见测试问题

### 问题 1：测试失败 - 模型下载

**错误**：
```
OSError: Can't load model from 'Qwen/Qwen2.5-VL-3B-Instruct'
```

**解决**：
```bash
# 方法 1：手动下载
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct

# 方法 2：设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 问题 2：CUDA out of memory

**解决**：
```bash
# 在 CPU 上运行测试
export CUDA_VISIBLE_DEVICES=""
python test_frozen_qwen.py
```

---

### 问题 3：测试超时

**解决**：
```bash
# 增加超时时间
timeout 600 python test_frozen_qwen.py
```

---

## 📊 测试覆盖范围

### Qwen 模型测试

- ✅ Qwen2.5-VL-3B-Instruct
- ✅ 图像预处理
- ✅ Token 编码
- ✅ 前向传播
- ✅ 注意力提取
- ✅ image_grid_thw 计算

### 数据管道测试

- ✅ PNG 数据集
- ✅ RefCOCO 数据集
- ✅ 图像加载
- ✅ Mask 处理
- ✅ Batch 拼接

---

## 📚 参考文档

### 测试相关问题
- **Qwen 适配**：[`../doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)
- **故障排除**：[`../doc/05-troubleshooting/README.md`](../doc/05-troubleshooting/README.md)

### 历史修复记录
测试过程中的修复记录已归档到：[`../doc/archive/`](../doc/archive/)

包含：
- `CRITICAL_FIX_CN.md` - 关键修复记录
- `BEFORE_AFTER_CN.md` - 修复前后对比
- `FIX_SUMMARY_CN.md` - 修复总结
- 等等...

---

## 🔧 开发者指南

### 添加新测试

1. **创建测试文件**：
   ```bash
   touch test_your_feature.py
   ```

2. **使用 unittest 框架**：
   ```python
   import unittest
   
   class TestYourFeature(unittest.TestCase):
       def test_something(self):
           # Your test code
           self.assertTrue(result)
   
   if __name__ == '__main__':
       unittest.main()
   ```

3. **添加到运行脚本**：
   ```bash
   # 编辑 run_tests.sh
   python test_your_feature.py
   ```

### 测试最佳实践

- ✅ 每个测试独立运行
- ✅ 使用有意义的测试名称
- ✅ 添加详细的 docstring
- ✅ 测试失败时输出有用信息
- ✅ 清理测试产生的临时文件

---

## 📈 测试统计

### 当前测试状态

| 模块 | 测试数量 | 通过率 | 最后更新 |
|------|----------|--------|----------|
| Qwen 模型 | 8 | 100% | 2025-11-08 |
| 数据管道 | 6 | 100% | 2025-11-08 |
| Token 编码 | 4 | 100% | 2025-11-08 |

---

## 📞 获取帮助

遇到测试问题？

1. **查看测试日志**：`test_results_*.log`
2. **运行诊断工具**：`python diagnose_image_grid_thw.py`
3. **查看故障排除文档**：[`../doc/05-troubleshooting/README.md`](../doc/05-troubleshooting/README.md)
4. **查看 Qwen 适配文档**：[`../doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../doc/04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)

---

## ✨ 贡献

欢迎贡献新的测试用例！请确保：
- 测试可重复运行
- 有清晰的文档说明
- 通过所有现有测试

---

**最后更新**：2025-11-09  
**维护者**：AI Assistant
