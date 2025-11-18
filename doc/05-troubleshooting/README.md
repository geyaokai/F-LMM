# F-LMM 故障排除指南

> 🔍 **遇到问题？这里可能有答案！**  
> 本文档收集了常见问题和解决方案

---

## 📋 问题分类

### 1️⃣ 训练相关问题
### 2️⃣ 数据加载问题  
### 3️⃣ 模型适配问题
### 4️⃣ 环境配置问题
### 5️⃣ 性能优化问题

---

## 1️⃣ 训练相关问题

### ⚠️ Warning: Image token count 391 != 1530

**问题描述**：
训练日志中出现大量警告：
```
Warning: Image token count 391 != 1530, trying to infer spatial dimensions
Inferred spatial dimensions: 17 x 23 = 391
```

**原因分析**：
- Qwen2.5-VL 使用**动态分辨率**，图像 token 数量根据实际图像大小变化
- 预期 token 数基于配置计算，但实际数量由 Qwen 处理器动态生成
- 代码会自动推断正确的空间维度

**解决方案**：
✅ **这不是错误，是自动修复机制**
- 系统已经正确处理，训练可以正常进行
- 如果想减少警告，可以统一输入图像分辨率

**验证方法**：
```bash
# 查看训练是否正常进行
tail -f work_dirs/your_experiment/logs/*.log | grep "loss:"
```

**相关文件**：
- `flmm/models/frozen_qwen.py` (第 376-386 行)

---

### 🔴 TypeError: 'NoneType' object is not iterable

**问题描述**：
训练时出现错误：
```python
TypeError: 'NoneType' object is not iterable
  in _forward at line 269
```

**原因分析**：
- `image_grid_thw` 参数缺失或为 None
- Qwen2.5-VL 必需此参数来确定图像 token 的空间布局

**解决方案**：
✅ **确保数据集提供 `image_grid_thw`**

修改数据集或在模型中添加自动计算：
```python
if 'image_grid_thw' not in data_sample or data_sample['image_grid_thw'] is None:
    # 从图像尺寸计算
    orig_h, orig_w = image.size  # PIL Image: (width, height)
    grid_h = (orig_h + patch_size - 1) // patch_size
    grid_w = (orig_w + patch_size - 1) // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
```

**验证方法**：
```bash
cd tests
python diagnose_image_grid_thw.py
```

**参考文档**：
- `tests/README.md` - Qwen 测试文档
- `04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`

---

### ⚠️ CUDA out of memory

**问题描述**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**：

**方案 1：减小 batch size**
```python
# 配置文件中
batch_size = 4  # 减小到 4 或更小
```

**方案 2：启用梯度累积**
```python
accumulative_counts = 16  # 增加累积步数
```

**方案 3：使用 DeepSpeed ZeRO-3**
```python
# 配置中
strategy = dict(
    zero_optimization=dict(stage=3)  # 改为 stage 3
)
```

**方案 4：减小图像分辨率**
```python
# 在数据处理器中调整 max_size
processor = dict(
    max_size=448,  # 减小到 448 或更小
)
```

---

### 🟡 Loss 不收敛 / NaN

**可能原因**：
1. 学习率太高
2. 梯度爆炸
3. 数据有问题

**解决方案**：

**检查学习率**：
```python
lr = 1e-4  # 尝试更小的学习率，如 5e-5
```

**启用梯度裁剪**：
```python
max_norm = 1.0  # 确保梯度裁剪已启用
```

**检查数据**：
```python
# 添加调试代码
print(f"Mask range: {masks.min()}, {masks.max()}")
print(f"Has NaN: {torch.isnan(masks).any()}")
```

---

## 2️⃣ 数据加载问题

### 🔴 FileNotFoundError: data/coco/...

**原因**：数据路径不正确

**解决方案**：
```bash
# 检查数据目录结构
ls -R data/coco/

# 应该包含：
# - train2017/
# - annotations/
# - refcoco/, refcoco+/, refcocog/
```

**修改配置中的路径**：
```python
data_root = 'data/coco/'  # 确保正确
```

---

### ⚠️ 数据加载很慢

**解决方案**：

**增加 workers**：
```python
dataloader_num_workers = 8  # 增加到 8
```

**启用 prefetch**：
```python
prefetch_factor = 4  # 预取更多数据
```

**使用 persistent_workers**：
```python
persistent_workers = True  # 避免重复创建
```

---

## 3️⃣ 模型适配问题

### 🔴 pixel_values 维度错误

**问题描述**：
```
Expected 4D tensor [B, C, H, W], got 2D tensor
```

**原因**：Qwen2.5-VL 的 `pixel_values` 是特殊的 2D 格式

**解决方案**：
✅ **不要手动 reshape**
- Qwen 模型内部会自动处理
- 直接传递原始的 2D tensor

**参考文档**：
- `tests/QWEN_PIXEL_VALUES_FORMAT.md`

---

### 🔴 vision tokens 找不到

**问题描述**：
```
Warning: Could not find vision tokens
```

**原因**：token ID 不正确或序列中没有图像 token

**解决方案**：
```python
# 检查 token IDs
print(f"Vision start ID: {tokenizer.convert_tokens_to_ids('<|vision_start|>')}")
print(f"Vision end ID: {tokenizer.convert_tokens_to_ids('<|vision_end|>')}")
print(f"Image pad ID: {tokenizer.convert_tokens_to_ids('<|image_pad|>')}")

# Qwen2.5-VL 应该输出：
# Vision start ID: 151652
# Vision end ID: 151653
# Image pad ID: 151655
```

---

## 4️⃣ 环境配置问题

### 🔴 transformers 版本问题

**问题描述**：
```
AttributeError: 'Qwen2_5_VLForConditionalGeneration' object has no attribute 'xxx'
```

**解决方案**：
```bash
# 确保使用正确的版本
pip install transformers==4.53.1
```

---

### ⚠️ DeepSpeed 初始化失败

**解决方案**：
```bash
# 检查 DeepSpeed 版本
pip install deepspeed==0.12.6

# 确保正确的启动命令
export NPROC_PER_NODE=2
xtuner train config.py --deepspeed deepspeed_zero2
```

---

## 5️⃣ 性能优化问题

### 🐌 训练速度慢

**优化建议**：

**1. 启用 bf16**：
```python
strategy = dict(
    config=dict(
        bf16=dict(enabled=True)
    )
)
```

**2. 增加 batch size**（如果显存允许）

**3. 减少日志频率**：
```python
default_hooks = dict(
    logger=dict(interval=20)  # 增加到 20
)
```

**4. 使用更快的数据后端**（如果使用 Petrel/Ceph）

---

## 🔧 诊断工具

### 运行完整诊断

```bash
cd tests
python diagnose_image_grid_thw.py
```

### 运行单元测试

```bash
cd tests
python test_frozen_qwen.py
```

### 验证数据管道

```bash
cd tests
python verify_data_pipeline.py
```

---

## 📚 相关文档

- **训练问题** → [`../02-training/RUNNER_AND_TRAINING.md`](../02-training/RUNNER_AND_TRAINING.md)
- **数据问题** → [`../01-architecture/DATASET_STRUCTURE.md`](../01-architecture/DATASET_STRUCTURE.md)
- **模型问题** → [`../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md`](../04-qwen-adaptation/QWEN_MODEL_ADAPTATION.md)
- **测试工具** → [`../../tests/README.md`](../../tests/README.md)

---

## 📞 仍然有问题？

1. **查看日志**：
   ```bash
   tail -100 work_dirs/your_experiment/logs/*.log
   ```

2. **启用调试模式**：
   ```python
   log_level = 'DEBUG'
   ```

3. **检查配置**：
   ```bash
   xtuner train config.py --dry-run
   ```

4. **查看堆栈跟踪**：完整错误信息通常包含解决线索

---

## ✨ 贡献

遇到新问题并解决了？欢迎补充到本文档！

**最后更新**：2025-11-09  
**维护者**：AI Assistant


