# 🚨 关键修复：Qwen2.5-VL pixel_values 格式问题

## 📋 问题描述

训练仍然失败，出现错误：
```
RuntimeError: shape '[4692, -1]' is invalid for input of size 2001920
RuntimeError: shape '[4968, -1]' is invalid for input of size 2119680
```

虽然我们已经让数据管道提取和传递 `image_grid_thw`，但错误仍然发生。

## 🔍 根本原因

**之前的错误理解**：
- 我们认为 Qwen 的 `pixel_values` 是 2D `[H, W]` 格式是"错误的"
- 我们尝试用 `torchvision.transforms` "修复"它为 4D `[B, C, H, W]`
- ❌ 这是完全错误的！

**实际情况**：
- Qwen2.5-VL 的 `pixel_values` **本来就应该是 2D 格式** `[num_patches, hidden_dim]`
- 这是 Qwen 的**正确格式**，不是bug！
- 当我们用 `torchvision.transforms` 重新处理图像时，我们：
  1. 破坏了原始的 patch 提取结果
  2. 导致 `pixel_values` 与 `image_grid_thw` 不匹配
  3. 引发 RuntimeError

## ✅ 正确的修复方案

### 1. 接受 Qwen 的 2D pixel_values 格式

**修改文件**: `flmm/datasets/qwen_image_processor.py`

**关键变更**：
```python
elif pixel_values.dim() == 2:
    # [num_patches, hidden_dim] - Qwen2.5-VL 的特殊格式
    # 这是正确的格式！不要修改！
    # 假设是单张图像（batch_size=1）
    batch_size = 1
    # 保持 pixel_values 不变，它应该是 [num_patches, hidden_dim] 格式
```

**删除的错误代码**：
```python
# ❌ 删除了这段"修复"代码：
try:
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((processed_h, processed_w)),
        transforms.ToTensor(),
    ])
    pixel_values = torch.stack([transform(img) for img in pil_images])
    # ...
except Exception as e2:
    # ...
```

### 2. 正确处理 meta_data

对于 2D `pixel_values`，我们无法从中推断处理后的尺寸，因此使用原始图像尺寸：

```python
if pixel_values.dim() == 2:
    # Qwen 格式：pixel_values 是 [num_patches, hidden_dim]
    # 使用原始图像尺寸
    processed_h = original_height
    processed_w = original_width
    scaled_h = original_height
    scaled_w = original_width
    scale = 1.0
    pad_h = 0
    pad_w = 0
```

### 3. 正确处理 pixel_values_list

对于 2D `pixel_values`，它不能按 batch 索引：

```python
if pixel_values.dim() == 2:
    # Qwen 格式：[num_patches, hidden_dim]，只有一个样本
    pv = pixel_values.cpu().numpy()
    pixel_values_list.append(pv)
else:
    # 标准格式：可以按 batch 索引
    for i in range(batch_size):
        pv = pixel_values[i].cpu().numpy()
        pixel_values_list.append(pv)
```

### 4. 改进 image_grid_thw 的转换逻辑

**修改文件**: `flmm/datasets/transforms.py` 和 `flmm/datasets/png.py`

```python
# Extract image_grid_thw for Qwen models (required for Qwen2.5-VL)
image_grid_thw = image_data.get('image_grid_thw', None)
if image_grid_thw is not None:
    # image_grid_thw 可能是列表或numpy array
    if isinstance(image_grid_thw, list):
        # 列表格式：[array([1, 34, 46])]，取第一个元素
        image_grid_thw = image_grid_thw[0]
    
    # 确保转换为tensor
    if isinstance(image_grid_thw, np.ndarray):
        image_grid_thw = torch.from_numpy(image_grid_thw)
    elif not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.tensor(image_grid_thw)
```

## 📊 修复前后对比

### 修复前 ❌

1. Qwen processor 返回 2D `pixel_values` `[num_patches, hidden_dim]`
2. 我们的代码认为这是"错误的"
3. 用 `torchvision.transforms` 重新处理图像 → 4D `[B, C, H, W]`
4. `image_grid_thw` 仍然基于原始处理结果
5. **不匹配！** → RuntimeError

### 修复后 ✅

1. Qwen processor 返回 2D `pixel_values` `[num_patches, hidden_dim]`
2. 我们**保持原样**，这是正确的格式
3. `image_grid_thw` 与 `pixel_values` 完全匹配
4. **一切正常！** → 训练应该能正常进行

## 🎯 关键要点

1. **不要假设所有模型都使用相同的格式**
   - Qwen2.5-VL 使用 `[num_patches, hidden_dim]` 格式
   - 这是设计如此，不是bug

2. **不要"修复"不是问题的东西**
   - 2D `pixel_values` 是Qwen的正确格式
   - 重新处理只会破坏结果

3. **保持数据管道的一致性**
   - `pixel_values` 和 `image_grid_thw` 必须来自同一次处理
   - 不能独立重新生成其中一个

## 🧪 验证步骤

1. 运行验证脚本（更新后的版本）：
   ```bash
   cd F-LMM/tests
   conda activate flmm-qwen-py310
   python verify_data_pipeline.py
   ```

2. 检查输出：
   - 应该看到 `image_grid_thw` 存在
   - `pixel_values` 可能是 2D 格式（这是正确的！）
   - 不应该看到 "Attempting to fix..." 消息

3. 运行训练测试：
   ```bash
   cd F-LMM
   # 使用你的训练配置
   xtuner train configs/qwen/...
   ```

4. 监控日志：
   - 不应该看到 `RuntimeError: shape '[X, -1]' is invalid`
   - 训练应该正常进行

## 📚 相关文档

- `flmm/datasets/qwen_image_processor.py` - Qwen processor 包装类（已修复）
- `flmm/datasets/transforms.py` - RefCOCO2PNG transform（已修复）
- `flmm/datasets/png.py` - PNGDataset（已修复）
- `tests/QWEN_PIXEL_VALUES_FORMAT.md` - Qwen pixel_values 格式说明

---

**修复时间**: 2025-11-08 05:00+  
**影响文件**: 3个  
**严重性**: 🔥 关键（blocking训练）  
**状态**: ✅ 已修复，等待验证

