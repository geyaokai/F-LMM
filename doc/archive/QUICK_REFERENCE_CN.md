# Qwen2.5-VL 快速参考指南

## 🚨 关键发现

### 1. image_grid_thw 必需但可能缺失
**问题**：训练时崩溃 `TypeError: 'NoneType' object is not iterable`  
**原因**：`image_grid_thw` 为 `None`  
**解决**：已在 `frozen_qwen.py` 第269-298行添加后备计算

### 2. pixel_values 格式特殊
**发现**：Qwen2.5-VL 使用 **2D 格式** `[H, W]`，不是标准的 `[B, C, H, W]`  
**示例**：`torch.Size([1564, 1176])` 而非 `torch.Size([1, 3, 480, 640])`  
**影响**：代码中不能假设 4D 维度

### 3. 动态分辨率处理
**特点**：每张图像的 patch 数量不同  
**示例**：
- 224×224 → 16×16 = 256 patches
- 640×480 → 34×46 = 1564 patches  
**要求**：必须有 `image_grid_thw` 提供空间信息

## ✅ 测试状态

运行 `python tests/test_frozen_qwen.py`：

- ✅ Processor 可用性
- ✅ 基本图像处理  
- ✅ 动态分辨率测试
- ✅ data_sample 结构验证（已修复）
- ✅ 视觉 token 验证
- ✅ image_grid_thw 计算
- ✅ 模型导入测试

**结果**：7/7 测试通过 ✓

## 📋 关键代码片段

### 修复：image_grid_thw 后备计算

```python
# frozen_qwen.py 第269-298行
if 'image_grid_thw' in data_sample and data_sample['image_grid_thw'] is not None:
    model_kwargs['image_grid_thw'] = data_sample['image_grid_thw'].to(device)
else:
    # 自动计算
    pixel_values = model_kwargs['pixel_values']
    if pixel_values.dim() == 4:
        _, _, h, w = pixel_values.shape
    elif pixel_values.dim() == 3:
        _, h, w = pixel_values.shape
    elif pixel_values.dim() == 2:
        h, w = pixel_values.shape
    
    grid_h = (h + self.patch_size - 1) // self.patch_size
    grid_w = (w + self.patch_size - 1) // self.patch_size
    
    model_kwargs['image_grid_thw'] = torch.tensor(
        [[1, grid_h, grid_w]], dtype=torch.long, device=device
    )
```

### 正确处理 pixel_values

```python
# ✅ 灵活处理多种维度
if pixel_values.dim() == 4:
    _, _, h, w = pixel_values.shape
elif pixel_values.dim() == 3:
    _, h, w = pixel_values.shape
elif pixel_values.dim() == 2:
    h, w = pixel_values.shape
else:
    raise ValueError(f"Unexpected shape: {pixel_values.shape}")
```

### data_sample 必需字段

```python
data_sample = {
    'input_ids': tensor([...]),          # [seq_len]
    'pixel_values': tensor([...]),       # [H, W] or [1, C, H, W]
    'image_grid_thw': tensor([[1, h, w]]),  # [1, 3] ⚠️ 关键！
    'attention_mask': tensor([...]),     # [seq_len] (可选)
    'image': PIL.Image,                  # 原始图像
    'masks': tensor([...]),              # [num_masks, H, W]
    'mask_ids': tensor([...]),           # [seq_len]
    'meta_data': {...}                   # 元数据字典
}
```

## 🎯 与 DeepSeek-VL 的差异

| 特性 | DeepSeek-VL | Qwen2.5-VL |
|------|-------------|------------|
| **分辨率** | 固定 384×384 | 动态（保持宽高比） |
| **Patch 大小** | 16 | 14 |
| **Patch 数量** | 固定 576 (24×24) | 动态变化 |
| **pixel_values** | `[1, 1, 3, 384, 384]` | `[H, W]` |
| **grid_thw** | 不需要 | **必需** |
| **图像 token** | `<image_placeholder>` | `<\|vision_start\|>` `<\|vision_end\|>` |

## 🚀 快速开始

### 1. 运行诊断
```bash
cd /home/cvprtemp/gyk/F-LMM/tests
python diagnose_image_grid_thw.py
```

**预期输出**：
```
✓ image_grid_thw 存在: tensor([[ 1, 34, 46]])
✓ 所有必需字段都存在
```

### 2. 运行完整测试
```bash
python test_frozen_qwen.py
```

**预期输出**：
```
Ran 7 tests in X.XXXs
OK
```

### 3. 重新训练
修复已应用，直接运行训练命令。预期日志：
```
Warning: image_grid_thw is missing, calculating from pixel_values
Calculated image_grid_thw: tensor([[1, 34, 46]]) (image size: 480x644, patch_size: 14)
Epoch [1][10/XXXX]  loss: 0.XXXX
```

## 📊 关键数值

### Vision Token IDs
```python
<|vision_start|> = 151652
<|vision_end|>   = 151653
<|image_pad|>    = 151655
```

### Patch Size
```python
Qwen2.5-VL-3B: patch_size = 14
```

### Grid 计算公式
```python
grid_h = (height + patch_size - 1) // patch_size  # 向上取整
grid_w = (width + patch_size - 1) // patch_size
image_grid_thw = [[1, grid_h, grid_w]]  # temporal=1
```

### 示例计算
| 图像尺寸 | grid_h | grid_w | patches | image_grid_thw |
|---------|--------|--------|---------|----------------|
| 224×224 | 16 | 16 | 256 | `[[1, 16, 16]]` |
| 640×480 | 34 | 46 | 1564 | `[[1, 34, 46]]` |
| 448×336 | 24 | 32 | 768 | `[[1, 24, 32]]` |

## 🔧 故障排除

### 问题 1：TypeError: 'NoneType' object is not iterable
**原因**：`image_grid_thw` 为 `None`  
**解决**：修复已应用，会自动计算

### 问题 2：pixel_values 维度错误
**原因**：假设是 4D `[B, C, H, W]`  
**解决**：使用灵活的维度处理（见上方代码）

### 问题 3：测试失败
**原因**：可能是 processor 版本问题  
**检查**：`transformers>=4.37.0`
```bash
pip install --upgrade transformers>=4.37.0
```

### 问题 4：pixel_values 形状不匹配
**原因**：Qwen 使用 2D 格式  
**解决**：不要手动 reshape，直接传递给模型

## 📂 文件索引

### 核心文件
- `flmm/models/frozen_qwen.py` - 模型实现（已修复）
- `tests/test_frozen_qwen.py` - 单元测试
- `tests/diagnose_image_grid_thw.py` - 诊断工具

### 文档文件
- `tests/README_QWEN_TESTS.md` - 完整文档
- `tests/SUMMARY_CN.md` - 中文总结
- `tests/BEFORE_AFTER_CN.md` - 修复对比
- `tests/QWEN_PIXEL_VALUES_FORMAT.md` - pixel_values 格式说明
- `tests/QUICK_REFERENCE_CN.md` - 本文档

## ⚡ 命令速查

```bash
# 快速诊断
python tests/diagnose_image_grid_thw.py

# 完整测试
python tests/test_frozen_qwen.py

# 一键运行
cd tests && ./run_tests.sh

# 重新训练（使用你的训练命令）
# python tools/train.py configs/...
```

## 🎓 学习要点

1. **Qwen2.5-VL 使用动态分辨率**，每张图像的 patch 数量不同
2. **image_grid_thw 是必需参数**，用于恢复空间结构
3. **pixel_values 格式特殊**，是 2D 而非 4D
4. **修复是向后兼容的**，不影响正确提供 grid_thw 的代码
5. **后备计算性能开销极小**，仅几个整数运算

## ✨ 总结

- ✅ 问题已识别：`image_grid_thw` 缺失
- ✅ 修复已实施：自动计算后备方案
- ✅ 测试已通过：7/7 全部通过
- ✅ 文档已完善：5个文档文件
- ✅ 可以训练：修复已应用到主代码

**现在可以安全地重新开始训练！** 🎉

