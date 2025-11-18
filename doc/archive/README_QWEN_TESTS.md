# FrozenQwen 单元测试和诊断文档

## 问题背景

根据训练日志 `frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png_20251108_033945.log`，训练时出现以下错误：

```
TypeError: 'NoneType' object is not iterable
File "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1757, in forward
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
```

**根本原因**：`image_grid_thw` 参数为 `None`，导致 Qwen2.5-VL 模型在处理视觉输入时失败。

## 文件说明

### 1. `test_frozen_qwen.py`
完整的单元测试套件，包含：

- **TestQwenDataSample**：测试数据样本结构
  - `test_01_processor_available`：验证 processor 可用性
  - `test_02_basic_image_processing`：测试基本图像处理
  - `test_03_dynamic_resolution`：测试 Qwen 的动态分辨率特性
  - `test_04_data_sample_structure`：验证完整的 data_sample 结构
  - `test_05_vision_tokens`：验证视觉 token ID
  - `test_06_image_grid_thw_calculation`：测试 grid_thw 计算逻辑

- **TestQwenModelIntegration**：测试模型集成
  - `test_01_model_import`：测试模型导入
  - `test_02_prepare_inputs_logic`：测试 _prepare_inputs 方法

### 2. `diagnose_image_grid_thw.py`
快速诊断脚本，用于：

- 检查 processor 输出中是否包含 `image_grid_thw`
- 验证不同输入格式的处理结果
- 提供详细的修复建议

### 3. `run_tests.sh`
一键运行所有测试的脚本

## 运行测试

### 方法 1：运行完整测试套件

```bash
cd /home/cvprtemp/gyk/F-LMM/tests
python test_frozen_qwen.py
```

### 方法 2：仅运行快速诊断

```bash
cd /home/cvprtemp/gyk/F-LMM/tests
python diagnose_image_grid_thw.py
```

### 方法 3：使用 shell 脚本

```bash
cd /home/cvprtemp/gyk/F-LMM/tests
chmod +x run_tests.sh
./run_tests.sh
```

## 已实施的修复

### 修复位置：`flmm/models/frozen_qwen.py`

在 `FrozenQwenSAM._forward()` 方法中添加了 `image_grid_thw` 的后备计算逻辑：

```python
# 添加 image_grid_thw（Qwen2.5-VL 必需）
if 'image_grid_thw' in data_sample and data_sample['image_grid_thw'] is not None:
    model_kwargs['image_grid_thw'] = data_sample['image_grid_thw'].to(self.qwen_model.device)
else:
    # 如果 image_grid_thw 缺失，手动计算
    print_log("Warning: image_grid_thw is missing, calculating from pixel_values")
    pixel_values = model_kwargs['pixel_values']
    
    # pixel_values 可能是 [1, C, H, W] 或 [C, H, W]
    if pixel_values.dim() == 4:
        _, _, h, w = pixel_values.shape
    elif pixel_values.dim() == 3:
        _, h, w = pixel_values.shape
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
    
    # 计算 patch grid 尺寸
    grid_h = (h + self.patch_size - 1) // self.patch_size
    grid_w = (w + self.patch_size - 1) // self.patch_size
    
    # 构建 image_grid_thw: [batch, 3] 格式为 [temporal, height_grids, width_grids]
    image_grid_thw = torch.tensor(
        [[1, grid_h, grid_w]], 
        dtype=torch.long,
        device=self.qwen_model.device
    )
    model_kwargs['image_grid_thw'] = image_grid_thw
    print_log(f"Calculated image_grid_thw: {image_grid_thw} (image size: {h}x{w}, patch_size: {self.patch_size})")
```

### 修复原理

1. **检查字段**：首先检查 `data_sample` 中是否包含 `image_grid_thw` 且不为 `None`
2. **动态计算**：如果缺失，从 `pixel_values` 的尺寸动态计算
3. **考虑动态分辨率**：Qwen2.5-VL 支持动态分辨率，计算公式为：
   - `grid_h = (height + patch_size - 1) // patch_size`
   - `grid_w = (width + patch_size - 1) // patch_size`
4. **正确格式**：构建 `[1, 3]` 形状的 tensor，格式为 `[temporal, height_grids, width_grids]`

## Qwen 与 DeepSeek-VL 的关键差异

| 特性 | DeepSeek-VL | Qwen2.5-VL |
|------|-------------|------------|
| **分辨率** | 固定 384×384 | 动态（原生分辨率） |
| **Patch 数量** | 固定 576 (24×24) | 动态变化 |
| **视觉 Token** | `<image_placeholder>` | `<\|vision_start\|>`, `<\|vision_end\|>`, `<\|image_pad\|>` |
| **Patch Size** | 16 | 14 |
| **Grid 信息** | 不需要 | 需要 `image_grid_thw` |
| **图像预处理** | 固定 resize | 保持宽高比 padding |

## 测试覆盖的场景

### 1. 不同分辨率的图像
- 正方形图像 (224×224)
- 宽矩形图像 (448×224)
- 高矩形图像 (224×448)
- 标准分辨率 (640×480)
- 大分辨率图像 (1024×768)

### 2. 数据结构验证
- `input_ids`：Token 序列
- `pixel_values`：预处理后的图像张量
- `image_grid_thw`：Grid 尺寸信息（关键！）
- `attention_mask`：注意力掩码
- `mask_ids`：每个 token 对应的 mask ID
- `masks`：Ground truth masks
- `meta_data`：图像元数据

### 3. 视觉 Token 识别
- `<|vision_start|>`: 151652
- `<|vision_end|>`: 151653
- `<|image_pad|>`: 151655

## 常见问题和解决方案

### Q1: 为什么会出现 `image_grid_thw is None` 错误？

**原因**：
1. Dataset 没有正确处理 processor 的输出
2. Processor 版本过旧，不返回 `image_grid_thw`
3. 自定义的 image processor wrapper 没有传递该字段

**解决方案**：
- 使用本修复（在模型中添加后备计算）
- 或在 Dataset 中显式添加该字段
- 或升级 transformers 到最新版本

### Q2: 如何确认修复是否生效？

运行测试脚本：
```bash
python tests/diagnose_image_grid_thw.py
```

查看是否输出：
```
✓ image_grid_thw 存在: tensor([[1, 34, 46]])
```

### Q3: 动态分辨率如何工作？

Qwen2.5-VL 不会将图像 resize 到固定大小，而是：
1. 保持图像宽高比
2. 在必要时添加 padding
3. 将图像分割为 14×14 的 patch
4. Grid 尺寸 = (图像高度/14) × (图像宽度/14)

例如：
- 640×480 图像 → 46×34 grid (46×14=644, 34×14=476)
- 224×224 图像 → 16×16 grid

### Q4: 如何在 Dataset 中添加 `image_grid_thw`？

在 `flmm/datasets/transforms.py` 或相关文件中：

```python
def calculate_image_grid_thw(pixel_values, patch_size=14):
    """计算 image_grid_thw"""
    if pixel_values.dim() == 4:
        _, _, h, w = pixel_values.shape
    else:
        _, h, w = pixel_values.shape
    
    grid_h = (h + patch_size - 1) // patch_size
    grid_w = (w + patch_size - 1) // patch_size
    
    return torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

# 在 __getitem__ 中：
if 'image_grid_thw' not in data_sample:
    data_sample['image_grid_thw'] = calculate_image_grid_thw(
        data_sample['pixel_values']
    )
```

## 预期测试输出

### 成功的诊断输出示例：

```
======================================================================
诊断 Qwen Processor 输出
======================================================================
✓ Processor 加载成功

测试 1: 使用 messages 格式
----------------------------------------------------------------------
输出 keys: ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
  - input_ids: shape=torch.Size([1, 37]), dtype=torch.int64
  - attention_mask: shape=torch.Size([1, 37]), dtype=torch.int64
  - pixel_values: shape=torch.Size([1, 3, 480, 644]), dtype=torch.float32
  - image_grid_thw: shape=torch.Size([1, 3]), value=tensor([[1, 34, 46]])

✓ image_grid_thw 存在！
  值: tensor([[1, 34, 46]])
  形状: torch.Size([1, 3])
```

### 成功的单元测试输出示例：

```
test_01_processor_available (__main__.TestQwenDataSample) ... ✓ Test 1: Processor 可用
ok
test_02_basic_image_processing (__main__.TestQwenDataSample) ... ✓ Test 2: 图像处理成功
  - input_ids shape: torch.Size([1, 35])
  - pixel_values shape: torch.Size([1, 3, 224, 224])
  - image_grid_thw: tensor([[1, 16, 16]])
ok
test_03_dynamic_resolution (__main__.TestQwenDataSample) ... ✓ Test 3: 动态分辨率测试
  - 正方形 (224x224):
    pixel_values: torch.Size([1, 3, 224, 224])
    image_grid_thw: tensor([[1, 16, 16]])
  - 宽矩形 (448x224):
    pixel_values: torch.Size([1, 3, 224, 448])
    image_grid_thw: tensor([[1, 16, 32]])
  ...
ok
```

## 下一步行动

1. **运行诊断**：`python tests/diagnose_image_grid_thw.py`
2. **运行单元测试**：`python tests/test_frozen_qwen.py`
3. **重新训练**：使用修复后的代码重新运行训练
4. **监控日志**：检查是否还有 `image_grid_thw is None` 错误

## 参考资源

- Qwen2.5-VL 文档：https://qwen.readthedocs.io/
- Transformers 文档：https://huggingface.co/docs/transformers/
- DeepSeek-VL 参考实现：`flmm/models/frozen_deepseek_vl.py`

