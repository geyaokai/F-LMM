# Qwen 模型接口测试脚本使用说明

## 概述

`test_qwen_interface.py` 是一个用于测试 Qwen2.5-VL/Qwen3-VL 模型接口的最小推理脚本。该脚本的主要目标是：

1. ✅ 验证 `process_vision_info` 的输出结构
2. ✅ 测试模型的实际调用接口
3. ✅ 检查注意力输出的格式
4. ✅ 记录关键信息以便完善 `FrozenQwen` 实现

## 前置要求

### 1. 安装依赖

**重要**：Qwen2.5-VL 需要从 GitHub 安装最新的 transformers：

```bash
# 创建 Python 3.10 环境（Qwen2.5-VL 需要 Python 3.9+）
conda create -n flmm-qwen-py310 python=3.10 -y
conda activate flmm-qwen-py310

# 安装 PyTorch

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 安装最新 transformers（支持 Qwen2.5-VL）
pip install git+https://github.com/huggingface/transformers

# 安装其他依赖
pip install qwen-vl-utils accelerate "numpy<2"
```

**注意**：如果使用 PyPI 的 transformers，会报错 `KeyError: 'qwen2_5_vl'`，必须从 GitHub 安装。

### 2. 配置 Hugging Face Token（如果需要）

如果模型需要认证，请先登录：

```bash
huggingface-cli login
```

### 3. 确保有足够的 GPU 内存（可选）

模型前向传播测试需要 GPU。如果没有 GPU，脚本会跳过该测试。

## 使用方法

### 基本运行

```bash
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.

python scripts/test_qwen_interface.py
```

### 指定测试图像（可选）

脚本默认会创建一个测试图像，你也可以指定自己的图像：

```python
# 在脚本中修改 test_process_vision_info 调用
vision_info = test_process_vision_info(processor, image_path="path/to/your/image.jpg")
```

## 测试内容

脚本会依次执行以下测试：

### 测试 1: Qwen Processor 基本功能
- 尝试加载不同的 Qwen 模型（Qwen2-VL、Qwen2.5-VL、Qwen3-VL）
- 检查 processor、tokenizer、image_processor 的属性
- 测试特殊 token（`<image>`、`<video>` 等）的 ID

### 测试 2: process_vision_info 输出结构
- 创建测试消息格式
- 调用 `process_vision_info` 处理图像
- 详细分析输出结构，包括：
  - `image_inputs` 的键和值类型
  - Tensor 的形状和数据类型
  - `video_inputs` 的结构

### 测试 3: apply_chat_template
- 测试 `apply_chat_template` 方法
- 生成完整的 prompt
- 分析 tokenized 结果
- 查找图像 token 的位置和数量

### 测试 3: 模型前向传播（需要 GPU）
- 加载模型（使用 `Qwen2_5_VLForConditionalGeneration`）
- 准备输入数据
- 执行前向传播
- 检查输出结构：
  - `hidden_states`: 37 层（包含输入层），形状 `[batch, seq_len, hidden_size]`
  - `attentions`: 36 层，但可能为 None（使用优化的 Attention 实现）
  - `logits`: `[batch, seq_len, vocab_size]`
  - Vision config 详细信息

## 输出结果

### 控制台输出

脚本会在控制台输出详细的测试信息，包括：
- 每个测试步骤的执行状态（✓ 成功 / ✗ 失败）
- 数据结构的详细分析
- 关键字段的类型和形状

### JSON 结果文件

测试完成后，会生成 `qwen_interface_test_results.json` 文件，包含：
- Processor 类型（`Qwen2_5_VLProcessor`）
- 模型名称
- Vision info 的键
- Image inputs 的键和形状
- 注意力层数（36 层，但可能为 None）
- Hidden state 层数（37 层）

## 示例输出

```
================================================================================
Qwen 模型接口测试脚本
================================================================================

================================================================================
测试 1: Qwen Processor 基本功能
================================================================================

尝试加载: Qwen/Qwen2.5-VL-3B-Instruct
✓ 成功加载: Qwen/Qwen2.5-VL-3B-Instruct

Processor 类型: <class 'transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor'>

✓ Processor 包含 tokenizer: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>

✓ Processor 包含 image_processor: <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast'>

  Patch size: 14

测试特殊 token:
  <image>: [27, 1805, 29]
  <video>: [27, 9986, 29]
  <|vision_start|>: [151652]
  <|vision_end|>: [151653]
  <|image_pad|>: [151655]

================================================================================
测试 2: process_vision_info 输出结构
================================================================================

创建测试图像...
测试图像已保存到: /tmp/test_qwen_image.jpg

使用 patch_size: 14

调用 process_vision_info...
✓ process_vision_info 调用成功

输出结构:
image_inputs: dict with keys: ['pixel_values', 'image_grid_thw', 'image_start_index', 'image_placeholder_index']
  pixel_values: Tensor [1, 3, 224, 224] (dtype=torch.float32)
  image_grid_thw: list of length 1
  image_start_index: list of length 1
  image_placeholder_index: list of length 1

...
```

## 常见问题

### 1. 导入错误

如果遇到 `ImportError`，请确保已安装所有依赖：

```bash
pip install transformers qwen-vl-utils
```

### 2. 模型加载失败

**常见错误**：`KeyError: 'qwen2_5_vl'` 或 `ValueError: The checkpoint you are trying to load has model type 'qwen2_5_vl' but Transformers does not recognize this architecture`

**原因**：PyPI 的 transformers 版本不支持 Qwen2.5-VL

**解决方案**：
```bash
# 必须从 GitHub 安装最新版本
pip install git+https://github.com/huggingface/transformers
```

**其他检查**：
- 检查网络连接
- 确认 Hugging Face token 已配置（如果需要）
- 确保 Python 版本 >= 3.9（推荐 3.10）
- 尝试使用较小的模型（如 Qwen2.5-VL-3B）

### 3. GPU 内存不足

脚本会自动检测 GPU 可用性。如果没有 GPU 或内存不足，会跳过模型前向传播测试。

### 4. process_vision_info 找不到

确保已安装 `qwen-vl-utils`：

```bash
pip install qwen-vl-utils
```

### 5. 注意力输出为 None

**现象**：`attentions` 输出为 None

**原因**：transformers 5.0.0 默认使用优化的 Attention 实现（SDPA），不返回注意力权重

**解决方案**：
```python
# 在模型加载时指定 attention 实现
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="eager",  # 使用标准实现，会返回注意力权重
    ...
)
```

**注意**：`eager` 实现较慢但会返回注意力，`sdpa` 或 `flash_attention_2` 更快但不返回注意力。

## 下一步

测试完成后，请：

1. 查看 `qwen_interface_test_results.json` 了解输出结构
2. 根据测试结果更新 `flmm/models/frozen_qwen.py`
3. 完善 `_prepare_inputs()` 和 `_forward()` 方法
4. 根据实际的注意力格式调整注意力提取逻辑

## 相关文件

- `flmm/models/frozen_qwen.py` - FrozenQwen 实现
- `doc/QWEN_MODEL_ADAPTATION.md` - Qwen 适配文档

