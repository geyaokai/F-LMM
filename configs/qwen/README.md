# Qwen 模型训练配置

本目录包含使用 xtuner 训练 Qwen2.5-VL 模型的配置文件。

## 配置文件说明

### frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py

基于 Qwen2.5-VL-3B-Instruct 的冻结视觉-语言模型训练配置，用于 referring segmentation 任务。

**模型架构：**
- 冻结的 Qwen2.5-VL-3B-Instruct 作为特征提取器
- UNet 作为 mask decoder
- SAM (Segment Anything Model) ViT-L 用于 mask 细化

**关键参数：**
- Batch size: 8 (per device)
- 梯度累积: 8 steps
- 有效 batch size: 8 × 8 × num_gpus = 64 × num_gpus
- 学习率: 1e-4
- 训练 epochs: 8
- 优化器: AdamW
- 混合精度: bfloat16

## 前置准备

### 1. 环境配置

安装 Qwen 相关依赖（需要 Python 3.10+）：

```bash
# 创建环境
conda create -n flmm-qwen-py310 python=3.10 -y
conda activate flmm-qwen-py310

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# 安装最新 transformers（支持 Qwen2.5-VL）
pip install git+https://github.com/huggingface/transformers

# 安装其他依赖
pip install qwen-vl-utils accelerate "numpy<2"

# 安装项目依赖
pip install -e .
# 或者
pip install -r requirements.txt
```

**重要：** 必须从 GitHub 安装最新的 transformers，PyPI 版本不支持 Qwen2.5-VL。

### 2. 数据准备

下载并准备以下数据集：

```bash
# COCO 数据集
data/coco/
├── train2017/              # COCO 训练图像
├── train2014/              # RefCOCO 图像
└── annotations/
    ├── png_coco_train2017.json
    ├── panoptic_train2017.json
    ├── panoptic_train2017/
    ├── refcoco/
    │   ├── instances.json
    │   └── refs(unc).p
    ├── refcoco+/
    │   ├── instances.json
    │   └── refs(unc).p
    └── refcocog/
        ├── instances.json
        └── refs(umd).p
```

### 3. 模型权重

下载 SAM checkpoint：

```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
cd ..
```

### 4. 验证配置

在训练前，建议先验证配置文件是否正确：

```bash
python scripts/test_qwen_config.py
```

## 开始训练

### 快速开始

使用默认配置（2 GPUs）：

```bash
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2
```

### 自定义训练

**使用 4 GPUs：**

```bash
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 4
```

**修改 DeepSpeed 策略：**

```bash
./train.sh \
  --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py \
  --gpus 2 \
  --deepspeed deepspeed_zero3
```

**查看完整日志：**

```bash
./train.sh \
  --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py \
  --full-log
```

### 直接使用 xtuner 命令

如果不使用 train.sh 脚本，可以直接使用 xtuner 命令：

```bash
export PYTHONPATH=.

NPROC_PER_NODE=2 xtuner train \
  configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py \
  --deepspeed deepspeed_zero2
```

## 训练监控

### 查看日志

训练日志保存在 `logs/` 目录下：

```bash
# 查看最新日志
tail -f logs/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png_*.log

# 只查看关键指标
tail -f logs/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png_*.log | grep -E 'loss|lr|step'
```

### 监控脚本

使用项目提供的监控脚本：

```bash
./monitor_training.sh
```

### 使用 TensorBoard

如果启用了 TensorBoard：

```bash
tensorboard --logdir work_dirs/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png
```

## 预期性能

### 内存使用

- 单 GPU（batch_size=8）：约 24GB VRAM
- 建议使用 NVIDIA A100 (40GB) 或 RTX 3090/4090 (24GB)

### 训练时间

- 2 × A100 (40GB)：约 X 小时/epoch
- 4 × RTX 3090 (24GB)：约 Y 小时/epoch

（具体时间取决于数据集大小和硬件配置）

## 常见问题

### 1. OOM (内存不足)

**问题：** `CUDA out of memory`

**解决方案：**
- 减小 batch_size（如改为 4 或 2）
- 增加 accumulative_counts 以保持有效 batch size
- 使用 DeepSpeed Zero-3

编辑配置文件：

```python
batch_size = 4  # 从 8 改为 4
accumulative_counts = 16  # 从 8 改为 16
```

### 2. Transformers 版本错误

**错误：** `KeyError: 'qwen2_5_vl'`

**解决方案：** 从 GitHub 安装最新 transformers：

```bash
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers
```

### 3. 数据加载错误

**错误：** `FileNotFoundError: data/coco/...`

**解决方案：** 
- 检查数据集路径是否正确
- 如果使用不同的数据路径，修改配置文件中的 `data_root`、`local_path` 等参数

### 4. SAM checkpoint 找不到

**错误：** `FileNotFoundError: checkpoints/sam_vit_l_0b3195.pth`

**解决方案：**

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoints/sam_vit_l_0b3195.pth
```

### 5. 注意力输出为 None

**问题：** 模型不返回注意力权重

**原因：** 默认使用 SDPA 优化实现

**解决方案：** 配置文件已设置 `attn_implementation="eager"`，如果仍有问题，检查：

```python
model = dict(
    ...
    model=dict(
        ...
        attn_implementation="eager",  # 确保这行存在
    ),
)
```

## 自定义配置

### 修改模型

如果要使用不同的 Qwen 模型（如 7B）：

```python
qwen_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# 需要相应调整 UNet 的输入通道数
# Qwen2.5-VL-7B: 32 heads × 36 layers = 1152
unet = dict(
    ...
    in_channels=1152,  # 根据模型调整
)
```

### 修改训练策略

```python
# 更改学习率
lr = 5e-5

# 更改训练 epochs
max_epochs = 16

# 更改保存频率
save_steps = 200
```

### 使用不同的数据集

在 `datasets_list` 中添加或删除数据集：

```python
datasets_list = [
    dict(type=PNGDataset, ...),
    # 添加自定义数据集
    dict(type=MyCustomDataset, ...),
]
```

## 评估和推理

训练完成后，使用以下命令进行评估：

```bash
# TODO: 添加评估脚本
python scripts/evaluate.py --config configs/qwen/... --checkpoint work_dirs/.../iter_XXX.pth
```

## 参考资料

- [Qwen2.5-VL 官方文档](https://github.com/QwenLM/Qwen2.5-VL)
- [XTuner 文档](https://github.com/InternLM/xtuner)
- [F-LMM 项目文档](../doc/QWEN_MODEL_ADAPTATION.md)
- [Qwen 测试脚本说明](../../scripts/README_qwen_test.md)

## 问题反馈

如果遇到问题，请检查：
1. Python 环境和依赖版本
2. GPU 内存和 CUDA 版本
3. 数据集路径和格式
4. 配置文件参数

也可以参考项目的其他文档和测试脚本。

