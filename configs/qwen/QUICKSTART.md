# Qwen 模型训练快速入门

## 1. 激活环境

```bash
# 使用 Qwen 专用环境（推荐）
conda activate flmm-qwen-py310

# 或使用默认 F-LMM 环境
conda activate flmm
```

## 2. 检查环境配置

```bash
cd /home/cvprtemp/gyk/F-LMM

# 测试配置文件加载
export PYTHONPATH=.
python scripts/test_qwen_config.py
```

如果遇到缺少依赖的问题，参考下面的安装步骤。

## 3. 安装 Qwen 依赖（如果需要）

如果使用 `flmm-qwen-py310` 环境，且依赖未安装：

```bash
conda activate flmm-qwen-py310

# 安装 transformers（必须从 GitHub 安装）
pip install git+https://github.com/huggingface/transformers

# 安装 Qwen 相关工具
pip install qwen-vl-utils

# 安装其他可能缺失的依赖
pip install accelerate "numpy<2" mmengine xtuner
```

## 4. 准备数据

确保数据目录结构正确：

```bash
ls -la data/coco/train2017/ | head -5
ls -la data/coco/train2014/ | head -5
ls -la data/coco/annotations/
```

如果数据不存在，需要从以下位置下载：
- COCO: https://cocodataset.org/
- RefCOCO: https://github.com/lichengunc/refer

## 5. 准备 SAM Checkpoint

```bash
# 检查是否已存在
ls -lh checkpoints/sam_vit_l_0b3195.pth

# 如果不存在，下载
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
cd ..
```

## 6. 测试训练配置（干跑）

在实际训练前，建议先做一个快速测试：

```bash
# 修改配置为测试模式（可选）
# 编辑 configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py
# 将 max_epochs = 8 改为 max_epochs = 1
# 将 save_steps = 100 改为 save_steps = 10

# 运行单 GPU 测试
export PYTHONPATH=.
xtuner train configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --work-dir work_dirs/qwen_test
```

## 7. 开始正式训练

### 方法 1: 使用训练脚本（推荐）

```bash
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2
```

### 方法 2: 直接使用 xtuner

```bash
export PYTHONPATH=.

# 使用 2 GPUs + DeepSpeed Zero2
NPROC_PER_NODE=2 xtuner train \
  configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py \
  --deepspeed deepspeed_zero2
```

### 方法 3: 使用 torchrun

```bash
export PYTHONPATH=.

torchrun --nproc_per_node=2 \
  $(which xtuner) train \
  configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py \
  --deepspeed deepspeed_zero2
```

## 8. 监控训练

### 查看日志

```bash
# 实时查看日志
tail -f logs/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png_*.log

# 只看关键信息
tail -f logs/*.log | grep -E 'loss|lr|step|epoch'
```

### 使用监控脚本

```bash
./monitor_training.sh
```

### 检查 GPU 使用情况

```bash
watch -n 1 nvidia-smi
```

## 9. 训练完成后

检查生成的 checkpoint：

```bash
ls -lh work_dirs/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png/
```

## 常见问题速查

### Q1: ModuleNotFoundError: No module named 'transformers'

```bash
pip install git+https://github.com/huggingface/transformers
```

### Q2: KeyError: 'qwen2_5_vl'

```bash
# 必须从 GitHub 安装最新 transformers
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers
```

### Q3: CUDA out of memory

编辑配置文件，减小 batch_size：

```python
batch_size = 4  # 从 8 改为 4
accumulative_counts = 16  # 从 8 改为 16（保持有效 batch size）
```

### Q4: FileNotFoundError: data/coco/...

检查数据路径，或修改配置文件中的路径：

```python
dict(type=PNGDataset,
     local_path='YOUR_PATH/coco/train2017',  # 修改为实际路径
     ...)
```

### Q5: 训练速度很慢

- 减少 `dataloader_num_workers`（如果内存不足）
- 使用更快的存储（SSD 而非 HDD）
- 检查数据加载是否成为瓶颈（使用 `htop` 监控）

## 下一步

1. **调整超参数**: 根据验证集性能调整学习率、batch size 等
2. **监控指标**: 关注 loss、accuracy、IoU 等指标
3. **模型评估**: 使用测试集评估模型性能
4. **模型导出**: 导出模型用于推理

## 获取帮助

- 查看详细文档: `configs/qwen/README.md`
- 查看 Qwen 适配文档: `doc/QWEN_MODEL_ADAPTATION.md`
- 查看测试脚本说明: `scripts/README_qwen_test.md`

