# 训练概念与断点续训指南

本文档详细说明训练过程中的基本概念、checkpoint 结构、断点续训方法以及相关注意事项。

## 目录

- [基本概念](#基本概念)
- [概念之间的关系](#概念之间的关系)
- [Checkpoint 结构](#checkpoint-结构)
- [断点续训方法](#断点续训方法)
- [更改配置后的影响](#更改配置后的影响)
- [常见问题与解决方案](#常见问题与解决方案)
- [最佳实践](#最佳实践)

---

## 基本概念

### 1. Batch（批次）
- **定义**：一次送入模型的一组样本
- **示例**：64 个图像样本组成一个 batch

### 2. Batch Size（批次大小）
- **定义**：每个 batch 中包含的样本数量
- **配置位置**：`batch_size = 64  # per_device`
- **注意**：这是**每张 GPU** 的 batch size

### 3. Iteration（迭代）和 Step（步）
- **定义**：处理一个 batch 并更新一次模型参数的过程
- **关系**：`1 iteration = 1 step`（在本项目中同义）
- **含义**：每处理一个 batch，iteration 计数器 +1

### 4. Epoch（轮次）
- **定义**：遍历整个训练数据集一次
- **配置位置**：`max_epochs = 8`
- **含义**：完成一个 epoch 意味着模型"看过了"所有训练数据一次

### 5. Gradient Accumulation（梯度累积）
- **定义**：累积多个 batch 的梯度后再更新参数
- **配置位置**：`accumulative_counts = 2`
- **作用**：在显存有限时，通过累积梯度来模拟更大的 batch size

---

## 概念之间的关系

### 计算公式

假设：
- 数据集大小：`N` 个样本
- 每 GPU batch size：`batch_size`
- GPU 数量：`num_gpus`
- 梯度累积步数：`accumulative_counts`

#### 1. 有效 Batch Size（Effective Batch Size）

```
有效 batch_size = batch_size × num_gpus × accumulative_counts
```

**示例**：
```python
batch_size = 64           # 每 GPU
num_gpus = 2              # 2 张 GPU
accumulative_counts = 2   # 梯度累积 2 步

有效 batch_size = 64 × 2 × 2 = 256
```

#### 2. 每个 Epoch 的 Iterations 数

```
iterations_per_epoch = N / (batch_size × num_gpus)
```

**注意**：梯度累积不影响每个 epoch 的 iterations 数，只影响参数更新频率。

#### 3. 总 Iterations 数

```
total_iterations = iterations_per_epoch × max_epochs
```

#### 4. 参数更新频率

- 梯度累积每 `accumulative_counts` 个 iteration 才真正更新一次参数
- 但 iteration 计数器仍然每个 batch 增加 1

**示例流程**：
```
Iteration 1: 处理 batch 1，累积梯度
Iteration 2: 处理 batch 2，累积梯度
→ Iteration 2 后: 真正更新参数（因为 accumulative_counts=2）

Iteration 3: 处理 batch 3，累积梯度
Iteration 4: 处理 batch 4，累积梯度
→ Iteration 4 后: 真正更新参数
...
```

### 实际示例

假设数据集有 100,000 个样本：

```
配置：
- batch_size = 64 (每 GPU)
- num_gpus = 2
- accumulative_counts = 2
- max_epochs = 8

计算：
- 每个 epoch 的 iterations = 100,000 / (64 × 2) = 781 iterations
- 总 iterations = 781 × 8 = 6,248 iterations
- 有效 batch_size = 64 × 2 × 2 = 256
- 参数更新次数 = 6,248 / 2 = 3,124 次
```

---

## Checkpoint 结构

### DeepSpeed ZeRO-2 Checkpoint

使用 DeepSpeed ZeRO-2 时，checkpoint 是一个**目录**，包含多个文件：

```
iter_1500.pth/
├── mp_rank_00_model_states.pt          # 模型参数（所有 GPU 共享）
├── bf16_zero_pp_rank_0_optim_states.pt # GPU 0 的优化器状态（分片）
└── bf16_zero_pp_rank_1_optim_states.pt # GPU 1 的优化器状态（分片）
```

#### 文件说明

1. **`mp_rank_00_model_states.pt`**
   - 模型权重（state_dict）
   - 所有 GPU 共享同一份
   - `mp_rank_00` 表示模型并行 rank 0

2. **`bf16_zero_pp_rank_X_optim_states.pt`**
   - 各 GPU 的优化器状态（分片）
   - `pp_rank_X` 表示 pipeline parallel rank
   - ZeRO-2 将优化器状态分片到各个 GPU 以节省显存

#### 命名规则

- `iter_1500`：表示第 1500 个 iteration 的 checkpoint
- `bf16_zero`：使用 bfloat16 精度和 ZeRO 优化
- `pp_rank_X`：Pipeline Parallel Rank（2 个 GPU 时为 0 和 1）
- `mp_rank_00`：Model Parallel Rank（单卡模型时为 00）

### Checkpoint 保存配置

```python
checkpoint=dict(
    type=CheckpointHook,
    by_epoch=False,        # 按 iteration 保存，不是按 epoch
    interval=save_steps,   # 每 100 个 iteration 保存一次
    max_keep_ckpts=1      # 最多保留 1 个 checkpoint
)
```

**文件名示例**：`iter_100.pth`, `iter_200.pth`, `iter_300.pth`...

---

## 断点续训方法

### 方法 1：自动从最新 Checkpoint 恢复（推荐）

在配置文件中设置：

```python
# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = True  # 改为 True
```

**工作原理**：
- 系统会自动查找 `work_dir` 中的最新 checkpoint
- 恢复模型权重、优化器状态、学习率调度器状态和迭代次数
- 从中断处继续训练

**注意**：需要确保 `work_dir` 中有可用的 checkpoint。

### 方法 2：从指定 Checkpoint 恢复

在配置文件中设置：

```python
# load from which checkpoint
load_from = 'work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/iter_1500.pth'

# whether to resume training from the loaded checkpoint
resume = True  # 保持 True
```

**重要提示**：
- `load_from` 应指向 checkpoint **目录**（如 `iter_1500.pth/`），而不是单个文件
- DeepSpeed checkpoint 是一个目录，包含多个分片文件
- 使用**绝对路径**更安全

**示例**：
```python
load_from = '/data/gyk/F-LMM/work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/iter_1500.pth'
```

### 方法 3：只加载权重（从头开始训练）

如果只想加载模型权重，但不恢复优化器状态和迭代次数：

```python
# load from which checkpoint
load_from = 'work_dirs/.../iter_1500.pth'

# whether to resume training from the loaded checkpoint
resume = False  # 保持 False
```

**用途**：常用于微调或迁移学习。

### 检查是否成功恢复

恢复训练时，日志中应该看到类似信息：

```
Auto resumed from the latest checkpoint work_dirs/.../iter_1500.pth.
```

如果没有看到此信息，说明可能没有成功恢复。

---

## 更改配置后的影响

### 改变 Batch Size 的影响

#### ⚠️ 问题 1：每个 Epoch 的 Iterations 数会改变

**示例**：
```
之前：batch_size = 128
- iterations_per_epoch = N / (128 × 2) = 390 iterations
- iter_1500 对应约 3.8 个 epoch

现在：batch_size = 64
- iterations_per_epoch = N / (64 × 2) = 781 iterations
- iter_1500 对应约 1.9 个 epoch
```

**影响**：Checkpoint 中的 iteration 计数（如 `iter_1500`）的语义会发生变化。

#### ⚠️ 问题 2：学习率调度可能有偏差

如果学习率调度器是基于 epoch 设计的：

```python
param_scheduler = [
    dict(
        type=LinearLR,
        by_epoch=True,  # 按 epoch 设计
        convert_to_iter_based=True  # 转换为基于 iteration
    )
]
```

改变 batch_size 后，每个 epoch 的 iterations 数改变，可能导致学习率调度不准确。

#### ⚠️ 问题 3：训练进度语义变化

```
之前的训练：
- iter_1500 ≈ 3.8 epochs（47.5% 完成）

改变 batch_size 后恢复：
- iter_1500 ≈ 1.9 epochs（23.8% 完成）
```

### 解决方案

#### ✅ 方案 1：保持有效 Batch Size 不变（最佳实践）

如果改变 `batch_size` 是为了解决 OOM，应该调整梯度累积或 GPU 数量：

```python
# 原来的配置
batch_size = 128
accumulative_counts = 1
num_gpus = 2
# 有效 batch_size = 128 × 2 × 1 = 256

# 解决 OOM 的配置（保持相同有效 batch_size）
batch_size = 64           # 减半
accumulative_counts = 2   # 加倍
num_gpus = 2
# 有效 batch_size = 64 × 2 × 2 = 256 ✅ 保持不变！
```

**优点**：
- ✅ 保持相同的有效 batch size（训练效果一致）
- ✅ 保持相同的训练进度语义
- ✅ 可以安全地续训
- ✅ 学习率调度不受影响

#### ✅ 方案 2：从头开始训练

如果有效 batch size 必须改变，建议从头开始：

```python
resume = False
load_from = None
```

#### ⚠️ 方案 3：手动调整学习率调度（不推荐）

如果必须续训且改变了有效 batch size，需要手动调整学习率调度：

```python
# 计算当前实际在哪个 epoch
original_iterations_per_epoch = 390  # 需要根据数据集大小计算
current_epoch = iter_1500 / original_iterations_per_epoch  # ≈ 3.85

# 手动设置学习率调度从当前 epoch 开始
param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        by_epoch=True,
        begin=current_epoch,  # 从当前 epoch 开始
        end=max_epochs,
        convert_to_iter_based=True)
]
```

---

## 常见问题与解决方案

### Q1: 为什么 checkpoint 有三个文件？

**A**: 这是 DeepSpeed ZeRO-2 的分布式 checkpoint 结构：
- 1 个模型参数文件（所有 GPU 共享）
- N 个优化器状态文件（每个 GPU 一个分片）

这是**正常现象**，不是错误。

### Q2: 如何查看可用的 checkpoint？

**A**: 
```bash
# 列出所有 checkpoint
ls -lt work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/iter_*.pth

# 查看最新 checkpoint 路径
cat work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/last_checkpoint
```

### Q3: 训练中断后如何恢复？

**A**: 
1. 检查是否有保存的 checkpoint（每 `save_steps` 个 iteration 保存一次）
2. 在配置文件中设置 `resume = True`
3. 如果使用指定 checkpoint，设置 `load_from = 'path/to/checkpoint'`
4. 重新运行训练脚本

### Q4: 改变 batch_size 后能续训吗？

**A**: 
- ✅ **可以**：如果保持有效 batch size 不变（通过调整 `accumulative_counts` 或 GPU 数量）
- ⚠️ **不推荐**：如果改变了有效 batch size（建议从头开始）

### Q5: 为什么日志中没有看到恢复 checkpoint 的信息？

**A**: 可能的原因：
1. 使用的是 `FlexibleRunner` 而不是 `CustomRunner`（xtuner 默认）
2. DeepSpeed 的恢复可能需要显式指定 checkpoint 路径
3. 检查 `load_from` 是否设置为 `None` 且 `resume = True`

**解决方法**：显式指定 checkpoint 路径：
```python
load_from = 'work_dirs/.../iter_1500.pth'
resume = True
```

### Q6: Checkpoint 保存太频繁/太少了怎么办？

**A**: 修改配置文件中的 `save_steps`：

```python
save_steps = 100  # 每 100 个 iteration 保存一次
```

**建议**：
- 训练初期可以设置较小值（如 50-100）
- 训练稳定后可以设置较大值（如 500-1000）
- 注意 `save_total_limit` 控制保留的 checkpoint 数量

---

## 最佳实践

### 1. Checkpoint 保存策略

```python
# 初期训练：频繁保存
save_steps = 100
save_total_limit = 3  # 保留多个 checkpoint

# 稳定训练：减少保存频率
save_steps = 500
save_total_limit = 1  # 只保留最新
```

### 2. 解决 OOM 问题

**推荐做法**：保持有效 batch size 不变

```python
# 方案 A：减小 batch_size，增加梯度累积
batch_size = 64
accumulative_counts = 2  # 从 1 增加到 2

# 方案 B：减小 batch_size，增加 GPU 数量
batch_size = 64
num_gpus = 4  # 从 2 增加到 4

# 方案 C：组合方案
batch_size = 32
accumulative_counts = 2
num_gpus = 4
```

### 3. 断点续训检查清单

- [ ] 确认 checkpoint 存在且完整
- [ ] 检查 `resume = True` 和 `load_from` 设置正确
- [ ] 确认配置文件中的 `batch_size`、`accumulative_counts` 等与之前一致（或有效 batch size 一致）
- [ ] 检查学习率调度配置是否合理
- [ ] 查看日志确认成功恢复

### 4. 训练监控

```bash
# 实时查看训练日志
tail -f logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_*.log

# 监控 GPU 使用情况
watch -n 1 nvidia-smi

# 检查 checkpoint 保存情况
ls -lht work_dirs/.../iter_*.pth
```

### 5. 配置文件管理

**建议**：
- 为每次重要训练保存配置文件副本
- 记录训练时的关键参数（batch_size, lr, epoch 等）
- 在配置文件中添加注释说明训练目标

---

## 总结

### 关键概念速查表

| 概念 | 定义 | 配置示例 |
|------|------|---------|
| **Batch** | 一组样本 | 64 个样本 |
| **Batch Size** | 每 batch 样本数 | `batch_size = 64`（每 GPU） |
| **有效 Batch Size** | 实际用于更新的样本数 | `64 × 2 × 2 = 256` |
| **Iteration/Step** | 处理一个 batch | 1 iter = 1 step |
| **Epoch** | 遍历整个数据集一次 | `max_epochs = 8` |
| **Checkpoint** | 训练状态快照 | `iter_1500.pth/` |

### 断点续训快速参考

```python
# 自动恢复（推荐）
resume = True
load_from = None

# 指定 checkpoint 恢复
resume = True
load_from = 'work_dirs/.../iter_1500.pth'

# 只加载权重（从头开始）
resume = False
load_from = 'work_dirs/.../iter_1500.pth'
```

### 更改配置的建议

- ✅ **可以改**：`save_steps`, `max_epochs`, `lr`（需谨慎）
- ⚠️ **谨慎改**：`batch_size`（确保有效 batch size 不变或从头开始）
- ❌ **不要改**：模型结构、优化器类型（续训时）

---

**最后更新**：2025-11-03  
**适用版本**：F-LMM with DeepSpeed ZeRO-2

