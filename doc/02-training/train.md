# F-LMM 训练使用指南

本文档介绍如何使用训练脚本 `train.sh` 和监控脚本 `monitor_training.sh` 来方便地进行模型训练。

## 快速开始

### 最简单的使用方式

```bash
cd /home/cvprtemp/gyk/F-LMM
./train.sh
```

这将使用默认配置启动训练：
- 配置文件：`configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`
- GPU 数量：2
- DeepSpeed 策略：`deepspeed_zero2`
- 日志保存：`logs/` 目录
- 终端显示：只显示关键信息（损失、学习率、步数等）

## train.sh 详细用法

### 基本语法

```bash
./train.sh [选项]
```

### 可用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--config PATH` | 配置文件路径 | `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py` |
| `--gpus N` | 使用的 GPU 数量 | `2` |
| `--deepspeed STR` | DeepSpeed 策略 | `deepspeed_zero2` |
| `--log-dir DIR` | 日志保存目录 | `logs` |
| `--full-log` | 显示完整日志（不过滤） | `false` |
| `--help, -h` | 显示帮助信息 | - |

### 使用示例

#### 1. 使用默认配置训练

```bash
./train.sh
```

#### 2. 使用更多 GPU

```bash
# 使用 4 张 GPU
./train.sh --gpus 4

# 使用 8 张 GPU
./train.sh --gpus 8
```

#### 3. 使用其他配置文件

```bash
# 使用 LLaVA 配置文件
./train.sh --config configs/llava/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.py

# 使用 MiniGemini 配置文件
./train.sh --config configs/mgm/frozen_mgm_vicuna_7b_unet_sam_l_refcoco_png.py
```

#### 4. 显示完整日志

```bash
# 终端显示所有输出（不过滤）
./train.sh --full-log
```

#### 5. 自定义日志目录

```bash
./train.sh --log-dir my_logs
```

#### 6. 组合使用多个选项

```bash
# 使用 4 张 GPU，LLaVA 配置，显示完整日志
./train.sh --gpus 4 \
           --config configs/llava/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.py \
           --full-log
```

## monitor_training.sh 监控脚本

训练启动后，可以在另一个终端使用监控脚本实时查看训练进度。

### 基本用法

```bash
# 自动查找最新日志并监控
./monitor_training.sh

# 或指定日志文件路径
./monitor_training.sh logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
```

### 监控模式

脚本提供 4 种监控模式：

1. **只显示关键信息**（推荐）
   - 显示：损失、学习率、步数、epoch、保存检查点等信息
   - 适合：日常监控训练进度

2. **显示完整日志**
   - 显示：所有训练输出
   - 适合：调试问题或需要查看详细信息

3. **只看错误和警告**
   - 显示：ERROR、WARNING、Exception、Traceback
   - 适合：快速发现问题

4. **只看损失变化**
   - 显示：损失值的变化
   - 适合：观察训练收敛情况

## 完整工作流程示例

### 步骤 1: 启动训练（终端 1）

```bash
cd /home/cvprtemp/gyk/F-LMM

# 启动训练（使用 2 张 GPU，默认配置）
./train.sh --gpus 2
```

输出示例：
```
==========================================
F-LMM 训练脚本
==========================================
配置文件: configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py
GPU 数量: 2
DeepSpeed: deepspeed_zero2
日志文件: logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
==========================================

检测到的 GPU:
0, NVIDIA A100, 39168 MiB
1, NVIDIA A100, 39168 MiB

训练将启动，终端只显示关键信息（损失、学习率、步数等）
完整日志保存在: logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log

在另一个终端可以使用以下命令监控训练:
  tail -f logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
  tail -f logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log | grep -E 'loss|lr|step'

开始训练...

[训练输出...]
```

### 步骤 2: 监控训练（终端 2，可选）

```bash
cd /home/cvprtemp/gyk/F-LMM

# 启动监控
./monitor_training.sh
```

选择监控模式：
```
选择监控模式:
  1) 只显示关键信息（损失、学习率、步数等）- 推荐
  2) 显示完整日志
  3) 只看错误和警告
  4) 只看损失变化
请选择 [1-4] (默认: 1):
```

### 步骤 3: 查看日志文件（如果需要）

```bash
# 查看完整日志
tail -f logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log

# 只看关键信息
tail -f logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log | grep -E 'loss|lr|step|epoch'

# 查看最后 100 行
tail -100 logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
```

## 日志文件说明

### 日志文件位置

训练日志保存在 `logs/` 目录下，文件名格式为：
```
{配置文件名}_{时间戳}.log
```

例如：
```
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
```

### 日志内容

日志文件包含：
- ✅ 完整的训练输出（所有 print、log 信息）
- ✅ 错误和警告信息
- ✅ 训练进度（损失、学习率、步数等）
- ✅ 检查点保存信息

### 查找最新日志

```bash
# 查找最新的日志文件
ls -t logs/*.log | head -1

# 或使用 find
find logs -name "*.log" -type f | sort -r | head -1
```

## 训练检查点位置

训练过程中，模型检查点会保存在 `work_dirs/` 目录下：

```
work_dirs/
└── {配置文件名}/
    └── {时间戳}/
        ├── iter_500.pth      # 每 500 steps 保存一次
        ├── iter_1000.pth
        ├── ...
        └── latest.pth        # 指向最新检查点
```

详细说明请参考项目 README 或 `gykreadme.md`。

## 常见问题

### Q1: 训练被中断了怎么办？

**A**: 训练脚本会自动保存日志。可以：
1. 查看日志文件了解中断原因
2. 如果需要继续训练，可以使用 `--resume` 参数（需要在配置文件中设置）

### Q2: 如何查看之前的训练日志？

**A**: 
```bash
# 列出所有日志
ls -lt logs/*.log

# 查看特定日志
cat logs/日志文件名.log
```

### Q3: 训练时显存不够怎么办？

**A**: 
1. 减少 GPU 数量（如从 4 减到 2）
2. 使用梯度累积（修改配置文件中的 `accumulative_counts`）
3. 减小 batch size（如果配置支持）

### Q4: 如何停止训练？

**A**: 在训练终端按 `Ctrl+C` 即可安全停止训练。

### Q5: 训练脚本报错 "找不到配置文件"

**A**: 确保：
1. 配置文件路径正确
2. 在项目根目录（`F-LMM/`）运行脚本
3. 使用相对路径或绝对路径

### Q6: 如何使用其他 DeepSpeed 策略？

**A**: 根据 README，目前推荐使用 `deepspeed_zero2`：
```bash
./train.sh --deepspeed deepspeed_zero2
```
注意：README 提到 `deepspeed_zero3` 有 bug，不建议使用。

## 高级用法

### 后台运行训练

如果想让训练在后台运行，可以使用：

```bash
# 使用 nohup（即使终端关闭也继续运行）
nohup ./train.sh > train_output.log 2>&1 &

# 查看后台进程
jobs

# 查看输出
tail -f train_output.log
```

### 使用 screen 或 tmux

```bash
# 使用 screen
screen -S training
./train.sh
# 按 Ctrl+A 然后 D 分离会话

# 重新连接
screen -r training
```

### 批量训练多个配置

可以创建一个批量训练脚本：

```bash
#!/bin/bash
# batch_train.sh

CONFIGS=(
    "configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py"
    "configs/llava/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.py"
)

for config in "${CONFIGS[@]}"; do
    echo "Training with $config"
    ./train.sh --config "$config" --gpus 2
    echo "Completed: $config"
done
```

## 检查训练状态

### 查看 GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat
gpustat -i 1
```

### 查看训练进程

```bash
# 查找训练进程
ps aux | grep xtuner

# 查看进程资源使用
top -p $(pgrep -f xtuner)
```

## 相关文档

- **项目主文档**: `README.md`
- **项目详细说明**: `gykreadme.md`
- **任务清单**: `todo.md`
- **Grounded Conversation 使用**: `scripts/demo/grounded_conversation.md`

## 注意事项

1. **确保环境变量**: 训练前确保设置了 `PYTHONPATH=.`（脚本会自动设置）
2. **数据准备**: 确保训练数据已准备好（参考 README 的 Data Preparation）
3. **检查点路径**: 确保 SAM 检查点已下载到 `checkpoints/sam_vit_l_0b3195.pth`
4. **DeepSpeed 版本**: 确保 DeepSpeed >= 0.13.2（见 README）
5. **显存管理**: 根据 GPU 显存调整 batch_size 和 GPU 数量

---

**祝训练顺利！** 🚀

