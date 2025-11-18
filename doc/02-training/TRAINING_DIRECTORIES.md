# F-LMM 训练目录说明

## 概述

F-LMM 训练会产生两个主要目录来保存不同的内容：

```
F-LMM/
├── logs/                    # train.sh 脚本保存的日志
└── work_dirs/               # MMEngine/Xtuner 框架的工作目录
```

## 目录详细说明

### 1. `logs/` 目录

**创建者**: `train.sh` 脚本  
**用途**: 保存训练命令的完整输出日志  
**内容**: 所有终端输出（stdout + stderr）

#### 文件命名规则

```
logs/{配置文件名}_{时间戳}.log
```

例如：
```
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_001626.log
```

#### 包含的内容

- ✅ 所有训练过程的完整输出
- ✅ 错误和警告信息
- ✅ 初始化信息（CUDA、GPU 检测等）
- ✅ 模型加载信息
- ✅ 训练迭代日志（损失、准确率等）
- ✅ 保存检查点的信息

#### 特点

- **时间戳对应训练启动时间**
- **每个训练运行一个文件**
- **完整记录，方便事后查看**

### 2. `work_dirs/` 目录

**创建者**: MMEngine/Xtuner 框架  
**用途**: 框架的工作目录，保存训练相关文件  
**结构**: 按配置文件名称和时间戳组织

#### 目录结构

```
work_dirs/
└── {配置文件名}/
    ├── {时间戳}/              # 每次训练的时间戳目录
    │   ├── {时间戳}.log       # 框架自己的日志文件
    │   ├── iter_500.pth       # 检查点文件
    │   ├── iter_1000.pth
    │   ├── latest.pth         # 最新检查点的软链接
    │   ├── vis_data/          # 可视化数据
    │   └── ...
    ├── frozen_xxx.py          # 配置文件副本
    ├── last_checkpoint        # 最新检查点路径记录
    └── zero_to_fp32.py        # DeepSpeed 工具脚本
```

#### 包含的内容

- ✅ **检查点文件** (`.pth`): 训练保存的模型权重
- ✅ **框架日志**: MMEngine 自己的日志系统输出
- ✅ **配置文件副本**: 训练使用的配置备份
- ✅ **可视化数据**: `vis_data/` 目录（如果启用）
- ✅ **DeepSpeed 相关**: `zero_to_fp32.py` 等工具

#### 特点

- **框架自动管理**
- **检查点保存在这里**
- **每个训练一个时间戳子目录**
- **方便框架恢复训练和加载模型**

## 为什么有两个日志？

### 原因

1. **`logs/`**: `train.sh` 脚本保存的原始输出
   - 包含所有内容（包括 bitsandbytes 警告等）
   - 方便查看完整的训练过程
   - 脚本层面的日志记录

2. **`work_dirs/`**: MMEngine 框架的日志
   - 框架自己的日志系统
   - 更结构化，只包含框架相关信息
   - 与检查点保存在同一位置，方便管理

### 对比

| 特性 | `logs/` | `work_dirs/` |
|------|---------|--------------|
| **创建者** | train.sh 脚本 | MMEngine 框架 |
| **内容** | 完整终端输出 | 框架日志 + 检查点 |
| **用途** | 查看完整训练过程 | 框架管理和恢复训练 |
| **检查点** | ❌ 不保存 | ✅ 保存 |
| **文件名** | 配置名_时间戳.log | 时间戳/时间戳.log |

## 实际例子

假设你运行训练：

```bash
./train.sh --gpus 2
```

会产生：

### logs/ 目录
```
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_120000.log
```
- 完整的训练输出
- 可以看到所有警告、错误、进度等

### work_dirs/ 目录
```
work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/
└── 20251103_120000/
    ├── 20251103_120000.log     # 框架日志
    ├── iter_500.pth            # 检查点
    ├── iter_1000.pth
    └── ...
```
- 框架管理的文件
- **检查点在这里！**

## 为什么有多个日志文件？

你看到多个日志文件是因为：

1. **多次运行训练**: 每次运行 `./train.sh` 都会创建新的日志文件
2. **时间戳不同**: 每次启动时间不同，文件名也不同

例如：
```
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_001626.log  # 第一次运行
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_001729.log  # 第二次运行（可能失败了）
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_003102.log  # 第三次运行
logs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png_20251103_004300.log  # 第四次运行
```

## 如何查找需要的文件？

### 查找最新的日志

```bash
# logs/ 目录最新日志
ls -t logs/*.log | head -1

# work_dirs/ 目录最新日志
find work_dirs -name "*.log" -type f | sort -r | head -1
```

### 查找最新的检查点

```bash
# 查找最新检查点
find work_dirs -name "iter_*.pth" -type f | sort -V | tail -1

# 或查看 last_checkpoint 文件
cat work_dirs/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png/last_checkpoint
```

### 查找对应关系的训练

如果想知道哪个 `logs/` 文件对应哪个 `work_dirs/` 目录：

```bash
# 查看 logs/ 文件的时间戳（文件名中）
# 然后在 work_dirs/ 中查找相同时间戳的目录

# 例如：
# logs/frozen_xxx_20251103_001626.log
# 对应
# work_dirs/frozen_xxx/20251103_001626/
```

## 清理建议

### 保留的文件

- ✅ **最新的日志文件**: 查看当前训练状态
- ✅ **检查点文件**: 用于恢复和评估
- ✅ **配置副本**: 了解训练使用的配置

### 可以删除的文件

- ⚠️ **旧的日志文件**: 如果不需要历史记录
- ⚠️ **失败的训练日志**: 如果确定不再需要

### 清理脚本示例

```bash
# 只保留最新的 3 个日志文件
cd logs
ls -t *.log | tail -n +4 | xargs rm -f

# 只保留最新的检查点（根据配置 save_total_limit）
# 框架会自动管理，手动删除需要谨慎
```

## 总结

| 目录 | 主要用途 | 重要文件 |
|------|---------|---------|
| **logs/** | 查看完整训练过程 | `.log` 文件 |
| **work_dirs/** | 保存检查点和框架管理 | `iter_*.pth` 检查点 |

**关键点**:
- `logs/` = 训练日志（查看用）
- `work_dirs/` = 工作目录（检查点在这里！）
- 两者互补，各有用途

## 快速查找指南

```bash
# 1. 查看当前训练进度（logs/）
tail -f logs/$(ls -t logs/*.log | head -1)

# 2. 找到最新检查点（work_dirs/）
ls -lt work_dirs/*/202*/iter_*.pth | head -1

# 3. 查看框架日志（work_dirs/）
tail -f work_dirs/$(ls -td work_dirs/*/202* | head -1)/*.log
```

