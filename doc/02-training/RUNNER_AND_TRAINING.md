# F-LMM 训练器与训练流程详解

本文档详细解释 F-LMM 的训练器（Runner）、训练循环组织、Hook 系统和检查点机制。

## 目录

- [Runner 概述](#runner-概述)
- [1. CustomRunner 自定义实现](#1-customrunner-自定义实现)
- [2. 训练循环组织](#2-训练循环组织)
  - [2.1 TrainLoop 配置](#21-trainloop-配置)
  - [2.2 训练循环结构](#22-训练循环结构)
  - [2.3 训练步骤详解](#23-训练步骤详解)
  - [2.4 配置文件理解](#24-配置文件理解)
- [3. Hook 系统](#3-hook-系统)
- [4. 检查点加载和保存机制](#4-检查点加载和保存机制)
- [5. 模型实现对比分析](#5-模型实现对比分析)
- [6. DeepSpeed 集成](#6-deepspeed-集成)
- [训练流程完整图](#训练流程完整图)

---

## Runner 概述

**Runner** 是 MMEngine/Xtuner 框架的核心组件，负责协调整个训练过程：

- **组织训练循环**: 管理 epoch、iteration 的执行
- **管理 Hook**: 在训练的不同阶段调用注册的 Hook
- **保存/加载检查点**: 处理模型状态的持久化
- **日志和可视化**: 记录训练过程和指标
- **分布式训练**: 协调多 GPU/多进程训练

---

## 1. CustomRunner 自定义实现

### 1.1 自定义 Runner 的原因

F-LMM 使用自定义的 `CustomRunner` (`flmm/runner.py`)，主要为了：

1. **DeepSpeed Checkpoint 兼容**: 处理 DeepSpeed 的特殊 checkpoint 格式
2. **灵活加载权重**: 使用 `guess_load_checkpoint` 自动识别 checkpoint 格式
3. **只保存可训练参数**: 只保存需要训练的模块（U-Net、投影层等）

### 1.2 CustomRunner 的关键方法

#### `load_or_resume()` - 检查点加载

```python
def load_or_resume(self) -> None:
    """加载或恢复检查点"""
    
    # 情况 1: 自动恢复（从最新 checkpoint）
    if self._resume and self._load_from is None:
        resume_from = find_latest_checkpoint(self.work_dir)
        self.resume(resume_from)  # 恢复训练（包括优化器状态）
    
    # 情况 2: 从指定 checkpoint 恢复
    elif self._resume and self._load_from is not None:
        self.resume(self._load_from)
    
    # 情况 3: 只加载权重（不恢复训练状态）
    elif self._load_from is not None:
        state_dict = guess_load_checkpoint(self._load_from)  # 自动识别格式
        self.model.load_state_dict(state_dict, strict=False)
```

**关键点**:
- `resume = True`: 恢复训练状态（optimizer, scheduler, iteration count）
- `resume = False`: 只加载模型权重，从头开始训练
- `guess_load_checkpoint`: 自动识别 DeepSpeed 或 PyTorch checkpoint 格式

#### `save_checkpoint()` - 检查点保存

```python
def save_checkpoint(self, ...):
    """保存检查点"""
    
    # 1. 准备元数据
    meta = {
        'epoch': self.epoch + 1,  # 当前 epoch
        'iter': self.iter + 1,     # 当前 iteration
        'cfg': self.cfg.pretty_text,  # 配置信息
        # ...
    }
    
    # 2. 只保存可训练的参数
    model_parameters = {
        k: v.detach() 
        for k, v in model.named_parameters() 
        if v.requires_grad  # 只保存需要训练的参数
    }
    
    # 3. 构建 checkpoint 字典
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(OrderedDict(model_parameters)),
        'optimizer': self.optim_wrapper.state_dict(),  # 优化器状态
        'param_schedulers': ...,  # 学习率调度器状态
        'message_hub': ...,  # 消息中心状态
    }
    
    # 4. 调用 Hook（如果有）
    self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
    
    # 5. 保存到文件
    save_checkpoint(checkpoint, filepath)
```

**关键点**:
- **只保存可训练参数**: `if v.requires_grad` - LMM 参数被冻结，不需要保存
- **DeepSpeed 兼容**: 框架会自动处理 DeepSpeed checkpoint 格式
- **Hook 支持**: `before_save_checkpoint` Hook 可以在保存前修改 checkpoint

---

## 2. 训练循环组织

### 2.1 TrainLoop 配置

```python
# 配置文件
train_cfg = dict(type=TrainLoop, max_epochs=8)
```

**TrainLoop** (`xtuner/engine/runner.py`) 负责：
- 管理 epoch 循环
- 管理 iteration 循环
- 在适当的时候调用 Hook

### 2.2 训练循环结构

```
训练开始
  ↓
before_run()  # 全局 Hook
  ↓
before_train()  # 训练前 Hook
  ↓
for epoch in range(max_epochs):
    ↓
    before_train_epoch()  # Epoch 开始前 Hook
      ↓
    for iteration, data_batch in train_dataloader:
        ↓
        before_train_iter()  # Iteration 开始前 Hook
          ↓
        # 1. 前向传播
        outputs = model.train_step(data_batch)
          ├─ model._forward(data_batch)  # 提取注意力，生成 mask
          ├─ model.compute_loss(data_batch)  # 计算损失
          └─ loss.backward() + optimizer.step()  # 反向传播和更新
          ↓
        after_train_iter()  # Iteration 结束后 Hook
        # (包括: LoggerHook 打印日志, CheckpointHook 保存模型等)
      ↓
    after_train_epoch()  # Epoch 结束后 Hook
  ↓
after_train()  # 训练结束后 Hook
  ↓
after_run()  # 全局 Hook
```

### 2.3 训练步骤详解

#### `model.train_step()` 内部流程

```python
# 在模型内部 (BaseModel.train_step)
def train_step(self, data, optim_wrapper):
    # 1. 前向传播
    loss_dict = self.compute_loss(data)
    
    # 2. 解析损失
    loss = sum(loss_dict.values())
    
    # 3. 反向传播（由 OptimWrapper 处理）
    optim_wrapper.update_params(loss)
    
    return loss_dict
```

#### `compute_loss()` 实现（F-LMM）

```python
# frozen_deepseek_vl.py: compute_loss()
def compute_loss(self, data):
    loss_dice = 0
    loss_mask = 0
    sam_loss_dice = 0
    sam_loss_mask = 0
    
    for data_sample in data:  # 遍历 batch 中的样本
        # 前向传播
        forward_output = self._forward(data_sample)
        pred_masks = forward_output['pred_masks']
        sam_pred_masks = forward_output['sam_pred_masks']
        
        # 计算 U-Net 损失
        loss_dice_, loss_mask_, ... = self._compute(pred_masks, gt_masks)
        
        # 计算 SAM 损失
        sam_loss_dice_, sam_loss_mask_, ... = self._compute(sam_pred_masks, sam_gt_masks)
        
        # 累积损失
        loss_dice += loss_dice_ * mask_cnt
        loss_mask += loss_mask_ * mask_cnt
        # ...
    
    # 平均损失
    return {
        'loss_mask': loss_mask / mask_cnts,
        'loss_dice': loss_dice / mask_cnts,
        # ...
    }
```

---

## 2.4 配置文件理解

配置文件是 F-LMM 训练的核心，定义了模型、数据、优化器、Hook 等所有训练相关的设置。

**重点配置文件**:
- `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`

配置文件通常分为 5 个主要部分：
1. **Settings** - 基础训练参数
2. **Model & Tokenizer & Image Processor** - 模型配置
3. **Dataset & Dataloader** - 数据配置
4. **Scheduler & Optimizer** - 优化配置
5. **Runtime** - Hook 和其他运行时配置

### PART 1: Settings（基础训练参数）

```python
# Scheduler & Optimizer
batch_size = 64  # per_device，每张 GPU 的 batch size
accumulative_counts = 2  # 梯度累积步数，有效 batch_size = 64 × num_gpus × 2
dataloader_num_workers = 8  # 数据加载进程数
max_epochs = 8  # 最大训练轮数
optim_type = AdamW  # 优化器类型
lr = 1e-4  # 学习率
betas = (0.9, 0.999)  # AdamW 的动量参数
weight_decay = 0.01  # 权重衰减（L2 正则化）
max_norm = 1  # 梯度裁剪阈值
warmup_ratio = 0.03  # Warmup 比例（3% 的 epochs 用于 warmup）

# Save
save_steps = 100  # 每 100 个 iteration 保存一次 checkpoint
save_total_limit = 1  # 最多保留 1 个 checkpoint（-1 表示不限制）
```

**关键概念**:
- **有效 Batch Size**: `batch_size × num_gpus × accumulative_counts`
  - 例如: 64 × 2 GPU × 2 = 256
- **梯度累积**: 在显存有限时，累积多个小 batch 的梯度后再更新参数
- **Warmup**: 训练初期逐步增加学习率，有助于稳定训练

### PART 2: Model & Tokenizer & Image Processor（模型配置）

#### 2.1 Prompt Template（提示模板）

```python
prompt_template = dict(
    SYSTEM='',  # 系统提示（此配置为空）
    INSTRUCTION='User: {input}\n\nAssistant:',  # 用户输入模板
    SUFFIX='<｜end▁of▁sentence｜>',  # 结束符
    SUFFIX_AS_EOS=True,  # 将 suffix 作为 EOS token
    SEP='\n',  # 分隔符
    STOP_WORDS=['<｜end▁of▁sentence｜>']  # 停止词
)
prompt = '<image_placeholder>'*576 + "Please give me a description of the image."
```

**说明**:
- `image_placeholder` 重复 576 次，对应 24×24=576 个图像 patch
- Prompt 会被插入到用户输入中

#### 2.2 U-Net 配置（Mask Head）

```python
unet = dict(
    type=UNetHead,
    normalize_input=True,  # 归一化输入
    upsample_input=64,  # 将 24×24 的注意力图上采样到 64×64
    in_channels=2048,  # 输入通道数（注意力层数 × 头数）
    base_channels=64,  # 基础通道数
    num_stages=4,  # U-Net 的 stage 数量
    strides=(1, 1, 1, 1),  # 每个 stage 的步长
    enc_num_convs=(2, 2, 2, 2),  # 编码器每层卷积数
    dec_num_convs=(2, 2, 2),  # 解码器每层卷积数
    downsamples=(True, True, True),  # 是否下采样
    enc_dilations=(1, 1, 1, 1),  # 编码器扩张率
    dec_dilations=(1, 1, 1),  # 解码器扩张率
    norm_cfg=dict(type=GroupNorm, num_groups=1),  # 归一化配置
    upsample_cfg=dict(type=InterpConv)  # 上采样配置
)
```

**U-Net 架构**:
- **输入**: 注意力图 `[num_masks, 2048, 24, 24]`
- **输出**: Mask logits `[num_masks, H, W]`（例如 `[num_masks, 64, 64]`）
- **作用**: 将多层的注意力图转换为分割 mask

#### 2.3 损失函数配置

```python
loss_mask = dict(
    type=CrossEntropyLoss,
    use_sigmoid=True,  # 使用 sigmoid + BCE Loss
    reduction='mean',
    loss_weight=1.0)

loss_dice = dict(
    type=DiceLoss,
    use_sigmoid=True,
    activate=True,  # 自动应用 sigmoid
    reduction='mean',
    naive_dice=True,  # 使用朴素 Dice 公式
    eps=1.0,  # 平滑项
    loss_weight=1.0)
```

**损失说明**:
- **CrossEntropyLoss**: 像素级分类损失，关注每个像素的预测
- **DiceLoss**: 区域重叠损失，对类别不平衡友好
- 两种损失组合使用，兼顾像素精度和区域重叠

#### 2.4 Tokenizer 和 Image Processor

```python
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path="deepseek-ai/deepseek-vl-1.3b-chat")

image_processor = dict(
    type=VLMImageProcessor.from_pretrained,
    pretrained_model_name_or_path="deepseek-ai/deepseek-vl-1.3b-chat")
```

**作用**:
- **Tokenizer**: 将文本转换为 token IDs
- **Image Processor**: 将图像预处理为模型输入格式（resize、normalize 等）

#### 2.5 完整模型配置

```python
model = dict(
    type=FrozenDeepseekVLSAM,
    sam=dict(
        type=SAMWrapper,
        use_text=True,  # 使用文本提示
        use_mask=True,  # 使用 mask 提示
        multimask_output=False,  # 单 mask 输出
        model_name='vit_l',  # SAM 模型规模（vit_l/vit_h/vit_b）
        checkpoint='checkpoints/sam_vit_l_0b3195.pth'),  # SAM 权重路径
    model=dict(
        type=MultiModalityCausalLM.from_pretrained,
        pretrained_model_name_or_path="deepseek-ai/deepseek-vl-1.3b-chat",
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
        low_cpu_mem_usage=True),  # 低内存模式
    mask_head=unet,  # U-Net 配置
    tokenizer=tokenizer,
    loss_mask=loss_mask,
    loss_dice=loss_dice,
)
```

**模型组件**:
- **FrozenDeepseekVLSAM**: 主模型类
- **SAM**: SAM 包装器，用于 mask 细化
- **MultiModalityCausalLM**: DeepSeek-VL 基础模型（冻结参数）
- **mask_head**: U-Net，可训练模块

### PART 3: Dataset & Dataloader（数据配置）

#### 3.1 数据集列表配置

```python
image_token = '<image_placeholder>'
backend_args = dict(
    backend='petrel',  # 使用 Petrel 后端（对象存储）
    path_mapping=dict({
        'data/coco/train2014/': 'openmmlab:s3://openmmlab/datasets/detection/coco/train2014/'})
)

# RefCOCO 数据集的 pipeline
refcoco_pipeline = [
    dict(type=PILLoadImageFromFile, backend_args=backend_args),  # 加载图像
    dict(
        type=LoadAnnotations,
        with_mask=True,  # 加载 mask 标注
        with_bbox=False,
        with_seg=False,
        with_label=False),
    dict(
        type=RefCOCO2PNG,
        image_processor=image_processor,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        prompt=prompt,
        image_token=image_token)  # 转换为 PNG 格式
]

# 数据集列表（多个数据集会被合并）
datasets_list = [
    # PNG 数据集（COCO Panoptic）
    dict(
        type=PNGDataset,
        json_file='data/coco/annotations/png_coco_train2017.json',
        panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
        panoptic_png_path='data/coco/annotations/panoptic_train2017',
        tokenizer=tokenizer,
        image_processor=image_processor,
        prompt_template=prompt_template,
        local_path='data/coco/train2017',
        ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2017',
        prompt=prompt,
        image_token=image_token),
    
    # RefCOCO 数据集
    dict(
        type=RefCocoDataset,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        pipeline=refcoco_pipeline,
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p'),
    
    # RefCOCO+ 数据集
    dict(
        type=RefCocoDataset,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        pipeline=refcoco_pipeline,
        ann_file='refcoco+/instances.json',
        split_file='refcoco+/refs(unc).p'),
    
    # RefCOCOg 数据集
    dict(
        type=RefCocoDataset,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        pipeline=refcoco_pipeline,
        ann_file='refcocog/instances.json',
        split_file='refcocog/refs(umd).p')
]
```

**数据集说明**:
- **PNGDataset**: Panoptic Narrative Grounding 数据集，将文本描述与分割区域关联
- **RefCocoDataset**: Referring Expression 数据集，通过文本描述定位图像区域
- **合并策略**: 使用 `concat_datasets` 将多个数据集合并成一个

#### 3.2 DataLoader 配置

```python
train_dataloader = dict(
    batch_size=batch_size,  # 64
    num_workers=dataloader_num_workers,  # 8
    persistent_workers=True,  # 保持 worker 进程活跃（加速数据加载）
    pin_memory=True,  # 将数据固定在内存中（加速 GPU 传输）
    prefetch_factor=2,  # 预取 batch 数量
    dataset=dict(
        type=concat_datasets,  # 使用 concat_datasets 合并多个数据集
        datasets_list=datasets_list),
    sampler=dict(
        type=DefaultSampler,
        shuffle=True),  # 随机打乱
    collate_fn=dict(type=custom_collate_fn)  # 自定义 batch 组装函数
)
```

**关键配置**:
- **persistent_workers**: 保持 worker 进程，避免重复创建（加速数据加载）
- **pin_memory**: 固定内存，加速 CPU → GPU 数据传输
- **custom_collate_fn**: 自定义 batch 组装，处理变长序列和多个 mask

### PART 4: Scheduler & Optimizer（优化配置）

#### 4.1 Optimizer Wrapper 配置

```python
optim_wrapper = dict(
    type=AmpOptimWrapper,  # 混合精度训练（Automatic Mixed Precision）
    optimizer=dict(
        type=AdamW,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    clip_grad=dict(
        max_norm=1,  # 梯度裁剪阈值
        error_if_nonfinite=False),  # 遇到非有限梯度不报错
    accumulative_counts=2,  # 梯度累积步数
    loss_scale='dynamic',  # 动态损失缩放（防止 FP16 下溢）
    dtype='bfloat16')  # 使用 bfloat16 精度
```

**OptimWrapper 说明**:
- **AmpOptimWrapper**: 支持混合精度训练（FP32 + BF16/FP16）
- **clip_grad**: 梯度裁剪，防止梯度爆炸
- **accumulative_counts**: 梯度累积，模拟更大 batch size
- **loss_scale**: 动态缩放损失，防止 FP16 下溢

#### 4.2 Learning Rate Scheduler 配置

```python
param_scheduler = [
    # Warmup 阶段：线性增加到目标学习率
    dict(
        type=LinearLR,
        start_factor=1e-5,  # 初始学习率 = lr × start_factor = 1e-9
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,  # 0.03 × 8 = 0.24 epochs
        convert_to_iter_based=True),  # 转换为基于 iteration
    
    # Cosine 衰减阶段：余弦退火衰减到 0
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,  # 最小学习率
        by_epoch=True,
        begin=warmup_ratio * max_epochs,  # 从 0.24 epochs 开始
        end=max_epochs,  # 到 8 epochs 结束
        convert_to_iter_based=True)
]
```

**学习率调度策略**:
- **Warmup**: 前 3% epochs 线性增加到 `lr=1e-4`
- **Cosine Annealing**: 剩余 epochs 余弦退火到 0
- **convert_to_iter_based=True**: 基于 iteration 更新（而非 epoch）

**学习率曲线示例**:
```
lr
↑
1e-4 ┤           ╭─────────────
     │          ╱
     │         ╱
     │        ╱
     │       ╱
1e-9 ┤──────╯
     └──────────────────────────→ epoch
     0    0.24                   8
    Warmup   Cosine Annealing
```

#### 4.3 Train Loop 配置

```python
train_cfg = dict(type=TrainLoop, max_epochs=8)
```

**说明**: 使用 `TrainLoop` 管理训练循环，最多训练 8 个 epochs

### PART 5: Runtime（运行时配置）

#### 5.1 Default Hooks 配置

```python
default_hooks = dict(
    # 计时 Hook：记录每次 iteration 的时间
    timer=dict(type=IterTimerHook),
    
    # 日志 Hook：每 10 个 iteration 打印一次日志
    logger=dict(
        type=LoggerHook,
        log_metric_by_epoch=False,  # 按 iteration 记录
        interval=10),  # 每 10 个 iteration 打印一次
    
    # 学习率调度 Hook：更新学习率
    param_scheduler=dict(type=ParamSchedulerHook),
    
    # Checkpoint Hook：每 100 个 iteration 保存一次 checkpoint
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,  # 按 iteration 保存（而非 epoch）
        interval=save_steps,  # 100
        max_keep_ckpts=save_total_limit),  # 最多保留 1 个
    
    # 分布式采样器种子 Hook：设置分布式训练的随机种子
    sampler_seed=dict(type=DistSamplerSeedHook),
)
```

**Hook 说明**:
- **IterTimerHook**: 记录训练时间（用于监控性能）
- **LoggerHook**: 打印训练指标（loss、accuracy、IoU 等）
- **ParamSchedulerHook**: 更新学习率（调用 `param_scheduler`）
- **CheckpointHook**: 保存检查点（每 100 个 iteration）
- **DistSamplerSeedHook**: 同步分布式训练的随机种子

#### 5.2 环境配置

```python
env_cfg = dict(
    # 是否启用 cuDNN benchmark（加速卷积运算）
    cudnn_benchmark=False,
    
    # 多进程配置
    mp_cfg=dict(
        mp_start_method='fork',  # 进程启动方法
        opencv_num_threads=0),  # OpenCV 线程数
    
    # 分布式配置
    dist_cfg=dict(backend='nccl'))  # NCCL 后端（NVIDIA GPU）
```

#### 5.3 其他运行时配置

```python
# 可视化器（未使用）
visualizer = None

# 日志级别
log_level = 'INFO'

# 加载 checkpoint 路径（用于恢复训练）
load_from = '/data/gyk/F-LMM/work_dirs/.../iter_100.pth'

# 是否恢复训练（True=恢复训练状态，False=只加载权重）
resume = True

# 随机性配置
randomness = dict(
    seed=None,  # 随机种子（None=使用系统随机）
    deterministic=False)  # 是否使用确定性模式

# 日志处理器配置
log_processor = dict(by_epoch=False)  # 按 iteration 处理日志
```

**关键配置**:
- **load_from**: 指定 checkpoint 路径
- **resume**: 
  - `True`: 恢复训练状态（optimizer、scheduler、iteration count）
  - `False`: 只加载模型权重，从头开始训练
- **randomness**: 控制随机性（用于可重复性实验）

### 配置文件使用流程

1. **加载配置**:
   ```python
   from mmengine.config import Config
   cfg = Config.fromfile('configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py')
   ```

2. **构建组件**:
   ```python
   from xtuner.registry import BUILDER
   model = BUILDER.build(cfg.model)
   train_dataloader = BUILDER.build(cfg.train_dataloader)
   optim_wrapper = BUILDER.build(cfg.optim_wrapper)
   ```

3. **启动训练**:
   ```bash
   xtuner train config.py --deepspeed deepspeed_zero2
   ```

### 配置文件关键要点总结

| 部分 | 核心配置 | 作用 |
|------|---------|------|
| **Settings** | batch_size, lr, max_epochs | 基础训练参数 |
| **Model** | model, mask_head, loss_mask | 模型架构和损失 |
| **Dataset** | datasets_list, train_dataloader | 数据加载 |
| **Optimizer** | optim_wrapper, param_scheduler | 优化策略 |
| **Runtime** | default_hooks, env_cfg | Hook 和环境 |

---

## 3. Hook 系统

### 3.1 Hook 概述

**Hook** 是一种回调机制，允许在训练的不同阶段插入自定义逻辑。

**20 个 Hook 插入点**:
- **全局**: `before_run`, `after_run`
- **训练**: `before_train`, `before_train_epoch`, `before_train_iter`, `after_train_iter`, `after_train_epoch`, `after_train`
- **验证**: `before_val`, `before_val_epoch`, `before_val_iter`, `after_val_iter`, `after_val_epoch`, `after_val`
- **测试**: `before_test`, `before_test_epoch`, `before_test_iter`, `after_test_iter`, `after_test_epoch`, `after_test`
- **检查点**: `before_save_checkpoint`, `after_save_checkpoint`

### 3.2 F-LMM 使用的默认 Hook

#### IterTimerHook - 计时

```python
timer=dict(type=IterTimerHook)
```

**作用**: 记录每次 iteration 的时间  
**输出**: `data_time`, `time`（数据加载时间，总时间）

#### LoggerHook - 日志记录

```python
logger=dict(
    type=LoggerHook, 
    log_metric_by_epoch=False,  # 按 iteration 记录，不是按 epoch
    interval=10)  # 每 10 个 iteration 打印一次
```

**作用**: 
- 打印训练指标（loss, accuracy, IoU 等）
- 记录到日志文件
- 更新 tensorboard/wandb（如果配置）

**输出示例**:
```
2025-11-03 11:03:30 - Iter(train) [10/190157]  loss_mask: 0.234  loss_dice: 0.567  accuracy: 0.89  aiou: 0.76  lr: 1e-04  data_time: 0.12  time: 1.23
```

#### ParamSchedulerHook - 学习率调度

```python
param_scheduler=dict(type=ParamSchedulerHook)
```

**作用**: 
- 在每个 iteration/epoch 后更新学习率
- 调用配置的 `param_scheduler`（LinearLR + CosineAnnealingLR）

**学习率调度配置**:
```python
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,  # Warmup 阶段
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,  # Cosine 衰减阶段
        convert_to_iter_based=True)
]
```

#### CheckpointHook - 检查点保存

```python
checkpoint=dict(
    type=CheckpointHook,
    by_epoch=False,        # 按 iteration 保存
    interval=save_steps,   # 每 100 个 iteration 保存一次
    max_keep_ckpts=1)      # 最多保留 1 个 checkpoint
```

**作用**:
- 定期保存检查点
- 管理检查点数量（删除旧的）
- 调用 `runner.save_checkpoint()` 方法

**保存时机**:
- `by_epoch=False, interval=100`: 每 100 个 iteration 保存一次
- `by_epoch=True, interval=1`: 每个 epoch 保存一次

**保存内容**:
- 模型权重（可训练参数）
- 优化器状态
- 学习率调度器状态
- 元数据（epoch, iter, config 等）

#### DistSamplerSeedHook - 分布式采样器种子

```python
sampler_seed=dict(type=DistSamplerSeedHook)
```

**作用**: 在分布式训练中设置采样器的随机种子，确保不同进程的数据顺序一致

### 3.3 Hook 调用顺序示例

假设当前是第 100 个 iteration：

```
1. before_train_iter()
   ↓
2. 数据加载
   ↓
3. 前向传播 + 反向传播
   ↓
4. after_train_iter()
   ├─ IterTimerHook: 记录时间
   ├─ LoggerHook: 每 10 个 iteration 打印日志（iteration 100）
   ├─ ParamSchedulerHook: 更新学习率
   └─ CheckpointHook: 每 100 个 iteration 保存检查点（iteration 100）
      ├─ before_save_checkpoint()
      ├─ runner.save_checkpoint()
      └─ after_save_checkpoint()
```

### 3.4 自定义 Hook

如果需要自定义 Hook，可以：

```python
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class MyCustomHook(Hook):
    def __init__(self, interval=100):
        self.interval = interval
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % self.interval == 0:
            # 自定义逻辑
            runner.logger.info(f'Custom hook at iteration {runner.iter}')

# 在配置文件中使用
custom_hooks = [
    dict(type='MyCustomHook', interval=100)
]
```

---

## 4. 检查点加载和保存机制

### 4.1 检查点结构

#### DeepSpeed ZeRO-2 Checkpoint（训练时）

```
work_dirs/.../iter_4300.pth/
├── mp_rank_00_model_states.pt          # 模型参数（所有 GPU 共享）
├── bf16_zero_pp_rank_0_optim_states.pt # GPU 0 的优化器状态（分片）
└── bf16_zero_pp_rank_1_optim_states.pt # GPU 1 的优化器状态（分片）
```

#### PyTorch Checkpoint（推理时）

```
checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth
├── meta: dict          # 元数据（epoch, iter, config 等）
├── state_dict: dict    # 模型权重（只包含可训练参数）
└── optimizer: dict     # 优化器状态（如果有）
```

### 4.2 检查点加载流程

#### 自动恢复（推荐）

```python
# 配置文件
load_from = None
resume = True

# Runner 自动执行
resume_from = find_latest_checkpoint(work_dir)  # 找到最新的 checkpoint
runner.resume(resume_from)  # 恢复训练状态
```

**恢复内容**:
- ✅ 模型权重
- ✅ 优化器状态
- ✅ 学习率调度器状态
- ✅ Epoch 和 Iteration 计数
- ✅ 随机数生成器状态

#### 从指定检查点恢复

```python
# 配置文件
load_from = 'work_dirs/.../iter_4300.pth'
resume = True

# Runner 执行
runner.resume(load_from)  # 从指定 checkpoint 恢复
```

#### 只加载权重

```python
# 配置文件
load_from = 'work_dirs/.../iter_4300.pth'
resume = False

# Runner 执行
state_dict = guess_load_checkpoint(load_from)  # 自动识别格式
model.load_state_dict(state_dict, strict=False)  # 只加载权重
# 优化器和调度器保持初始状态
```

### 4.3 检查点保存流程

#### 保存时机

```python
# CheckpointHook 在每个 iteration 后检查
if runner.iter % interval == 0:  # interval = save_steps = 100
    runner.save_checkpoint(filename=f'iter_{runner.iter}.pth')
```

#### 保存内容

```python
checkpoint = {
    'meta': {
        'epoch': self.epoch + 1,
        'iter': self.iter + 1,
        'cfg': self.cfg.pretty_text,
        'time': '20251103_110336',
        # ...
    },
    'state_dict': {
        # 只包含 requires_grad=True 的参数
        'mask_head.conv_seg.weight': ...,
        'mask_head.conv_seg.bias': ...,
        'text_proj.weight': ...,
        'text_layer_weights': ...,
        # 不包含 LMM 参数（已被冻结）
    },
    'optimizer': {
        # 优化器状态（AdamW 的 momentum、variance 等）
    },
    'param_schedulers': [
        # 学习率调度器状态
    ],
    'message_hub': {
        # 消息中心状态（日志等）
    }
}
```

#### 检查点管理

```python
# max_keep_ckpts=1 表示最多保留 1 个 checkpoint
# 当保存新的 checkpoint 时，会自动删除旧的

iter_300.pth  # 保存 iter_400.pth 时被删除
iter_400.pth  # 保留
iter_500.pth  # 保存 iter_600.pth 时被删除
iter_600.pth  # 保留
```

### 4.4 `guess_load_checkpoint()` 的作用

```python
from xtuner.model.utils import guess_load_checkpoint

# 自动识别 checkpoint 格式并加载
state_dict = guess_load_checkpoint(checkpoint_path)
```

**支持的格式**:
- DeepSpeed checkpoint 目录（自动合并分片）
- PyTorch checkpoint 文件（直接加载）
- HuggingFace checkpoint（如果兼容）

**使用场景**:
- 从训练 checkpoint 转换为推理 checkpoint
- 加载不同格式的预训练权重

---

## 5. 模型实现对比分析

F-LMM 支持多种 LMM（Large Multimodal Model），每个模型都有其独特的实现方式。理解不同模型的适配模式有助于：
- 理解 F-LMM 的设计哲学（通用接口 + 模型特定处理）
- 学习如何适配新的 LMM
- 了解不同模型的优势和局限性

### 5.1 通用抽象和接口

所有 F-LMM 模型实现都遵循相同的接口：

#### 5.1.1 核心接口

```python
class FrozenXXX(BaseModel):
    def __init__(self, model, mask_head, loss_mask, loss_dice, ...):
        # 1. 初始化冻结的 LMM
        # 2. 构建可训练的 mask_head (U-Net)
        # 3. 注册损失函数
        
    def _forward(self, data_sample):
        # 1. 准备输入（图像 + 文本）
        # 2. 通过冻结的 LMM 提取 hidden states 和 attentions
        # 3. 处理注意力（重塑、分组、合并）
        # 4. U-Net 生成粗粒度 mask
        # 5. SAM 细化 mask（如果有）
        
    def compute_loss(self, data):
        # 遍历 batch，计算损失
        
    def apply_merge(self, x, dim=1):
        # 合并注意力头（mean 或 max）
```

#### 5.1.2 共同的设计模式

1. **冻结策略**: 所有 LMM 参数都被冻结 (`requires_grad_(False)`)
2. **可训练模块**: 
   - `mask_head` (U-Net)
   - `text_proj` (投影层，用于 SAM)
   - `text_layer_weights` (层权重，用于加权融合)
3. **损失函数**: 统一的 `loss_mask` 和 `loss_dice`
4. **SAM 集成**: 所有模型都有 `FrozenXXXSAM` 版本

### 5.2 DeepSeekVL vs LLaVA 详细对比

#### 5.2.1 图像处理器差异

**DeepSeekVL**:
```python
# 使用 VLMImageProcessor
image_processor = VLMImageProcessor.from_pretrained("deepseek-ai/deepseek-vl-1.3b-chat")

# 固定图像 token 数量
self.image_token_idx = self.tokenizer.encode('<image_placeholder>', ...)[-1]
self.clip_shape = 24  # 固定 24×24 = 576 个 patch
```

**LLaVA**:
```python
# 使用 CLIPImageProcessor (通过 LLaVA 模型间接访问)
image_processor = self.llava.get_vision_tower().image_processor

# 动态计算特征图尺寸
self.patch_size = self.llava.config.vision_config.patch_size
llava_h = padded_h // self.patch_size
llava_w = padded_w // self.patch_size
```

**关键差异**:
- **DeepSeekVL**: 固定 token 数量（576），独立图像处理器
- **LLaVA**: 动态尺寸，图像处理器集成在模型中

#### 5.2.2 Tokenizer 差异

**DeepSeekVL**:
```python
# 使用 DeepSeek-VL 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-vl-1.3b-chat")
image_token = '<image_placeholder>'  # 自定义 token
```

**LLaVA**:
```python
# 使用 LLM 的 tokenizer（通常是 Vicuna）
tokenizer = self.llava.get_model().embed_tokens  # 通过模型访问
image_token = IMAGE_TOKEN_INDEX  # 使用特殊 token index
```

**关键差异**:
- **DeepSeekVL**: 自定义 image token 字符串，需要手动编码
- **LLaVA**: 使用标准的 `IMAGE_TOKEN_INDEX` 常量

#### 5.2.3 模型接口差异

**DeepSeekVL**:
```python
# 使用 prepare_inputs_embeds 准备输入
inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
    input_ids=input_ids,
    pixel_values=pixel_values,
    images_seq_mask=images_seq_mask,
    images_emb_mask=images_emb_mask)

# 直接调用 language_model
outputs = self.deepseek_vl.language_model(
    inputs_embeds=inputs_embeds,
    output_hidden_states=True,
    output_attentions=True)
```

**LLaVA**:
```python
# 直接传递输入字典
inputs = dict(
    input_ids=input_ids,
    pixel_values=pixel_values,
    mask_ids=mask_ids,
    labels=labels)

# 模型内部处理图像和文本融合
outputs = self.llava(
    **inputs,
    attention_mask=attention_mask,
    output_hidden_states=True,
    output_attentions=True)

# 从输出中获取图像 token 位置
attentions = [attn[0, ..., outputs['image_to_overwrite'][0]] 
              for attn in outputs.attentions]
```

**关键差异**:
- **DeepSeekVL**: 
  - 显式准备输入嵌入（图像和文本分开处理）
  - 需要手动标记图像 token 位置 (`images_seq_mask`)
  - 固定数量的图像 token（576）
  
- **LLaVA**: 
  - 模型内部自动处理图像和文本融合
  - 从输出中获取图像 token 位置 (`image_to_overwrite`)
  - 动态数量的图像 token

#### 5.2.4 注意力处理差异

**DeepSeekVL**:
```python
# 提取对图像 token 的注意力
attentions = [attn[0, ..., images_seq_mask[0]] for attn in outputs.attentions]

# 重塑为固定空间维度
attentions = [attn.view(*attn.shape[:-1], 24, 24) for attn in attentions]
```

**LLaVA**:
```python
# 提取对图像 token 的注意力
attentions = [attn[0, ..., outputs['image_to_overwrite'][0]] 
              for attn in outputs.attentions]

# 重塑为动态空间维度
padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
llava_h = padded_h // self.patch_size
llava_w = padded_w // self.patch_size
attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
```

**关键差异**:
- **DeepSeekVL**: 固定空间维度（24×24）
- **LLaVA**: 动态空间维度（取决于图像尺寸和 patch size）

### 5.3 其他模型实现

#### 5.3.1 LLaVA-Next

**特点**:
```python
# 处理两种分辨率的注意力（coarse 和 fine）
in_channels = num_heads * num_layers * 2  # 乘以 2

attentions_with_coarse = [...]  # 粗粒度注意力
attentions_with_fine = [...]    # 细粒度注意力

# 将两种分辨率对齐到 fine 分辨率
attention_maps = torch.cat([
    F.interpolate(attentions_with_coarse.float(), 
                  size=(fine_h, fine_w), mode='bilinear'),
    F.interpolate(attentions_with_fine.float(), 
                  size=(fine_h, fine_w), mode='bilinear')
], dim=1)
```

**特殊处理**:
- **双分辨率**: 同时使用 coarse 和 fine 两种分辨率的图像特征
- **对齐操作**: 将两种分辨率对齐到 fine 分辨率，然后拼接

#### 5.3.2 MGM

**特点**:
```python
# 支持 image_grid 和 image_global
image_grid = getattr(self.mgm.config, 'image_grid', 1)  # 例如 2×2
use_global_image = getattr(self.mgm.config, 'image_global', False)

# 处理多网格图像
if image_grid > 1:
    # 将图像分割为多个网格
    # 可选的全局图像特征
    # 重塑注意力以匹配网格布局
```

**特殊处理**:
- **图像网格**: 支持将大图像分割为多个网格（如 2×2=4 个网格）
- **全局图像**: 可选的全局图像特征，用于捕获全局上下文
- **动态 in_channels**: 根据 `image_grid` 和 `image_global` 计算

#### 5.3.3 HPT

**特点**:
```python
# 独立的视觉编码器和投影器
self.visual_encoder = BUILDER.build(visual_encoder)
self.projector = BUILDER.build(projector)
self.llm = BUILDER.build(llm)

# 处理位置编码插值
def interpolate_pos_embed(model, new_size):
    # 调整视觉编码器的位置编码以适配新的图像尺寸
```

**特殊处理**:
- **模块化设计**: 视觉编码器、投影器、LLM 分离
- **位置编码插值**: 支持调整图像尺寸时的位置编码插值
- **自定义视觉编码器**: 可以使用不同的视觉编码器（SigLIP、CLIP 等）

### 5.4 适配新模型的通用流程

要适配一个新的 LMM，需要实现以下步骤：

#### 步骤 1: 创建基础模型类

```python
class FrozenNewLMM(BaseModel):
    def __init__(self, model, mask_head, loss_mask, loss_dice, ...):
        super().__init__()
        # 1. 初始化 LMM（冻结参数）
        self.new_lmm = BUILDER.build(model)
        self.new_lmm.requires_grad_(False)
        
        # 2. 计算 U-Net 输入通道数
        in_channels = (num_heads * num_layers)
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        
        # 3. 注册损失函数
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
```

#### 步骤 2: 实现 `_forward()` 方法

```python
def _forward(self, data_sample):
    # 1. 准备输入（模型特定）
    inputs = self._prepare_inputs(data_sample)
    
    # 2. 通过 LMM 提取表示
    with torch.no_grad():
        outputs = self.new_lmm(**inputs,
                               output_hidden_states=True,
                               output_attentions=True)
    
    # 3. 提取和处理注意力（模型特定）
    attentions = self._extract_attentions(outputs)
    attentions = self._reshape_attentions(attentions)
    
    # 4. 提取 hidden states（模型特定）
    hidden_states = self._extract_hidden_states(outputs)
    hidden_states = self._weighted_fusion(hidden_states)
    
    # 5. 按 mask 分组
    mask_attentions, text_embeds = self._group_by_mask(
        attentions, hidden_states, mask_ids)
    
    # 6. U-Net 生成 mask
    pred_masks = self.mask_head(mask_attentions)[:, 0]
    
    # 7. SAM 细化（可选）
    if hasattr(self, 'sam'):
        sam_pred_masks = self.sam(image, pred_masks, text_embeds)
    
    return dict(pred_masks=pred_masks, sam_pred_masks=sam_pred_masks)
```

#### 步骤 3: 实现模型特定的辅助方法

```python
def _prepare_inputs(self, data_sample):
    """准备模型输入（每个模型不同）"""
    # DeepSeekVL: prepare_inputs_embeds
    # LLaVA: 直接传递 input_ids, pixel_values
    # MGM: prepare_inputs_labels_for_multimodal
    pass

def _extract_attentions(self, outputs):
    """提取注意力权重（每个模型不同）"""
    # DeepSeekVL: outputs.attentions, 使用 images_seq_mask
    # LLaVA: outputs.attentions, 使用 image_to_overwrite
    # MGM: outputs.attentions, 使用 image_places
    pass

def _reshape_attentions(self, attentions):
    """重塑注意力到空间维度"""
    # 固定尺寸: view(..., H, W)
    # 动态尺寸: 根据 patch_size 和图像尺寸计算
    pass
```

#### 步骤 4: 创建 SAM 版本

```python
class FrozenNewLMMSAM(FrozenNewLMM):
    def __init__(self, sam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        # 投影 hidden_size 到 SAM 的 embed_dim
        self.text_proj = nn.Linear(
            self.new_lmm.config.hidden_size,
            self.sam.model.prompt_encoder.embed_dim)
```

### 5.5 关键设计决策对比表

| 特性 | DeepSeekVL | LLaVA | LLaVA-Next | MGM | HPT |
|------|-----------|-------|------------|-----|-----|
| **图像 Token 数量** | 固定 (576) | 动态 | 动态 (双分辨率) | 动态 (可网格) | 动态 |
| **输入准备** | `prepare_inputs_embeds` | 直接传递 dict | 直接传递 dict | `prepare_inputs_labels_for_multimodal` | `prepare_inputs_labels_for_multimodal` |
| **图像处理器** | `VLMImageProcessor` | `CLIPImageProcessor` | `CLIPImageProcessor` | 内置 | `SigLIP/CLIP` |
| **注意力提取** | `images_seq_mask` | `image_to_overwrite` | `image_to_overwrite` | `image_places` | `image_places` |
| **空间维度** | 固定 24×24 | 动态 H×W | 动态 (双分辨率) | 动态 (可网格) | 动态 |
| **特殊处理** | 固定 token | 标准处理 | 双分辨率对齐 | 网格分割 | 位置编码插值 |

### 5.6 适配新模型的检查清单

- [ ] 实现基础模型类（继承 `BaseModel`）
- [ ] 冻结 LMM 参数
- [ ] 计算并配置 `mask_head` 的 `in_channels`
- [ ] 实现 `_forward()` 方法
- [ ] 实现 `_prepare_inputs()`（模型特定）
- [ ] 实现 `_extract_attentions()`（模型特定）
- [ ] 实现 `_reshape_attentions()`（固定或动态尺寸）
- [ ] 实现 `compute_loss()`（通常可复用）
- [ ] 实现 `apply_merge()`（通常可复用）
- [ ] 创建 SAM 版本（可选）
- [ ] 添加 `text_layer_weights`（可选，用于加权融合）
- [ ] 处理 padding 移除（如果需要）
- [ ] 编写配置文件示例

---

## 6. DeepSpeed 集成

### 6.1 DeepSpeed ZeRO-2

DeepSpeed ZeRO-2 用于：
- **参数分片**: 将优化器状态分片到多个 GPU
- **显存优化**: 大幅减少显存占用
- **梯度同步**: 自动处理分布式训练的梯度同步

### 6.2 训练命令

```bash
NPROC_PER_NODE=2 xtuner train config.py --deepspeed deepspeed_zero2
```

**内部流程**:
1. Xtuner 检测到 `--deepspeed` 参数
2. 初始化 DeepSpeed 引擎
3. 包装模型和优化器
4. 使用 DeepSpeed 的训练循环

### 6.3 DeepSpeed 与 Runner 的交互

```python
# Runner 创建时
runner = CustomRunner(...)
# Xtuner 会自动检测 DeepSpeed 并包装模型

# 训练时
with torch.no_grad():
    outputs = model(...)  # DeepSpeed 处理分布式

# 保存时
runner.save_checkpoint(...)  
# DeepSpeed 自动保存分片的 checkpoint
```

### 6.4 DeepSpeed Checkpoint 转换

```python
# 训练时保存的是 DeepSpeed 格式
# iter_4300.pth/ (目录，包含分片文件)

# 转换为推理格式
python scripts/deepspeed2torch_state_dict.py \
    --deepspeed_path iter_4300.pth \
    --torch_path checkpoint.pth

# 结果: 单个 .pth 文件，包含合并后的模型权重
```

---

## 训练流程完整图

```
启动训练
  ↓
Runner 初始化
  ├─ 加载配置
  ├─ 构建模型
  ├─ 构建数据加载器
  ├─ 构建优化器和调度器
  └─ 注册 Hook
  ↓
load_or_resume()
  ├─ 检查是否有 checkpoint
  ├─ 如果有且 resume=True: 恢复训练状态
  └─ 如果有且 resume=False: 只加载权重
  ↓
before_run() Hook
  ↓
before_train() Hook
  ↓
TrainLoop 开始
  ↓
for epoch in range(max_epochs):
    ↓
    before_train_epoch() Hook
      ↓
    for iteration, data_batch in train_dataloader:
        ↓
        before_train_iter() Hook
          ↓
        # 1. 数据加载（已在循环中完成）
          ↓
        # 2. 模型前向传播
        loss_dict = model.train_step(data_batch, optim_wrapper)
          ├─ model._forward(data_batch)  # 提取注意力，生成 mask
          ├─ model.compute_loss(data_batch)  # 计算损失
          └─ loss.backward() + optim_wrapper.step()  # 反向传播和更新
          ↓
        after_train_iter() Hook
          ├─ IterTimerHook: 记录时间
          ├─ LoggerHook: 打印日志（每 10 iter）
          ├─ ParamSchedulerHook: 更新学习率
          └─ CheckpointHook: 保存检查点（每 100 iter）
             ├─ runner.save_checkpoint()
             ├─ 只保存可训练参数
             ├─ 保存优化器和调度器状态
             └─ 管理检查点数量
    ↓
    after_train_epoch() Hook
  ↓
after_train() Hook
  ↓
after_run() Hook
  ↓
训练结束
```

---

## 关键设计特点

### 1. 轻量级检查点

**只保存可训练参数**:
```python
model_parameters = {
    k: v.detach() 
    for k, v in model.named_parameters() 
    if v.requires_grad  # LMM 参数被冻结，不需要保存
}
```

**优势**:
- 检查点文件小（只有 U-Net + 投影层，~24MB，vs 完整模型可能数 GB）
- 加载速度快
- 不保存不必要的冻结参数

### 2. Hook 驱动的设计

**解耦训练逻辑**:
- 训练循环（TrainLoop）负责流程
- Hook 负责具体功能（日志、保存、调度等）
- 易于扩展和定制

### 3. DeepSpeed 无缝集成

**自动处理**:
- 检查点格式转换
- 分布式训练的协调
- 优化器状态的分片和同步

---

## 相关文件

- **CustomRunner**: `flmm/runner.py`
- **TrainLoop**: `xtuner/engine/runner.py` (来自 Xtuner)
- **Hook 基类**: `mmengine/hooks/hook.py` (来自 MMEngine)
- **配置示例**: `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`

---

## 参考

- MMEngine Runner: [MMEngine Runner Documentation](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html)
- MMEngine Hook: [MMEngine Hook Tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/hooks.html)
- DeepSpeed: [DeepSpeed Documentation](https://www.deepspeed.ai/)
- Xtuner: [Xtuner GitHub](https://github.com/InternLM/xtuner)

