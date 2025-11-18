# Qwen2.5-VL 训练存档策略与恢复指引

该配置同时启用了 **step** 与 **epoch** 两类 CheckpointHook：

- `iter_{iter:06d}.pth`：每 `save_steps`（默认 100）次迭代写一次，便于排查训练中途的数据波动。
- `epoch_{epoch:03d}.pth`：每个 epoch 结束写一次，保证即使调整 `batch_size`/梯度累积，也能继续训练。

## 1. 训练过程中如何保存

配置位于 `configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py` 的 `default_hooks`：

```python
ckpt_step = dict(
    type=CheckpointHook,
    by_epoch=False,
    interval=save_steps,
    filename_tmpl='iter_{iter:06d}.pth',
)
ckpt_epoch = dict(
    type=CheckpointHook,
    by_epoch=True,
    interval=1,
    filename_tmpl='epoch_{epoch:03d}.pth',
)
```

生成的文件保存在 `work_dirs/<exp_name>/<timestamp>/`，可同时观察两类快照。

## 2. 使用 `train.sh` 启动与恢复训练

脚本默认指向 Qwen2.5-VL 配置，常用示例如下：

| 场景 | 命令 |
| --- | --- |
| 正常启动 | `./train.sh` |
| 自定义配置/GPU 数量 | `./train.sh --config configs/qwen/...py --gpus 4` |
| 从指定 checkpoint 继续训练 | `./train.sh --resume --resume-from work_dirs/.../epoch_004.pth` |
| 自动查找最新 checkpoint 续训 | `./train.sh --resume`（脚本会查找 `last_checkpoint` 或最新的 `epoch_*.pth`/`iter_*.pth` 并传给 `--resume`） |
| 指定 work_dir | `./train.sh --work-dir work_dirs/my_exp` |

脚本会将完整日志写到 `logs/` 目录，同时终端默认只输出损失、学习率等关键信息，可用 `--full-log` 查看全部输出。若需要仅加载权重进行评估，请直接调用 `xtuner test` 或自定义评估脚本。

## 3. 恢复训练的推荐做法

| 场景 | 操作建议 |
| --- | --- |
| **训练中断（batch 设置不变）** | 直接 `--resume`，脚本会让 MMEngine 自动加载目录下最新的 checkpoint。 |
| **调整 batch / gradient accumulation 后继续** | 指定某个 `epoch_xxx.pth`：`--resume --resume-from work_dirs/.../epoch_008.pth`，可确保 optimizer / scheduler 同步。 |
| **只想做评估 / 可视化** | 仅 `--resume-from`，不加 `--resume`，只加载模型权重。 |

> 提示：epoch checkpoint 不依赖步数，因此更适合跨配置续训；step checkpoint 则方便分析训练过程的瞬态状态。

## 4. 常见问题解答

- **两个 Hook 会相互覆盖吗？** 不会。我们为 step / epoch 配置了不同的 `filename_tmpl`，文件名包含 `iter` 或 `epoch` 前缀，可共存。
- **如何限制存档数量？** `max_keep_ckpts` 仍受 `save_total_limit` 控制，若希望 step 与 epoch 各自保留更多，可视情况增大该值。
- **想禁用某一种？** 将对应 Hook 注释或删除即可。例如移除 `ckpt_step`，则只保留 epoch 存档。

## 5. 常用指令示例

```bash
# 使用 train.sh 恢复到第 4 个 epoch 的快照并继续训练
./train.sh --resume --resume-from work_dirs/.../epoch_004.pth

# 仅加载模型权重进行验证（不恢复优化器）
./train.sh --config configs/qwen/...py --resume-from work_dirs/.../epoch_004.pth
```

保持两个层级的 checkpoint 能兼顾细粒度调试与稳定的断点续训，建议在日常训练中保留此配置。
