# 可解释框架文档入口

这一组文档只关心你现在这条线：

- `Qwen2.5-VL + F-LMM grounding`
- `ask -> answer -> phrase -> ground -> ROI re-answer`
- `token-to-region`
- `region-to-token`
- `stability_eval`

命名约定已经统一：

- 只使用 `token-to-region`
- 只使用 `region-to-token`
- 旧命名不再作为对外接口保留

## 建议阅读顺序

1. `DEMO_USAGE_AND_CODE_GUIDE.md`
   - 最实用的一份
   - 讲清楚 demo 怎么跑、结果写到哪里、代码主流程怎么走
2. `ATTENTION_DIRECTIONS.md`
   - 讲清楚 attention 的方向、`images_seq_mask`、为什么当前 `region-to-token` 是对同一块 `answer-to-image` attention 的列聚合
3. `STABILITY_MANIFEST_GUIDE.md`
   - 讲清楚固定样例集怎么写
4. `ROADMAP.md`
   - 讲清楚课题分阶段怎么推进

## 最常用命令

单卡交互式 demo：

```bash
cd /home/hechen/gyk/F-LMM

CUDA_VISIBLE_DEVICES=0 /home/hechen/miniconda3/envs/flmm-qwen/bin/python \
  scripts/demo/interact.py \
  configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --device cuda:0 \
  --device-map none \
  --image data/custom/shampoo_room.png
```

稳定性评测：

```bash
cd /home/hechen/gyk/F-LMM

CUDA_VISIBLE_DEVICES=0 /home/hechen/miniconda3/envs/flmm-qwen/bin/python \
  scripts/demo/stability_eval.py \
  --manifest scripts/demo/manifests/stability_cases.v1.json \
  --config configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --run-name phase1_v1_full_single_gpu \
  --device cuda:0 \
  --device-map none \
  --enable-roi
```

`token-to-region`：

```bash
cd /home/hechen/gyk/F-LMM

CUDA_VISIBLE_DEVICES=0 /home/hechen/miniconda3/envs/flmm-qwen/bin/python \
  scripts/demo/token_to_region_demo.py \
  configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --device cuda:0 \
  --device-map none \
  --image data/custom/shampoo_room.png \
  --prompt "Where is the shampoo?" \
  --token-start 7 \
  --token-end 8
```

`region-to-token`：

```bash
cd /home/hechen/gyk/F-LMM

CUDA_VISIBLE_DEVICES=0 /home/hechen/miniconda3/envs/flmm-qwen/bin/python \
  scripts/demo/region_to_token_demo.py \
  configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --device cuda:0 \
  --device-map none \
  --image data/custom/shampoo_room.png \
  --prompt "Where is the shampoo?" \
  --bbox 112 165 150 247 \
  --topk 8
```

## 结果一般写到哪里

- 交互式 demo：`scripts/demo/results/qwen`
- 稳定性评测：`scripts/demo/results/stability/<run_name>`
- 单独的 `token-to-region` demo：`scripts/demo/results/token_to_region`
- 单独的 `region-to-token` demo：`scripts/demo/results/region_to_token`
- Web backend / worker：`results/`

如果你现在只想快速搞清楚“怎么跑”和“代码怎么走”，直接看：

- `doc/06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`
