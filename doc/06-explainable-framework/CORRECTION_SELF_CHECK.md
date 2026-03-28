# Correction Self-Check

这个文档对应任务三当前可落地的最小工程链路：

- 不依赖前端
- 不假设已经有完整 human-in-the-loop UI
- 先把 baseline / auto_correction / human_override 三套结果格式和自检节奏固定下来

入口脚本：

- `scripts/demo/correction_selfcheck.py`

## 1. 输入来源

当前 correction 自检不直接重跑模型，而是消费第一阶段稳定性报告：

```bash
python scripts/demo/stability_eval.py \
  --manifest scripts/demo/manifests/stability_cases.v1.json \
  --report-dir scripts/demo/results/stability
```

得到的 `report.json` 里已经有：

- `raw_answer / answer / roi_answer`
- phrase 候选及 concept 字段
- grounding records
- 自动失败标签

这些信息足够先把“concept 粒度”的纠错判定层固定下来。

## 2. 生成三套快照

```bash
python scripts/demo/correction_selfcheck.py \
  --report scripts/demo/results/stability/<run>/report.json \
  --output-dir scripts/demo/results/correction
```

输出目录里会生成：

- `baseline.json`
- `auto_correction.json`
- `human_override.template.jsonl`
- `selfcheck.json`

如果再提供人工覆盖文件：

```bash
python scripts/demo/correction_selfcheck.py \
  --report scripts/demo/results/stability/<run>/report.json \
  --overrides scripts/demo/results/correction/<run>/human_override.filled.jsonl \
  --output-dir scripts/demo/results/correction
```

还会额外生成：

- `human_override.json`

## 3. 三套结果各自表示什么

### baseline

保留当前系统原样输出，但把记录统一成 concept 粒度：

- `mention_text`
- `concept_text`
- `answer_span`
- `roi_bbox`
- `judge_label`
- `judge_reason`
- `judge_source=baseline`

### auto_correction

当前只做“可确定的、结构化的自动建议”，不冒充完整模型纠错。

现阶段自动规则只做两类事：

1. grounding 缺失时直接 `reject`
2. mention 和 concept 不一致时，给出 `repair`

也就是说它更像 correction layer 的第一版裁决器，而不是最终修复器。

### human_override

由 `human_override.template.jsonl` 填写后合并生成。

当前是文件式 human-in-the-loop，占位未来前端 UI：

- 一行对应一个 concept 记录
- 可填 `judge_label`
- 可填 `corrected_concept_text`
- 可填 `corrected_answer_span`
- 可填 `corrected_bbox`

## 4. 自检规则

`selfcheck.json` 至少会统计：

- 每种模式的样例数 / concept 数
- `keep/reject/repair` 计数
- 失败计数：
  - `detection_failure`
  - `judgment_failure`
  - `repair_failure`
  - `human_override_failure`

当前判定标准：

- 没有 `roi_bbox` 视为 detection failure
- 没有有效 `answer_span` 或 `judge_label` 非法，视为 judgment failure
- `judge_label=repair` 但没有任何修复载荷，视为 repair failure
- human override 指向不存在的 concept 或 label 非法，视为 human override failure

## 5. 为什么先做这个

在前端还不稳定时，先把 concept 级别的数据结构、自检输出和人工覆盖文件格式固定下来，后面接 UI 就不会再从零重想一遍状态模型。

这一步的目标不是“证明纠错已经有效”，而是先保证：

1. baseline 能追溯到 answer span 和 ROI
2. auto/human 两条链路有统一结果格式
3. 每次改动后可以快速知道是检测、判定还是修复层坏了
