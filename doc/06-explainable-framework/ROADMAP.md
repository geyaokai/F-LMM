# 可解释端到端框架推进计划

本文将当前课题状态、下一阶段实施顺序和已落地的工程入口固定下来，避免路线分散推进。

相关补充说明：

- demo 怎么跑、结果写到哪里、核心代码怎么走：`doc/06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`
- attention 方向、`images_seq_mask`、token-to-region / region-to-token 的技术说明：`doc/06-explainable-framework/ATTENTION_DIRECTIONS.md`

## 1. 当前状态

### 第一层：Grounding 基座
- backbone 已切到 Qwen2.5-VL，F-LMM grounding 流程可跑。
- `answer -> phrase extract -> ground -> ROI re-answer` 已形成闭环，主流程在 `scripts/demo/interact.py`。
- demo 可用，但稳定性不足，问题更可能出在“短语抽取 + 自动 top-k grounding + ROI 重答”整条链，而不只是 mask head。

### 第二层：可解释系统
- `token-to-region` 已具备基础能力：
  - 回答中的 phrase 可触发 bbox / segmentation grounding。
  - worker 已支持 token-to-region，可以从输出 token 回看图像证据。
- `region-to-token` 已打通后端最小闭环：
  - worker 已支持 region-to-token，可从 bbox / grounding record 回看 top-k token / phrase。
  - 当前已输出 `bbox_overlay / region_overlay / ranking.json / meta.json`，前端仍未接入。
  - `external/lvlm-interpret` 仍可作为后续增强参考，但不再是后端落地前提。

### 第三层：纠错与性能提升
- ROI 重答已经落地，是当前唯一成体系的纠错入口。
- 重采样做过尝试，但尚未固化到统一评测框架。
- 提示重构还没有进入实验矩阵。

### VCD 结合
- 当前做法偏“回答后取高亮概念，再去 VCD 查询展示”。
- 后续应改为“grounded region -> concept/attribute -> answer phrase”的 region-centric 结构。

## 2. 推进原则

1. 不同时均匀推进三层，优先形成论文闭环。
2. 先收敛稳定性基线，再补 region-to-token 的评测与前端接入，最后做纠错和概念层。
3. 所有新能力都必须接到统一结果格式和固定样例集上，避免 demo 导向开发。

## 3. 执行顺序

### 阶段 1：稳定性基线
目标：固定 30-50 个样例，明确失败类型，并让后续改动可比较。

交付物：
- 固定 manifest 驱动的样例集。
- 统一报告，记录：
  - 原始回答 / 最终回答
  - phrase 抽取结果
  - grounding 结果与 bbox
  - ROI 重答结果
  - 自动失败标签
- 失败类型先按以下粗分类收敛：
  - `phrase_missing`
  - `ground_missing`
  - `bbox_missing`
  - `bbox_low_iou`
  - `roi_error`
  - `answer_missing_required_text`
  - `answer_contains_forbidden_text`

实现入口：
- `scripts/demo/stability_eval.py`

### 阶段 2：补全双向 grounding
目标：把第二层从“后端最小闭环”推进到“可展示、可评测、可写论文”。

当前状态：
- 已在现有 task queue 上实现 region-to-token。
- 当前输入支持：`bbox` / 最近一次 grounding 的 `record_index`。
- 当前输出包括：
  - top-k 相关 token / phrase
  - layer 汇总信息
  - 可存盘的解释元数据与可视化图

约束：
- 先把结果 JSON 和人工分析链路整理清楚，再决定前端展示粒度。
- 与 token-to-region 保持同一目录结构与元数据风格。

### 阶段 3：纠错实验矩阵
目标：把“解释信号用于校正”从单点技巧变成可对比策略。

统一对比四组：
1. 无纠错
2. ROI 重答
3. 重采样
4. 提示重构

检测信号优先级：
1. grounding 是否为空 / bbox 是否异常
2. token-to-region 是否显示输出 token 依赖图像证据
3. region-to-token 是否显示关键区域真正影响输出 token

### 阶段 4：VCD 与概念层
目标：从“概念展示”升级到“解释单元”。

解释单元统一格式：
- 图像区域
- 视觉概念 / 属性
- 对应 answer span / token
- grounding / attention 证据
- 是否触发纠错

## 4. 立即实施项

### 已完成
- 本文档已落盘到 `doc/06-explainable-framework/ROADMAP.md`。
- manifest 制定指南已新增：`doc/06-explainable-framework/STABILITY_MANIFEST_GUIDE.md`。
- 第一阶段评测脚本已新增：`scripts/demo/stability_eval.py`。
- manifest 模板已新增：`scripts/demo/manifests/stability_cases.template.json`。

### 下一步直接执行
1. 整理第一批固定样例 manifest，优先覆盖：
   - phrase 抽取失败
   - grounding 为空
   - bbox 偏移
   - ROI 重答把答案带偏
   - 多轮上下文导致目标漂移
2. 用稳定性脚本跑出第一版报告。
3. 按报告里最集中的失败类型，决定是先改 phrase 选择，还是先改 ROI 触发策略。
4. 在稳定性基线固定后，再把 region-to-token 接到人工评测链路和前端展示链路。

## 5. 第一阶段样例规范

manifest 每条样例建议包含：

```json
{
  "id": "case_001",
  "image_path": "path/to/image.jpg",
  "question": "What is the person holding?",
  "history": [],
  "enable_roi": true,
  "auto_topk": 1,
  "expected_bbox": [x1, y1, x2, y2],
  "expectations": {
    "answer_contains": ["umbrella"],
    "answer_not_contains": ["bag"]
  },
  "notes": "short comment"
}
```

说明：
- `expected_bbox` 可选，用于自动打 `bbox_low_iou` 标签。
- `history` 可选，用于复现多轮场景。
- `expectations` 可选，用于做轻量自动判错，不替代人工复核。
- 可直接从模板开始填写：`scripts/demo/manifests/stability_cases.template.json`。

## 6. 为什么按这个顺序做

当前最缺的不是再加一个 demo，而是把已有能力变成稳定、可复现、可对比的实验链路。

如果没有固定样例集：
- 反向 grounding 做出来，也很难证明它真的有帮助；
- 纠错策略做出来，也无法知道它是在修正幻觉，还是只是偶然改写了答案；
- VCD 结合也会继续停留在展示层，而不是解释层。

所以先做稳定性基线，是后面两层成立的前提。
