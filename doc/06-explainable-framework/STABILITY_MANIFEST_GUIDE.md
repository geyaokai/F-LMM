# Stability Manifest 制定指南

这份 manifest 不是数据集标注文件，而是第一阶段“固定失败样例集”的实验入口。

目标不是追求大规模，而是让后续每次改动都能在同一批样例上复现、比较。

如果你还不清楚稳定性脚本怎么跑，先看：

- `doc/06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`

## 1. 先定样例，不先定格式

建议先从你已经见过的不稳定案例里挑，而不是随机抽图。

第一批建议 10 到 15 条，优先覆盖这 5 类：

1. phrase 抽取失败
2. grounding 为空或框错
3. ROI 重答把答案带偏
4. 多实例同类物体，模型容易指错
5. 多轮对话里代词或省略导致目标漂移

等第一批跑通后，再扩展到 30 到 50 条。

## 2. 一条样例是什么

一条样例对应一次完整 QA，不是“一张图”。

同一张图可以出现多条样例，例如：
- 问颜色
- 问位置关系
- 问手里拿的物体
- 第二轮继续问 “它旁边的那个是什么”

这样更适合测你现在的系统，因为你的问题主要出在“回答链路”和“多轮上下文”，不只是图像本身。

## 3. 每条样例最少要填什么

最低可运行字段只有这些：

```json
{
  "id": "case_001",
  "image_path": "/abs/path/to/image.jpg",
  "question": "What is the person holding?"
}
```

但这样只能跑，不能有效定位问题。

## 4. 强烈建议填的字段

### `answer_contains`
写你认为正确答案里应该出现的关键词。

例子：

```json
"answer_contains": ["umbrella"]
```

作用：
- 自动打 `answer_missing_required_text`

### `answer_not_contains`
写明显不该出现的词。

例子：

```json
"answer_not_contains": ["bag", "suitcase"]
```

作用：
- 自动打 `answer_contains_forbidden_text`

### `expected_bbox`
如果你希望系统用 ROI 重答或 ground 纠错，这个字段很有价值。

它表示：
- 对当前问题来说，最关键的证据区域大致在哪里
- 坐标必须是原图坐标系下的 `[x1, y1, x2, y2]`

例子：

```json
"expected_bbox": [412, 188, 603, 471]
```

作用：
- 自动计算首个 predicted bbox 的 IoU
- 自动打 `bbox_low_iou`

## 5. 多轮样例怎么写

如果问题依赖前文，把前文写进 `history`。

例子：

```json
"history": [
  {"role": "user", "text": "What is the man holding?"},
  {"role": "assistant", "text": "He is holding a tennis racket."}
]
```

当前轮再问：

```json
"question": "What color is it?"
```

这样能专门测“代词接续”和“上下文漂移”。

## 6. `expected_bbox` 该框什么

只框“支持答案所必需的最小关键区域”，不要框太大。

例如问题是：
- “What is the man holding?”

更合适的框：
- 手里的球拍

不合适的框：
- 整个人
- 半张图

因为你后面要评估的是 grounding 是否真的找到了证据，而不是有没有碰到目标附近。

## 7. 建议的样例构成

第一批 15 条可以这样分：

1. 4 条单轮、实体明确、应当很容易对
2. 3 条多实例同类目标，如两个杯子、两个人、两辆车
3. 3 条细粒度目标，如手中小物体、衣服细节、文字区域
4. 3 条多轮代词样例
5. 2 条你已经确认会失败的“硬例”

这样第一轮报告会比较有信息量，不会全是简单样例。

## 8. 命名建议

`id` 建议编码进失败模式和场景，后面查报告更快。

例如：

```json
"id": "phrase_fail_hat_001"
"id": "multiturn_pronoun_racket_002"
"id": "small_object_phone_003"
```

## 9. 一个实用原则

如果你现在时间有限，不要一开始就给每条样例都标 `expected_bbox`。

优先顺序建议是：

1. 先把 `image_path`、`question`、`notes` 填完
2. 再给 most important 的 10 条补 `answer_contains`
3. 最后只给最关键的 5 到 10 条补 `expected_bbox`

因为当前第一阶段最重要的是固定失败，不是一次性做完整标注。

## 10. 起步方式

直接复制模板：

- `scripts/demo/manifests/stability_cases.template.json`

复制后建议改名：

- `scripts/demo/manifests/stability_cases.v1.json`

然后先填 10 条，先跑一轮报告，再决定是否补框和补多轮。

## 11. 你的 `where is the shampoo?` 例子怎么写

这张图很适合作为第一批样例：

- 问题是 location-oriented，不只是分类。
- 目标很小，容易暴露 grounding 和 ROI 重答问题。
- 目标在室内复杂背景里，不是特别显眼。

推荐写法：

```json
{
  "id": "custom_shampoo_room_001",
  "image_path": "/home/hechen/gyk/F-LMM/data/custom/shampoo_room.png",
  "question": "Where is the shampoo?",
  "answer_contains": ["dresser"],
  "answer_not_contains": ["bookshelf", "bed"],
  "expected_bbox": [112, 165, 150, 247],
  "enable_roi": true,
  "auto_topk": 1,
  "notes": "Small bottle on the left dresser, near the round mirror. Good for testing small-object grounding."
}
```

几点说明：

1. `answer_contains` 我只建议先写 `dresser`，不要一开始写得太死。
   - 比如同时要求 `left`、`mirror`、`dresser`，会让自动检查过严。
2. `answer_not_contains` 先排掉几个最容易答偏的大物体即可。
3. `expected_bbox` 这里给的是近似框，不需要像检测标注那样特别精细。
   - 只要能稳定覆盖 shampoo 瓶子即可。
4. 如果你后面发现模型经常回答成
   - “on the dresser”
   - “on the left side table”
   - “near the mirror”
   
   那这条样例的自动判定应继续以 `expected_bbox` 为主，而不是把 `answer_contains` 设得过多。

仓库里也补了一个对应示例文件：

- `scripts/demo/manifests/stability_cases.v1.example.json`
