# Frontend Agent Handoff

本文档面向“另一台机器上的前端开发者 / agent”。

目标不是解释全部模型细节，而是回答下面几个实际问题：

- 后端现在已经有什么能力
- 前端该调哪些接口
- explainable 视图应该走同步接口还是异步任务
- 哪些字段可以直接展示，哪些只是调试信息
- 当前工程离“可上线”还差什么

## 1. 当前结论

当前这套 Web 结构适合：

- 研究型 demo
- 组内演示
- 前后端并行开发
- 后续拆分成独立服务

当前这套 Web 结构暂时不适合：

- 直接公网生产上线
- 高并发多用户
- 多机共享状态

原因很简单：

- session 存在后端进程内存里
- 推理结果和上传图片写本地磁盘
- 异步任务队列用的是单机 SQLite
- 后端默认单模型实例 + `model_lock`
- 默认 CORS 比较宽松
- 没有鉴权、限流、审计和对象存储

所以可以把它理解为：

“研究代码已经完成服务化第一步，但还不是生产化产品。”

## 2. 目录与职责

你主要只需要关注这几块：

- `scripts/demo/web/backend/main.py`
  - FastAPI 入口
  - session 管理
  - 同步 `/ask`、`/ground`
  - 异步 `/tasks`
- `scripts/demo/web/backend/task_queue/worker.py`
  - 真正执行异步任务的 worker
  - 支持 `ASK`、`GROUND`、`TOKEN_TO_REGION`、`REGION_TO_TOKEN`
- `scripts/demo/web/frontend/src/`
  - 当前前端原型
  - 只覆盖基本问答、图片上传、ground 展示
  - 还没有完整 explainable 前端
- `scripts/demo/interact.py`
  - CLI 主逻辑
  - backend/worker 本质上都在复用这里的流水线
- `doc/06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`
  - 后端、worker、token-to-region、region-to-token 的代码说明
- `doc/06-explainable-framework/ATTENTION_DIRECTIONS.md`
  - explainable 两个方向的理论和代码映射

## 3. 推荐运行方式

如果你准备做 explainable 前端，推荐使用“后端不占 GPU + worker 独立跑模型”的模式。

也就是：

- FastAPI backend 只负责 HTTP、session、静态文件、任务入队
- worker 负责真正推理

这样前端接入最稳定，也方便后面把 worker 单独迁到另一台 GPU 机器。

### Terminal A: backend

```bash
cd /home/hechen/gyk/F-LMM

export HF_HUB_OFFLINE=1
export FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
export FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
export FLMM_WEB_RESULTS_DIR=./results
export FLMM_WEB_PROMPT_FILE=scripts/demo/web/backend/prompt/detail.json
export FLMM_WEB_NO_MODEL=1

uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
```

### Terminal B: worker

```bash
cd /home/hechen/gyk/F-LMM

export HF_HUB_OFFLINE=1
export FLMM_WEB_RESULTS_DIR=./results
export FLMM_WEB_PROMPT_FILE=scripts/demo/web/backend/prompt/detail.json
export CUDA_VISIBLE_DEVICES=0
export FLMM_WEB_DEVICE=cuda:0
export FLMM_WEB_DEVICE_MAP=

python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5
```

关键要求：

- `FLMM_WEB_RESULTS_DIR` 必须和 backend 保持一致
- 如果自定义了 `FLMM_WEB_TASK_DB`，backend 和 worker 也必须一致

## 4. 前端应该怎么接

### 4.1 最小同步链路

当前原型前端走的是同步接口：

1. `POST /session/create`
2. `POST /load_image`
3. `POST /ask`
4. `POST /ground`
5. `POST /clear`

这个链路适合：

- 本地单人调试
- 先把聊天 + ground UI 做出来

不太适合：

- 后端不加载模型的模式
- explainable 异步任务
- 之后拆 worker

### 4.2 explainable 前端推荐链路

推荐改成：

1. `POST /session/create`
2. `POST /load_image`
3. `POST /tasks` 入队 `ASK`
4. `GET /tasks/{task_id}` 轮询到 `DONE`
5. 基于返回的 `phrases` / `verification.records` 展示 grounding
6. 用户点击 token / phrase 时，再入队 `TOKEN_TO_REGION`
7. 用户框选 bbox 或点击已有 ground record 时，再入队 `REGION_TO_TOKEN`

这样后面前端不需要关心模型是不是在 backend 进程内。

## 5. API 约定

所有接口都返回统一 envelope：

```json
{
  "status": "ok",
  "message": "",
  "data": {}
}
```

### 5.1 Session

#### `POST /session/create`

请求：

```json
{}
```

或：

```json
{
  "image_path": "/abs/path/to/image.png"
}
```

返回重点字段：

- `data.session_id`
- `data.session_dir`
- `data.session_dir_url`
- `data.image`

#### `POST /load_image`

请求：

```json
{
  "session_id": "your-session-id",
  "image_base64": "data:image/png;base64,..."
}
```

或：

```json
{
  "session_id": "your-session-id",
  "image_path": "/abs/path/to/image.png"
}
```

### 5.2 同步问答

#### `POST /ask`

请求：

```json
{
  "session_id": "your-session-id",
  "question": "Where is the shampoo?",
  "reset_history": false,
  "enable_roi": true
}
```

返回重点字段：

- `data.answer`
  - 最终展示答案
- `data.raw_answer`
  - 第一次回答，尚未经过 ROI re-answer
- `data.phrases`
  - 当前用于 grounding 的短语列表
- `data.verification`
  - ROI 重答、bbox、records 等调试和解释信息

前端展示建议：

- 主回答区显示 `answer`
- 调试面板可折叠显示 `raw_answer`
- grounding 面板用 `phrases`
- 如果要展示自动 ROI 验证结果，再读 `verification`

### 5.3 同步 grounding

#### `POST /ground`

请求：

```json
{
  "session_id": "your-session-id",
  "indices": [0]
}
```

返回重点字段：

- `data.records[*].overlay_url`
- `data.records[*].mask_url`
- `data.records[*].roi_url`
- `data.records[*].bbox`

## 6. explainable 异步任务

### 6.1 入队接口

#### `POST /tasks`

支持类型：

- `ASK`
- `GROUND`
- `TOKEN_TO_REGION`
- `REGION_TO_TOKEN`

返回：

```json
{
  "status": "ok",
  "data": {
    "task_id": 123
  },
  "message": ""
}
```

#### `GET /tasks/{task_id}`

轮询任务状态：

- `PENDING`
- `RUNNING`
- `DONE`
- `FAILED`

真正结果放在：

- `data.task.output_json`

### 6.2 `ASK`

建议 payload：

```json
{
  "type": "ASK",
  "session_id": "your-session-id",
  "payload": {
    "question": "Where is the shampoo?",
    "enable_roi": true
  }
}
```

如果 session 已经 load 过图，通常不需要重复传 `image_path`。

### 6.3 `TOKEN_TO_REGION`

作用：

- 给定输出 token / phrase
- 回看这个 token 主要依赖图像哪里

建议 payload：

```json
{
  "type": "TOKEN_TO_REGION",
  "session_id": "your-session-id",
  "payload": {
    "token_span": [7, 8],
    "topk": 8
  }
}
```

返回 `output_json` 重点字段：

- `overlay_png_url`
- `heatmap_png_url`
- `meta`

### 6.4 `REGION_TO_TOKEN`

作用：

- 给定图像区域
- 回看这个区域最影响哪些 token / phrase

两种用法：

#### 直接传 bbox

```json
{
  "type": "REGION_TO_TOKEN",
  "session_id": "your-session-id",
  "payload": {
    "bbox": [112, 165, 150, 247],
    "topk": 8
  }
}
```

#### 复用 ground record

```json
{
  "type": "REGION_TO_TOKEN",
  "session_id": "your-session-id",
  "payload": {
    "record_index": 0,
    "topk": 8
  }
}
```

返回 `output_json` 重点字段：

- `bbox_overlay_png_url`
- `region_overlay_png_url`
- `region_heatmap_png_url`
- `ranking_json_url`
- `top_tokens`
- `top_phrases`
- `meta`

## 7. 前端最容易踩坑的语义

这部分很重要。

### 7.1 `answer` 不一定等于 `raw_answer`

当前 `ASK` 流水线是：

1. 先生成第一次回答
2. 从第一次回答里抽短语
3. 做 grounding
4. 用 bbox 触发 ROI re-answer
5. 最终把 ROI 答案作为 `answer`

所以：

- `raw_answer` 是第一次回答
- `answer` 是最终展示答案

### 7.2 `phrases` 来自第一次回答，不一定来自最终 `answer`

这意味着前端如果高亮短语：

- 应该把它理解成“grounding 证据短语”
- 而不是“最终展示答案中的逐字短语”

也就是说，`phrases` 和最终 `answer` 之间可能有轻微不一致。

这是当前设计下的正常现象，不是前端 bug。

### 7.3 URL 字段优先，路径字段次要

前端优先使用：

- `session_dir_url`
- `overlay_url`
- `mask_url`
- `roi_url`
- `overlay_png_url`
- `heatmap_png_url`

不要优先依赖本地磁盘路径。

`path` / `session_dir` 更适合调试，不适合作为浏览器最终访问地址。

### 7.4 当前原型前端只覆盖同步接口

`scripts/demo/web/frontend/src/App.tsx` 现在主要是参考实现。

它能说明：

- session 怎么建
- 图片怎么传
- `/ask` 和 `/ground` 怎么调

但它还不是 explainable 前端最终形态，因为它没有完整接入：

- `POST /tasks`
- `GET /tasks/{id}`
- `TOKEN_TO_REGION`
- `REGION_TO_TOKEN`

## 8. 静态资源与结果目录

后端会把结果目录挂到：

- 默认 `/results`

worker / backend 会把结果写到：

- `results/sessions/{session_id}/images/...`
- `results/sessions/{session_id}/turns/turn_{idx}/ground/...`
- `results/sessions/{session_id}/turns/turn_{idx}/attn/token_to_region/...`
- `results/sessions/{session_id}/turns/turn_{idx}/attn/region_to_token/...`

对前端来说，只需要知道：

- 所有可视化文件最终都会有 `*_url`
- 可以直接当 `<img src>` 使用

## 9. 建议的前端里程碑

建议分三步做，不要一开始把所有 explainable 交互堆进一个页面。

### Milestone 1

- 图片上传
- 会话创建
- 异步 `ASK`
- 展示 `answer`
- 展示 `phrases`
- 展示自动 grounding records

### Milestone 2

- 用户点击 phrase
- 入队 `TOKEN_TO_REGION`
- 展示 token-to-region heatmap

### Milestone 3

- 用户框选 bbox 或点击已有 ground record
- 入队 `REGION_TO_TOKEN`
- 展示 top tokens / top phrases 排名
- 展示 `bbox_overlay` / `region_heatmap`

## 10. 如果后面准备真正上线

至少要补这些东西：

- 鉴权
- 限流
- 日志与异常监控
- session 持久化
- 对象存储替代本地磁盘
- PostgreSQL / Redis / MQ 替代 SQLite
- backend/worker 容器化
- 明确 GPU 资源调度
- 删除或隔离调试字段
- 把前端从研究仓库里独立成单独应用

## 11. 一句话交接

如果你是新的前端开发者，建议这样理解这套系统：

- backend 提供会话、静态文件和任务入队
- worker 提供真正的推理和 explainable 计算
- 前端以后应该以异步 `/tasks` 为主，而不是把 `/ask` 当唯一入口
- `scripts/demo/web/frontend/src/App.tsx` 只能当参考原型，不要把它当最终架构
