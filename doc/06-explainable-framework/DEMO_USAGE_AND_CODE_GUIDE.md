# Demo 使用说明与关键代码逻辑

这份文档只回答两个问题：

1. demo 现在到底该怎么用；
2. 关键代码到底是怎么串起来的。

如果你只看一份，就看这份。

## 1. 先把几个脚本的职责分开

### `scripts/demo/interact.py`

这是最核心的单会话 demo。

它负责：

- 加载模型
- 载入图片
- 执行主流程：`answer -> phrase extract -> ground -> ROI re-answer`
- 把中间产物写到 session 目录

如果你要理解现在系统的主逻辑，先看这个文件。

### `scripts/demo/stability_eval.py`

这是批量稳定性评测脚本。

它负责：

- 读取 manifest
- 对每条样例跑一遍 `pipeline_default_ask`
- 汇总 `report.json`
- 生成适合人工复核的 `review_template.jsonl`

如果你要做第一层稳定性分析，主要用它。

### `scripts/demo/token_to_region_demo.py`

这是单独分析某个输出 token / phrase 对应图像证据的位置。

输入是：

- 图片
- 问题
- 一个 answer token span

输出是：

- 热图
- overlay 图
- 元数据

### `scripts/demo/region_to_token_demo.py`

这是单独分析某个图像区域最影响哪些 answer token / phrase。

输入是：

- 图片
- 问题
- 一个 bbox

输出是：

- 区域 overlay
- token 排名
- phrase 排名
- 元数据

### `scripts/demo/web/backend/task_queue/worker.py`

这是 web demo 后端真正跑模型的地方。

它不是新逻辑，只是把：

- `ASK`
- `GROUND`
- `TOKEN_TO_REGION`
- `REGION_TO_TOKEN`

都接到了统一的任务队列上。

## 2. 最稳妥的运行方式

目前你的环境经验已经很明确了：

- 单卡更稳
- `--device-map none` 更稳
- 先把单卡链路跑顺，再考虑 `device_map=auto` 或多卡分片

所以下面所有命令默认都按单卡写。

## 3. 最常用命令

### 3.1 交互式主 demo

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

启动后常用命令只有几个：

- `load <image_path>`
- `ask <question>`
- `ask --reset-history <question>`
- `ground <idx ...>`
- `clear`
- `exit`

一个最小例子：

```text
ask Where is the shampoo?
ground 0
```

这里的 `ground 0` 是对上一轮答案里第 0 个 phrase 候选做 grounding，不是对 question 做 grounding。

### 3.2 稳定性评测

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

如果先做烟雾测试：

```bash
cd /home/hechen/gyk/F-LMM

CUDA_VISIBLE_DEVICES=0 /home/hechen/miniconda3/envs/flmm-qwen/bin/python \
  scripts/demo/stability_eval.py \
  --manifest scripts/demo/manifests/stability_cases.v1.json \
  --config configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
  --run-name phase1_v1_smoke \
  --device cuda:0 \
  --device-map none \
  --enable-roi \
  --limit 1
```

### 3.3 `token-to-region`

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

这里的 `token-start / token-end` 是 answer token 的区间，不是 question token。

### 3.4 `region-to-token`

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

### 3.5 spaCy 短语提取环境检查

`pipeline_default_ask()` 里的自动短语提取，第一优先级是 spaCy：

- `extract_phrases_via_model(...)`
- `_extract_phrases_spacy(...)`
- `spacy.load("en_core_web_sm")`

所以如果 `flmm-qwen` 环境里没有 `spacy` 或没有 `en_core_web_sm`，就很容易出现：

- `ASK` 正常回答了
- 但是返回 `phrases=[]`
- 自动 `ground` 没跑起来
- 后面只能靠前端手工传 `shirt`、`man`、`yellow shirt` 之类的 phrase 去补 ground

先检查：

```bash
/home/hechen/miniconda3/envs/flmm-qwen/bin/python -c \
"import spacy; print(spacy.__version__); spacy.load('en_core_web_sm'); print('en_core_web_sm OK')"
```

如果这里报：

- `ModuleNotFoundError: No module named 'spacy'`
- 或 `OSError: [E050] Can't find model 'en_core_web_sm'`

就需要在 `flmm-qwen` 里补装。按当前 `requirements_qwen2_5_vl.txt`，推荐直接执行：

```bash
/home/hechen/miniconda3/envs/flmm-qwen/bin/pip install \
  spacy==3.8.7 spacy-legacy==3.0.12 spacy-loggers==1.0.5

/home/hechen/miniconda3/envs/flmm-qwen/bin/python -m spacy download en_core_web_sm
```

然后再验证一次：

```bash
/home/hechen/miniconda3/envs/flmm-qwen/bin/python -c \
"import spacy; nlp = spacy.load('en_core_web_sm'); print('loaded', nlp.meta['name'], nlp.meta['version'])"
```

如果 worker 已经在跑，补装以后要重启 worker；否则旧进程不会重新加载 spaCy 模型。

如果你是在受限沙箱里执行命令，可能会遇到 conda 环境目录只读；这种情况要在宿主机 shell 里手动执行上面的安装命令。

### 3.6 Web demo 后端

如果你要跑异步后端，需要起两个进程：

1. FastAPI backend
2. task worker

重点先记住：

- FastAPI 负责收请求、写 SQLite 队列、返回静态文件 URL
- 真正跑模型的是 `scripts/demo/web/backend/task_queue/worker.py`
- 如果你想把 GPU 压力全放到 worker 上，backend 可以设置 `FLMM_WEB_NO_MODEL=1`

#### 3.6.1 Terminal A: 启动 FastAPI backend
tmux new -s fastapi
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

这里最关键的几个环境变量：

- `FLMM_WEB_CONFIG`
- `FLMM_WEB_CHECKPOINT`
- `FLMM_WEB_RESULTS_DIR`
- `FLMM_WEB_PROMPT_FILE`
- `FLMM_WEB_NO_MODEL`

如果 `FLMM_WEB_NO_MODEL=1`：

- backend 本身不加载模型
- `/ask`、`/ground` 这类同步接口会不可用或返回 503
- 正确用法是走 `/tasks`

#### 3.6.2 Terminal B: 启动 worker
tmux new -s worker
```bash
cd /home/hechen/gyk/F-LMM

export HF_HUB_OFFLINE=1
export FLMM_WEB_RESULTS_DIR=./results
export FLMM_WEB_PROMPT_FILE=scripts/demo/web/backend/prompt/detail.json
export CUDA_VISIBLE_DEVICES=0
export FLMM_WEB_DEVICE=cuda:0
export FLMM_WEB_DEVICE_MAP=
conda activate flmm-qwen
python -m scripts.demo.web.backend.task_queue.worker \
  --sleep 0.5
```

这里最重要的一点是：

- `FLMM_WEB_RESULTS_DIR` 必须和 backend 保持一致

否则 backend 和 worker 会各自看不同的：

- `results/`
- `task_queue.db`

那就会出现“后端收到了任务，但 worker 不处理”的假死现象。

#### 3.6.3 为什么要分成 backend 和 worker

因为 `ASK / GROUND / TOKEN_TO_REGION / REGION_TO_TOKEN` 都比较重：

- 会占 GPU
- 会写 session 产物
- 可能一次执行几秒到几十秒

所以现在设计成：

```text
frontend / client
-> FastAPI backend
-> SQLite task queue
-> worker
-> 写 results/sessions/... 产物
-> backend 返回静态文件 URL
```

这样做的好处是：

- HTTP 请求线程不会被模型推理卡死
- backend 可以不占 GPU
- 后续更容易加多 worker 或迁移机器

#### 3.6.4 backend 现在支持哪些任务

worker 统一处理下面四类：

- `ASK`
- `GROUND`
- `TOKEN_TO_REGION`
- `REGION_TO_TOKEN`

其中：

- `ASK` 还是走 `pipeline_default_ask()`
- `GROUND` 还是走 `handle_ground()` 或 `perform_ground_custom()`
- `TOKEN_TO_REGION` 还是走 `build_token_to_region_heatmap()`
- `REGION_TO_TOKEN` 还是走 `build_region_to_token_scores()`

所以 web backend 不是另一套逻辑，只是把本地 demo 逻辑服务化了。

#### 3.6.5 backend 结果写到哪里

默认在：

```text
results/
```

更具体一点：

```text
results/
  task_queue.db
  sessions/<session_id>/
    images/
    turns/
      turn_0000/
        ground/
        attn/
          token_to_region/
          region_to_token/
```

所以你在 web demo 里看到的图，本质上就是 worker 写到这些目录里的文件。

#### 3.6.6 什么时候还需要看 backend 自己的 README

如果你要看更细的接口细节，再去看：

- `scripts/demo/web/backend/README.md`

06 这份文档现在已经覆盖了：

- 怎么启动
- backend 和 worker 的关系
- 为什么要分离
- 常见任务类型

backend README 保留作更细的接口参考。

#### 3.6.7 一条完整异步请求链是怎么走的

这一段是最重要的后端逻辑图。

如果 frontend 走异步模式，最常见的链路是：

```text
POST /session
-> POST /load_image
-> POST /tasks    (type=ASK)
-> backend 写入 task_queue.db
-> worker 轮询并 claim 任务
-> worker 跑 pipeline_default_ask()
-> worker 把产物写到 results/sessions/... 
-> worker 更新 tasks.status = DONE
-> GET /tasks/{task_id}
-> frontend 读取 output_json 和静态文件 URL
```

更细一点：

```text
frontend
-> main.py:/tasks
-> queue.py:enqueue_task()
-> SQLite tasks 表
-> worker.py:claim_next_task()
-> worker.py:handle_task()
-> interact.py / token_to_region.py / region_to_token.py
-> results/ + tasks.output_json
-> main.py:/tasks/{id}
-> frontend
```

#### 3.6.8 最小异步调用示例

##### 第一步：创建 session

```bash
curl -X POST http://localhost:9000/session \
  -H "Content-Type: application/json" \
  -d '{}'
```

你会拿到一个 `session_id`。

##### 第二步：给 session 载入图片

```bash
curl -X POST http://localhost:9000/load_image \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "image_path": "/home/hechen/gyk/F-LMM/data/custom/shampoo_room.png"
  }'
```

##### 第三步：异步提交 ASK 任务

```bash
curl -X POST http://localhost:9000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ASK",
    "session_id": "YOUR_SESSION_ID",
    "payload": {
      "question": "Where is the shampoo?",
      "enable_roi": true
    }
  }'
```

返回里最关键的是：

- `task_id`

##### 第四步：轮询任务结果

```bash
curl http://localhost:9000/tasks/TASK_ID
```

如果任务还没做完，通常会看到：

- `status = PENDING`
- 或 `status = RUNNING`

做完以后会变成：

- `status = DONE`

这时结果在：

- `task.output_json`

失败则会是：

- `status = FAILED`
- `task.error`

#### 3.6.9 `ASK` 任务入队后，backend 具体做了什么

入口在：

- `scripts/demo/web/backend/main.py`

`POST /tasks` 的逻辑很简单：

1. 检查 session 是否存在
2. 取 `request.payload`
3. 如果是 `ASK` 且 payload 没给 `image_path`，就自动补当前 session 的 `current_image_path`
4. 调 `enqueue_task(...)`
5. 返回 `task_id`

这里有一个很实用的细节：

- 你在前面已经 `load_image` 过的话，后面发 `ASK` 通常不需要再手动传 `image_path`

#### 3.6.10 SQLite queue 里到底存了什么

SQLite 表在：

- `scripts/demo/web/backend/task_queue/schema.sql`

每条任务最关键的字段是：

- `type`
- `status`
- `session_id`
- `turn_idx`
- `input_json`
- `output_json`
- `error`
- `worker_id`

状态流转固定是：

```text
PENDING -> RUNNING -> DONE
PENDING -> RUNNING -> FAILED
```

对应实现看：

- `scripts/demo/web/backend/task_queue/queue.py`

其中：

- `enqueue_task()` 负责插入 `PENDING`
- `claim_next_task()` 负责抢占并改成 `RUNNING`
- `mark_done()` 负责写 `output_json`
- `mark_failed()` 负责写 `error`

还有一个小细节：

- `ASK` 在同一个 `session_id + turn_idx` 下是唯一的
- 其他类型如 `GROUND / TOKEN_TO_REGION / REGION_TO_TOKEN` 允许同一轮有多条

所以同一轮问题重复入队 ASK，会覆盖旧 ASK，而不是一直堆积。

#### 3.6.11 worker 拿到任务后怎么分发

入口在：

- `scripts/demo/web/backend/task_queue/worker.py`

worker 主循环做的事很固定：

1. `claim_next_task()`
2. 根据 `task.type` 调 `handle_task()`
3. 成功则 `mark_done()`
4. 失败则 `mark_failed()`

`handle_task()` 再按类型分发：

- `ASK -> handle_ask()`
- `GROUND -> handle_ground()`
- `TOKEN_TO_REGION -> handle_attention(..., kind=token_to_region)`
- `REGION_TO_TOKEN -> handle_attention(..., kind=region_to_token)`

所以从架构上看，worker 只是一个统一调度层，真正的业务逻辑仍然在原来的 demo 代码里。

#### 3.6.12 `ASK` 在 worker 里最后会跑到哪里

最终还是会跑到：

- `scripts/demo/interact.py`

也就是：

- `pipeline_default_ask()`

所以无论你是：

- 本地 CLI 直接 `ask`
- 还是 web backend 异步发一个 `ASK` 任务

最终主链都是同一套：

```text
model.answer()
-> phrase extract
-> perform_ground()
-> visual_cot_resample()
-> 返回 answer / phrases / verification
```

#### 3.6.13 `TOKEN_TO_REGION` / `REGION_TO_TOKEN` 在 worker 里怎么落盘

这两类任务也是 worker 里统一处理的。

`TOKEN_TO_REGION`：

1. 从缓存 answer 里拿 `attention_maps`
2. 调 `build_token_to_region_heatmap()`
3. 生成 `overlay.png`、`heatmap.png`、`meta.json`
4. 把这些路径写回任务结果

`REGION_TO_TOKEN`：

1. 从 payload 拿 `bbox`，或者从 grounding record 里取 `record_index`
2. 调 `build_region_to_token_scores()`
3. 生成 `bbox_overlay.png`、`region_overlay.png`、`region_heatmap.png`
4. 生成 `ranking.json`、`meta.json`
5. 把 top-k token / phrase 和路径写回任务结果

所以你在前端看到的解释图，本质上都是 worker 写出的静态文件。

#### 3.6.14 同步接口和异步接口的边界

backend 其实同时保留了两套调用方式：

同步接口：

- `/ask`
- `/ground`

异步接口：

- `/tasks`
- `/tasks/{task_id}`

什么时候用哪套：

- 如果 backend 自己加载了模型，可以直接用同步 `/ask`、`/ground`
- 如果 backend 设置了 `FLMM_WEB_NO_MODEL=1`，就只能走异步 `/tasks`

你现在这套更推荐的方式仍然是：

- backend 不占 GPU
- worker 独占 GPU
- 前端统一走 `/tasks`

这样最接近后面真正可维护的服务化形态。

## 4. 结果文件怎么看

### 4.1 交互式 demo

默认结果目录：

```text
scripts/demo/results/qwen
```

一个 session 下面通常长这样：

```text
sessions/<session_id>/
  images/
  turns/
    turn_0000/
      ground/
        ground_0000/
      attn/
        token_to_region/
        region_to_token/
```

### 4.2 Grounding 产物

`ground/ground_xxxx/` 下面最常看的是：

- `overlay_00.png`
- `mask_00.png`
- `roi_00.png`
- `summary.png`

其中：

- `overlay_00.png` 是原图上叠 mask
- `mask_00.png` 是二值 mask
- `roi_00.png` 是从 mask bbox 裁出来的局部图

### 4.3 `token-to-region` 产物

常见文件：

- `overlay.png`
- `heatmap.png`
- `meta.json`

### 4.4 `region-to-token` 产物

常见文件：

- `bbox_overlay.png`
- `region_overlay.png`
- `region_heatmap.png`
- `ranking.json`
- `meta.json`

`ranking.json` 里最重要的是：

- `top_tokens`
- `top_phrases`
- `token_scores`

### 4.5 稳定性评测产物

每次 run 会生成：

- `report.json`
- `review_template.jsonl`
- `artifacts/`

其中：

- `report.json` 适合程序汇总
- `review_template.jsonl` 适合你人工复核

## 5. 主流程到底怎么走

这一节只讲真正重要的主链。

### 5.1 `load_model()`

位置：

- `scripts/demo/interact.py`

它做的事情很直接：

1. 处理 `--device-map` 和 `--device-max-memory`
2. 用 `BUILDER.build(cfg.model)` 构建模型
3. 如果给了 checkpoint，就把 checkpoint load 进去
4. 把模型放到目标设备
5. 调 `model._prepare_for_generation(...)`，把 prompt 模板、max token、额外 prompt 等配置好

这里的关键点是：

- 生成相关设置不是散在很多地方，而是统一在 `_prepare_for_generation()` 里准备
- `interact.py`、`stability_eval.py`、worker 最终都走这个入口

### 5.2 `handle_load()`

位置：

- `scripts/demo/interact.py`

它做三件事：

1. 读取图片
2. 把图片复制到当前 session 目录
3. 清空上一张图留下来的 answer / phrase / ground 缓存

所以换图以后，之前的问题缓存不会继续沿用。

### 5.3 `pipeline_default_ask()`

位置：

- `scripts/demo/interact.py`

这是现在最重要的函数。

它的固定流程就是：

```text
answer
-> phrase extract
-> build phrase candidates
-> auto ground top-k phrases
-> ROI re-answer
-> update history
```

更具体一点：

1. `session.model.answer(...)`
   - 先得到原始回答
   - 同时拿到 `hidden_states` 和 `attention_maps`
2. `build_offsets(...)`
   - 把 answer 文本和 answer token 对齐
3. `extract_phrases_via_model(...)`
   - 从 answer 文本里提 noun phrase
   - 优先走 spaCy 的 `en_core_web_sm`
   - spaCy 不可用时，再退化到 LLM 抽取和规则 fallback
4. `build_phrase_candidates(...)`
   - 给每个 phrase 补上 `char_span` 和 `token_span`
5. `perform_ground(...)`
   - 用这些 answer phrase 的 token span 做 grounding
6. `_first_bbox_record(...)`
   - 取第一个 grounding bbox
7. `visual_cot_resample(...)`
   - 用这个 bbox 做 ROI 重答
8. `append_history_entry(...)`
   - 把最终 answer 写回会话历史

这里最容易误解的一点是：

- 当前自动 grounding 的短语来源是 `answer`
- 不是 `question`

所以如果自动抽出来的是 `dresser`，那么 ROI 框更可能围绕 `dresser` 证据，而不是直接围绕 `shampoo` 这个 question 词。

这也正是你前面观察到的现象来源。

这里还有一个很常见的工程性故障：

- 如果 spaCy 或 `en_core_web_sm` 没装好，`ASK` 可能直接出现 `phrases=[]`
- 这时候不是 grounding 模块先坏了，而是 phrase extract 这一步先掉了
- 排查时先看 worker 日志里有没有：
  - `spaCy import unavailable`
  - `spaCy model 'en_core_web_sm' unavailable`
- 修法见前面的“3.5 spaCy 短语提取环境检查”

## 6. `FrozenQwen.answer()` 到底缓存了什么

位置：

- `flmm/models/frozen_qwen.py`

这个函数不是只返回一段文本，它还把后面 grounding 和解释要用的缓存一次性准备好了。

### 6.1 它先做正常生成

核心步骤：

1. `_build_conversation(...)` 组织 system/history/user/image
2. `processor.apply_chat_template(...)` 生成 Qwen 输入文本
3. `processor(...)` 生成 `input_ids / pixel_values / image_grid_thw`
4. `qwen_model.generate(...)`
   - 要求返回 `attentions`
   - 要求返回 `hidden_states`

### 6.2 `images_seq_mask` 是干什么的

`_prepare_inputs()` 里会扫描整个序列，找到：

- `<|vision_start|>`
- `<|vision_end|>`

然后把它们中间的图像 token 位置标成 `True`。

所以 `images_seq_mask` 的作用不是算 attention，而是回答：

```text
full sequence 里哪些列属于 image token？
```

### 6.3 `attention_maps` 不是 full attention

当前缓存下来的 `attention_maps` 已经是裁剪过的子块：

```text
generated answer token  ->  image token
```

也就是：

```text
rows = answer token
cols = image patch
```

最终 reshape 成：

```text
[layers, heads, generated_tokens, H, W]
```

所以后面的：

- `token-to-region`
- `region-to-token`
- `ground()`

都不是直接重新跑 full attention，而是在复用这一块缓存。

### 6.4 为什么这很重要

因为这说明：

- `token-to-region` 和 `region-to-token` 来自同一块 `answer-to-image` attention
- 只是一个按 token 方向聚合，一个按 region 方向聚合

更详细的 attention 方向说明见：

- `doc/06-explainable-framework/ATTENTION_DIRECTIONS.md`

## 7. Grounding 是怎么做出来的

位置：

- `flmm/models/frozen_qwen.py`
- `scripts/demo/interact.py`

### 7.1 `perform_ground()`

`perform_ground()` 做的是把 phrase 选出来，然后调用模型的 `ground()`。

输入核心是：

- `positive_ids`
  - 也就是若干个 `(token_start, token_end)`
- `hidden_states`
- `attention_maps`
- `meta_data`

### 7.2 `FrozenQwen.ground()`

它的逻辑是：

1. 对每个 phrase span，从 `attention_maps` 里切出对应 token 行
2. 对 span 内 token 聚合
3. 把所有 layer 的结果拼起来，送进 `mask_head`
4. 用 `text_proj(hidden_states[start:end])` 提供文本条件
5. 如果启用 SAM，再用 SAM 做细化
6. 最后把 mask resize 回原图尺寸

所以 grounding 不是靠额外的 detector，而是：

- 基于 answer token 对 image token 的注意力
- 再经过轻量 `mask_head`
- 再 optionally 经过 SAM refine

### 7.3 `save_ground_outputs()`

这个函数负责把预测结果落盘成你平时看到的：

- `mask_xx.png`
- `overlay_xx.png`
- `roi_xx.png`

同时它还会从二值 mask 里提一个 bbox。

后续 ROI 重答默认就是拿这个 bbox。

## 8. ROI 重答到底是什么

位置：

- `flmm/models/frozen_qwen.py`

函数：

- `visual_cot_resample(...)`

它不是把原图重新裁一张再完整送进 processor。

它现在的做法更接近：

1. 从 answer cache 里取已经缓存的 `vision_tokens`
2. 按 bbox 选出 ROI 对应的 patch token
3. 用一段只包含 ROI image token 的 prompt 重新生成回答

所以它是：

- 基于视觉 token 的局部重推理
- 不是简单的 PIL crop 后再正常问一次

这也是它跟普通“裁图再问”最大的区别。

## 9. `token-to-region` 的代码逻辑

位置：

- `scripts/demo/token_to_region.py`

输入：

- `attention_maps`
- `token_span`

步骤很直接：

1. 选 layer / head
   - 可以指定具体层头
   - 也可以默认取 mean
2. 取出 `token_span` 对应的 token 行
3. 对 span 内 token 聚合
   - `mean` 或 `max`
4. 再对 layer / head 聚合
5. 得到二维热图
6. 做 min-max 归一化
7. 叠回原图

所以它回答的问题是：

```text
这个输出 token / phrase 在生成时主要看图上的哪里？
```

## 10. `region-to-token` 的代码逻辑

位置：

- `scripts/demo/region_to_token.py`

它做的不是“图像自己向文本发 attention”，而是：

- 在同一块 `answer-to-image` attention 里
- 取某个区域对应的列
- 再看哪些 answer token 在这些列上响应高

具体步骤：

1. `build_region_mask(...)`
   - 把原图 bbox 投影到 attention grid
   - 得到一个二维 region mask
2. `build_region_to_token_scores(...)`
   - 用这个 mask 去乘 `attention_maps`
   - 对区域内 patch 聚合
   - 生成每个 token 的分数
3. `build_token_records(...)`
   - 把 token index、文本、char span、score 组织起来
4. `build_phrase_records(...)`
   - 用 phrase span 对 token score 再做一次 phrase 级聚合
5. `rank_records(...)`
   - 取 top-k

所以它回答的问题是：

```text
图中的这个区域，最影响哪些输出 token / phrase？
```

## 11. 为什么你会感觉“框的是 dresser，不是 shampoo”

这和当前主流程的设计直接有关。

因为当前自动链路是：

```text
question
-> answer
-> 从 answer 抽 phrase
-> ground phrase
```

所以：

- 如果 answer 里抽到的是 `dresser`
- 那 grounding 的 token span 就是 `dresser`
- 那 ROI 也是从 `dresser` 对应的 mask / bbox 来的

这不是 bug，而是当前设计的自然结果。

它的优点是：

- 可以直接解释“最终答案依赖了什么”

它的局限是：

- 不保证自动框到 question 里的目标词

如果以后你要专门研究 “question target grounding”，那是另一条逻辑，不是现在这条 answer-grounding 逻辑。

## 12. 当前代码是不是“一次推理里同时对 question 和 answer 都做 grounding”

不是。

当前实现更准确地说是：

1. 先跑一次 answer，缓存 answer token 的 `attention_maps / hidden_states`
2. 再从 answer 里选 phrase
3. 再对这些 answer phrase 做 grounding
4. 如果需要，再基于 ROI 做一次重答

也就是说：

- question 是用来触发生成的
- answer phrase 才是后续 grounding 的默认锚点

## 13. Worker 和本地脚本的关系

这块其实很简单。

### 本地脚本

- `interact.py`
- `stability_eval.py`
- `token_to_region_demo.py`
- `region_to_token_demo.py`

这些是你本地直接调的入口。

### Web worker

- `scripts/demo/web/backend/task_queue/worker.py`

这个文件只是把同样的逻辑换成任务队列调用。

例如：

- `ASK` 还是走 `pipeline_default_ask()`
- `GROUND` 还是走 `handle_ground()` / `perform_ground_custom()`
- `TOKEN_TO_REGION` 还是走 `build_token_to_region_heatmap()`
- `REGION_TO_TOKEN` 还是走 `build_region_to_token_scores()`

所以如果本地脚本逻辑你理解了，worker 其实就不难了。

## 14. 现在最推荐的阅读顺序

如果你要真正把代码吃透，建议按下面顺序读：

1. `scripts/demo/interact.py`
2. `flmm/models/frozen_qwen.py`
3. `scripts/demo/token_to_region.py`
4. `scripts/demo/region_to_token.py`
5. `scripts/demo/stability_eval.py`
6. `scripts/demo/web/backend/task_queue/worker.py`

其中：

- `interact.py` 负责“流程编排”
- `frozen_qwen.py` 负责“模型内部缓存和 grounding / ROI 重答”
- `token_to_region.py` 和 `region_to_token.py` 负责“解释分析”
- `stability_eval.py` 负责“批量评测”
- `worker.py` 负责“后端服务化”

## 15. 你现在最需要记住的三句话

1. 当前自动 grounding 是 answer-driven，不是 question-driven。
2. `token-to-region` 和 `region-to-token` 都来自同一块 `answer-to-image` attention 缓存。
3. 如果要先把系统做稳，优先盯 `report.json + review_template.jsonl`，不要先追求更多新可视化。
