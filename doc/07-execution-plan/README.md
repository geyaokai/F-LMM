# Runtime / 直连接入 / 纠错层执行文档

本文档面向项目推进本身，不是面向单次对话。

适用范围：

- 前端脱离 VS Code port forwarding，直接访问 FastAPI backend
- 从 F-LMM 打出可长期运行的 runtime 版本
- runtime 稳定后继续推进可解释框架纠错层与 human-in-the-loop

## 1. 文档分工

### 放在 `agent.md` 的内容

- 后续 agent 必须知道的当前真实情况
- 必做验证流程
- review / commit 纪律
- 当前优先级与禁区

### 放在本项目文档的内容

- 需求背景
- 工程分阶段执行顺序
- 各阶段交付物
- 验收标准
- 风险、依赖和回滚点

原因很简单：
- `agent.md` 解决“新对话怎么不跑偏”
- 本文档解决“项目整体怎么推进和验收”

## 2. 截至 2026-03-23 的已验证事实

### 2.1 前后端现状

- 当前真实在用的前端不在仓库内；仓库内旧的 `scripts/demo/web/frontend/` 废弃原型已移除。
- 当前真实后端入口是 `scripts/demo/web/backend/main.py`。
- worker 入口是 `scripts/demo/web/backend/task_queue/worker.py`。

### 2.2 直连访问现状

以下事实已经在 2026-03-23 验证：

- FastAPI 实际监听在 `0.0.0.0:9000`
- 服务器内网 IP 为 `10.208.40.11`
- `GET http://10.208.40.11:9000/healthz` 返回 `200 OK`
- CORS 预检通过，说明后端本身不限制本地前端跨域访问
- `POST /session/create` 可正常返回 session 信息
- `POST /tasks` 与 `GET /tasks/{task_id}` 队列链路可用

### 2.3 当前运行模式

当前 backend 环境变量中已确认：

```bash
FLMM_WEB_NO_MODEL=1
FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
FLMM_WEB_RESULTS_DIR=./results
FLMM_WEB_PROMPT_FILE=scripts/demo/web/backend/prompt/detail.json
```

因此当前应理解为：

- backend 只负责 HTTP / session / 静态文件 / tasks 入队
- worker 负责实际推理

### 2.4 已暴露出的真实工程点

- 后端返回的静态资源 URL 目前是相对路径 `/results/...`
- 因此跨机器前端接入时，前端要负责把相对路径补成绝对 URL
- 当前异步 ASK 任务已经验证会把错误可靠返回到 task 状态中
  - 2026-03-23 的一次实测里，任务因 GPU 忙导致 `CUDA out of memory`
  - 这说明队列和错误回传是工作的，但不能把当前 GPU 环境视为稳定可推理状态

## 3. 总体执行顺序

严格按下面顺序推进：

1. 先打通“前端直连 backend”
2. 再把 backend/worker 打成独立 runtime
3. 最后推进 correction layer

理由：

- 如果前端还依赖 VS Code 转发，就无法稳定验证 runtime
- 如果 runtime 还没独立，纠错层就很难有稳定运行环境
- 如果没有稳定 runtime，human-in-the-loop 交互会一直绑在临时开发环境上

## 4. 任务一：前端直接访问 FastAPI backend

### 4.1 目标

让本地前端或后续迁移后的前端，不再依赖 VS Code port forwarding，而是直接访问服务器 IP / 域名上的 backend。

### 4.2 现阶段结论

后端代码本身没有把访问锁死在 `localhost`。

当前已经满足的条件：

- 监听地址是 `0.0.0.0`
- CORS 默认放开
- 静态文件目录已经挂载到 `/results`

当前尚未固化的部分：

- 前端的 `BACKEND_BASE_URL` 约定
- 相对资源 URL 的绝对化处理
- 若未来换成域名，需要把 CORS 从 `*` 收紧成固定 origin

### 4.3 交付物

- 前端运行时可配置的 `BACKEND_BASE_URL`
- 静态资源 URL 绝对化策略
  - 前端拼接，或
  - backend 增加 `PUBLIC_BASE_URL`
- 一份可重复执行的网络自检脚本或命令清单

### 4.4 验收标准

下面四项全部通过，才算任务一完成：

1. 前端机器可直接访问 `GET /healthz`
2. 前端机器可跨域调用 `POST /session/create`
3. 前端机器能正确加载 `/results/...` 图片资源
4. 在当前运行模式下，前端能走 `POST /tasks` + `GET /tasks/{task_id}` 完成异步链路

### 4.5 可验证流程

如果环境带代理，优先加 `--noproxy '*'`，否则内网 IP 可能被代理错误接管。

#### Step 1: 健康检查

```bash
curl --noproxy '*' -i http://10.208.40.11:9000/healthz
```

通过标准：
- HTTP 状态码 `200`
- 返回体里有 `status: ok`

#### Step 2: CORS 预检

```bash
curl --noproxy '*' -i -X OPTIONS http://10.208.40.11:9000/ask \
  -H 'Origin: http://localhost:5173' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type'
```

通过标准：
- 有 `access-control-allow-origin`
- 有 `access-control-allow-methods`

#### Step 3: 创建会话

```bash
curl --noproxy '*' -sS -X POST http://10.208.40.11:9000/session/create \
  -H 'Content-Type: application/json' \
  -d '{}'
```

通过标准：
- 返回 `session_id`
- 返回 `session_dir_url`

#### Step 4: 验证资源 URL 处理

如果返回：

```json
{
  "session_dir_url": "/results/sessions/<session_id>"
}
```

前端必须将其转换为：

```text
http://10.208.40.11:9000/results/sessions/<session_id>
```

否则跨机器前端无法正确加载图片资源。

#### Step 5: 异步任务链路

当前 backend 是 `FLMM_WEB_NO_MODEL=1`，因此不要把 `/ask` 作为主要验收路径，应使用：

1. `POST /session/create`
2. `POST /load_image` 或在创建 session 时带 `image_path`
3. `POST /tasks`
4. `GET /tasks/{task_id}`

通过标准：
- 可以成功入队
- 可以轮询到 `PENDING/RUNNING/DONE/FAILED`
- 即使失败，也必须能在 `task.error` 中看到明确错误

## 5. 任务二：独立 runtime 打包与常驻部署

### 5.1 目标

从当前 F-LMM 研究仓库中提取出一个可长期运行的 runtime 版本，使：

- F-LMM 继续做研究开发
- runtime 在服务器上长期运行
- runtime 的启动方式、依赖、结果目录和版本都可追踪

### 5.2 推荐策略

先做“可打包 runtime”，再考虑是否拆独立仓库。

推荐第一版目标：

- 仍以 F-LMM 为唯一开发源
- 新增一个 `runtime/` 或 `deployment/` 级别的打包入口
- 打出 versioned bundle，例如：
  - `artifacts/flmm-runtime-20260323-<gitsha>.tar.gz`

这样可以先把“可运行、可归档、可回滚”做实，再决定是否拆 repo。

### 5.2.1 与开发仓库的隔离原则

如果 runtime 是从独立 bundle 解压目录启动，那么后续继续修改开发中的 `F-LMM/` 仓库代码，不会直接影响已经部署的 runtime。

这个结论成立的前提是：

- runtime 不是直接从当前开发仓库启动
- runtime 使用独立的 release 目录
- runtime 使用独立的 `.env` / `results/` / `task_queue.db`

反过来，下面这些情况会让开发改动影响 runtime：

- runtime 直接跑在当前开发仓库里
- 重新 build 并重新 deploy 了 bundle
- runtime 的 prompt / config / 其他关键资源仍指向开发仓库中的可变路径
- runtime 与开发环境共用同一份结果目录或同一份任务 DB

因此，任务二的验收不只是“能打包”，还包括“部署后与开发工作目录解耦”。

### 5.3 runtime 最小内容

runtime bundle 至少应包含：

- backend 启动入口
- worker 启动入口
- prompt 文件
- 必要配置文件
- requirements lock 或环境说明
- `.env.example`
- `VERSION` 或 manifest
- 启动脚本
- 自检脚本

如果要长期托管，建议继续补：

- `systemd` service 文件或 supervisor 配置
- 日志目录约定
- 结果目录约定
- 端口与 GPU 绑定说明

### 5.4 交付物

- 可生成的 runtime bundle
- bundle 内的版本描述
- 启动文档
- smoke test
- 部署回滚说明

### 5.5 验收标准

以下项目全部通过，runtime 才算可交付：

1. 在独立目录解压后可启动 backend
2. worker 能连接到同一 task DB
3. `GET /healthz` 正常
4. `POST /session/create` 正常
5. `POST /tasks` / `GET /tasks/{task_id}` 正常
6. GPU 空闲时，至少有一个 ASK 样例能成功跑通
7. 失败时错误信息能被保留到日志和 task 状态中

### 5.6 可验证流程

#### Step 1: 生成 bundle

```bash
git -C /home/hechen/gyk/F-LMM rev-parse --short HEAD
```

将 git short sha 写入 runtime 版本名或 manifest。

#### Step 2: 在独立目录解压

bundle 不应依赖开发工作目录里的隐式相对路径。

#### Step 3: 启动 backend

推荐命令：

```bash
source /home/hechen/miniconda3/etc/profile.d/conda.sh
conda activate flmm-qwen

uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
```

当前建议 conda 环境固定为 `flmm-qwen`。

补充说明：

- 2026-03-23 的一次本地实测里，`conda run -n flmm-qwen ...` 曾触发 `ModuleNotFoundError: scripts.demo`
- 因此更稳妥的方式是 `conda activate flmm-qwen` 后再进入仓库目录启动

#### Step 4: 启动 worker

```bash
source /home/hechen/miniconda3/etc/profile.d/conda.sh
conda activate flmm-qwen

python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5
```

#### Step 5: 跑 smoke test

至少验证：

- `/healthz`
- `/session/create`
- `/tasks`
- `/tasks/{task_id}`

如果不想占住当前终端，可以直接用 `tmux` 启动 backend/worker。

backend 参考命令：

```bash
tmux new-session -d -s flmm-runtime-backend \
  "bash -lc 'source /home/hechen/miniconda3/etc/profile.d/conda.sh && \
  conda activate flmm-qwen && \
  cd /path/to/F-LMM && \
  PYTHONUNBUFFERED=1 \
  FLMM_WEB_NO_MODEL=1 \
  FLMM_WEB_HOST=127.0.0.1 \
  FLMM_WEB_PORT=19000 \
  FLMM_WEB_RESULTS_DIR=/tmp/flmm-runtime-backend/results \
  FLMM_WEB_TASK_DB=/tmp/flmm-runtime-backend/results/task_queue.db \
  FLMM_WEB_PROMPT_FILE=scripts/demo/web/backend/prompt/detail.json \
  python -m uvicorn scripts.demo.web.backend.main:app --host 127.0.0.1 --port 19000'"
```

2026-03-23 本地 backend 实测结果：

- `GET http://127.0.0.1:19000/healthz` 返回 `200`
- `POST /session/create` 返回有效 `session_id`
- `POST /tasks` 在有效 session 上可成功入队
- `GET /tasks/{task_id}` 可返回 `PENDING`

注意：

- 本机或内网验证时，如果 shell 环境带代理，建议显式 `--noproxy '*'`
- 否则像 `127.0.0.1` 这样的地址也可能被代理接管，导致 `502 Bad Gateway` 之类的伪故障

#### Step 6: 再做一次真实 ASK

如果失败，先判定失败类型：

- 如果是 `CUDA out of memory`，归类为环境资源问题
- 如果是路径缺失、导入失败、配置错误，归类为 runtime 打包问题

### 5.7 git 存档策略

以下时点适合 commit：

- runtime bundle 结构首次稳定
- 首次在服务器上独立跑通 smoke test
- 首次形成可回滚版本

提交前必须：

1. review diff
2. 回读启动文档
3. 跑 smoke test
4. 确认版本号 / manifest / 启动方式一致

必要时加 tag，而不是只靠 commit message 记忆。

## 6. 任务三：纠错层与 human-in-the-loop

### 6.1 前置条件

任务三默认排在 runtime 稳定之后。

不满足以下条件时，不建议启动大规模纠错开发：

- runtime 启动和验证流程未固定
- 样例集未固定
- 当前失败类型还没有被分层记录

### 6.2 问题拆分

当前已观察到的失败类型至少有两类：

1. 生成脏输出
   - 例如 `addCriterion`
   - 例如莫名的 `***`
2. ground 的内容和文本不一致

当前优先聚焦第二类，因为它更适合被解释信号和 human feedback 利用。

### 6.3 推荐设计

先把 correction layer 拆成三步，而不是一次做成“大纠错器”。

#### Step A: Ground 合理性判定

针对每个 grounded concept，判断：

- 当前 answer span 是否真的对应这个 ROI
- 当前 ROI 是否支持这个 concept
- 置信度是否足够保留

输出建议至少包含：

- `concept_text`
- `answer_span`
- `roi_bbox`
- `judge_label`
  - `keep`
  - `reject`
  - `repair`
- `judge_reason`
- `judge_source`
  - `model`
  - `human`

#### Step B: 自动纠错

如果判定为 `repair`，再触发纠错逻辑：

- 保留原 answer
- 生成 corrected concept / corrected grounding
- 必要时再触发 ROI 重答

#### Step C: Human-in-the-loop

前端应支持“按 concept 粒度”给出人工反馈，而不是只给整句 answer 反馈。

最低能力建议：

- 展示每个 grounded concept
- 人工标记：
  - 正确
  - 错误
  - 不确定
- 若错误，允许：
  - 重新选 ROI
  - 重新指定 concept
  - 触发纠错

### 6.4 最小交付物

- 一组固定 manifest 样例
- baseline 结果
- correction 结果
- human override 结果
- 统一结果格式

### 6.5 验收标准

以下标准至少满足前四项，才算 correction layer 进入可持续迭代状态：

1. 固定样例集可重复跑
2. 每个样例都有 baseline 结果
3. 每个样例都有 correction 前后对照
4. 每个 ground record 都能追溯到 answer span 和 ROI
5. human override 后结果能稳定落盘
6. 失败原因能区分为：
   - 检测失败
   - 判定失败
   - 修复失败
   - UI 交互失败

### 6.6 可验证流程

建议固定三套对照：

1. `baseline`
2. `auto_correction`
3. `human_override`

每次改动后至少比较：

- answer 是否更符合文本
- ground 是否更符合 ROI
- 是否引入新的脏输出
- 是否让正确样例被误修

## 7. 统一自检流程

任何 agent 在执行上述三类任务时，都应按同一节奏工作：

1. 读 `agent.md`
2. 读本执行文档
3. 看当前 dirty files
4. 只做当前阶段需要的最小改动
5. 运行对应阶段的最小验证
6. 回读 diff
7. 明确记录：
   - 改了什么
   - 验证了什么
   - 还有什么没验证

## 8. review 与提交规则

### 8.1 提交前必须 review

最少执行：

```bash
git -C /home/hechen/gyk/F-LMM status --short
git -C /home/hechen/gyk/F-LMM diff --stat
git -C /home/hechen/gyk/F-LMM diff -- doc/07-execution-plan/README.md
git -C /home/hechen/gyk/F-LMM diff -- agent.md
```

若涉及代码改动，再补充 relevant path 的 diff review。

### 8.2 什么时候值得 commit

- 一个阶段的验收链路已经跑通
- 一份文档已经足够支撑新对话执行
- runtime 已形成可回滚里程碑

### 8.3 不应 commit 的情况

- 只有半成品命令
- 验证还没跑
- diff 里夹杂 unrelated 改动但还没分清

## 9. 本文档之后的推荐下一步

如果要开新对话执行，建议按这个顺序：

1. 先落实前端 `BACKEND_BASE_URL` 与 `/results/...` 绝对化策略
2. 再抽 runtime 打包结构与 smoke test
3. runtime 稳定后，再开始 correction layer 的数据结构与交互草图
