# Runtime Bundle And Self-Check

本目录只负责任务二的 runtime 打包、启动模板和 smoke/self-check。

明确边界：

- 不修改也不依赖 `scripts/demo/web/frontend/`
- bundle 默认排除废弃前端、`results/`、测试产物和 `.git/`
- `checkpoints/` 和 `data/` 默认也不打进包里，避免生成巨型归档；需要时用 `--include-model-assets`

## 1. 生成 bundle

在仓库根目录执行：

```bash
python deployment/runtime/build_bundle.py
```

默认输出：

- `artifacts/runtime/flmm-runtime-<timestamp>-<gitsha>.tar.gz`

bundle 根目录会额外生成：

- `VERSION`
- `RUNTIME_MANIFEST.json`

manifest 会记录：

- git sha
- dirty worktree 摘要
- include / exclude 列表
- backend / worker / smoke test 入口

## 2. 解压后目录

解压后的 bundle 仍以仓库根目录形态运行，避免 runtime 继续依赖开发工作目录里的隐式相对路径。

至少包含：

- backend / worker 代码
- `configs/`
- `requirements/`
- `scripts/demo/web/backend/prompt/`
- 启动脚本
- smoke test 脚本
- `.env.example`
- systemd 模板

## 3. 开发隔离边界

如果 runtime 是按本目录方案构建、解压并从独立 release 目录启动，那么你之后继续修改当前开发中的 `F-LMM/` 仓库代码，不会直接影响已经在运行的 runtime。

前提是下面三条同时满足：

- runtime 代码来自独立解压目录，而不是直接从当前开发仓库启动
- runtime 使用自己的 `.env`、`results/`、`task_queue.db`
- 你没有把 runtime 的关键入口继续指回开发仓库里的可变文件

也就是说，下面两种情况要区分清楚：

### 不会影响已部署 runtime

- 你修改的是当前开发仓库 `F-LMM/` 里的 Python 代码
- 已部署 runtime 跑在独立 release 目录，例如 `/srv/flmm-runtime/releases/<version>/`
- 没有重新 build / deploy / 切换 `current` 符号链接

### 会影响 runtime 行为

- runtime 本身就是直接从当前开发仓库启动的
- 你重建了 bundle 并重新部署
- runtime 的 `.env` 指向了开发仓库里的 prompt / config / 其他可变资源
- runtime 和开发环境共用了同一份 `results/` 或 `task_queue.db`

因此，runtime 的真正价值不是“打包过一次”本身，而是：

1. 独立代码快照
2. 独立启动目录
3. 独立环境变量与结果目录
4. 可回滚的 release 版本

## 4. 启动方式

先复制环境模板：

```bash
cp deployment/runtime/.env.example deployment/runtime/.env
```

至少改这几个值：

- `FLMM_WEB_CHECKPOINT`
- `FLMM_WEB_RESULTS_DIR`
- `FLMM_WEB_TASK_DB`
- `FLMM_WEB_NO_MODEL`
- `FLMM_WEB_DEVICE`

当前建议运行环境：

- conda env: `flmm-qwen`

推荐启动方式是先 `conda activate flmm-qwen`，再执行脚本；不要默认依赖 `conda run -n flmm-qwen ...`，因为 2026-03-23 的一次本地实测里，这种方式曾触发 `ModuleNotFoundError: scripts.demo`。

另外，如果当前 shell 里已经带了外部 `PYTHONPATH`（例如指向别的项目根目录），也可能干扰 `scripts.demo` 的导入解析。当前 `deployment/runtime/bin/start_backend.sh` 和 `deployment/runtime/bin/start_worker.sh` 会主动把 runtime 根目录 prepend 到 `PYTHONPATH`，优先使用这两个脚本启动。

然后分别启动：

```bash
source /home/hechen/miniconda3/etc/profile.d/conda.sh
conda activate flmm-qwen

bash deployment/runtime/bin/start_backend.sh
bash deployment/runtime/bin/start_worker.sh
```

如果当前阶段只想先验证 HTTP / queue，不想让 backend 抢 GPU，保留：

```bash
FLMM_WEB_NO_MODEL=1
```

## 5. tmux 后端验证

如果不想占住当前终端，推荐直接用 `tmux` 起 backend / worker。

2026-03-23 已在本机验证下面这条 backend 启动方式可用：

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

实测通过项：

- `GET /healthz` 返回 `200`
- `POST /session/create` 返回 `session_id`
- 在 `FLMM_WEB_NO_MODEL=1` 模式下，`POST /tasks` 可成功入队
- `GET /tasks/{task_id}` 可看到 `PENDING`

验证时如果 shell 环境带代理，访问本机或内网地址时也建议显式禁用代理。例如：

```bash
curl --noproxy '*' -sS http://127.0.0.1:19000/healthz
curl --noproxy '*' -sS -X POST http://127.0.0.1:19000/session/create \
  -H 'Content-Type: application/json' \
  -d '{}'
```

如果要看 tmux 日志：

```bash
tmux capture-pane -pt flmm-runtime-backend
```

## 6. Smoke / Self-Check

最小 smoke：

```bash
bash deployment/runtime/bin/smoke_test.sh \
  --image-path /abs/path/to/image.png \
  --allow-failed-task \
  --cleanup-session
```

脚本会依次验证：

1. `GET /healthz`
2. `POST /session/create`
3. `POST /tasks`
4. `GET /tasks/{task_id}` 直到 `DONE` 或 `FAILED`

说明：

- `DONE` 表示 runtime + worker + task queue 最小链路可用
- `FAILED` 只要 error 被保留，也说明链路可用；常见环境类失败如 `CUDA out of memory`
- 路径缺失、导入失败、配置找不到，应归类为 runtime 打包问题

## 7. systemd 模板

模板位于：

- `deployment/runtime/systemd/flmm-backend.service.example`
- `deployment/runtime/systemd/flmm-worker.service.example`

使用前需要把 `__RUNTIME_ROOT__` 替换成部署目录绝对路径。

## 8. 回滚建议

每次 bundle 都带独立版本号，因此回滚优先用“切回上一个解压目录”的方式，不要在原目录原地覆盖。

推荐目录布局：

```text
/srv/flmm-runtime/
  releases/
    flmm-runtime-20260323_120000-abc123/
    flmm-runtime-20260324_091500-def456/
  current -> releases/flmm-runtime-20260324_091500-def456
```

回滚时：

1. 停 backend / worker
2. 把 `current` 切回上一个 release
3. 重启服务
4. 再跑一遍 smoke test
