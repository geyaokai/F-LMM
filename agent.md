# F-LMM Agent Brief

本文件面向后续新对话中的 agent / subagent。

目标：减少重复口头对齐，让后续执行优先遵守当前项目真实约束、验证流程和提交纪律。

## 1. 当前真实情况

- 截至 2026-03-23，当前实际在用的前端不在本仓库里；仓库内旧的 `scripts/demo/web/frontend/` 废弃原型已移除，不应默认在该路径继续开发。
- 当前实际后端入口是 `scripts/demo/web/backend/main.py`，异步推理由 `scripts/demo/web/backend/task_queue/worker.py` 执行。
- 截至 2026-03-23，服务器上的 FastAPI 已验证可直接通过内网 IP 访问，不依赖 VS Code port forwarding：
  - 服务监听 `0.0.0.0:9000`
  - `GET /healthz` 可从 `http://10.208.40.11:9000` 返回 `200 OK`
  - CORS 预检通过
- 当前运行模式是 `FLMM_WEB_NO_MODEL=1`：
  - backend 负责 HTTP、session、静态文件、任务入队
  - worker 负责模型推理
- 当前推荐 conda 环境是 `flmm-qwen`。
- 如果需要在本机验证 backend / worker，优先用 `tmux` 启动；不要默认依赖 `conda run -n flmm-qwen ...`，2026-03-23 的一次实测里这条路径曾触发 `ModuleNotFoundError: scripts.demo`。
- 后端返回的静态资源 URL 目前是相对路径 `/results/...`，前后端分离时前端必须补全后端 base URL，或后端后续增加 public base URL 配置。

## 2. 当前任务优先级

按顺序推进，不要并行平均用力：

1. 前端直接访问 FastAPI runtime，不再依赖 VS Code 转发
2. 从 F-LMM 里打出独立 runtime 版本，并让 runtime 常驻服务器
3. 在 runtime 稳定之后推进可解释框架的纠错层与 human-in-the-loop

项目执行细节见：
- `doc/07-execution-plan/README.md`
- `doc/webdemo/README.md`
- `doc/webdemo/FRONTEND_AGENT_HANDOFF.md`
- `doc/06-explainable-framework/ROADMAP.md`

## 3. 工作边界

- 不要把已移除的 `scripts/demo/web/frontend/` 原型路径当作现有开发入口；除非用户明确要求恢复该原型，否则不要在该路径继续开发。
- 不要把“能跑 demo”和“适合长期运行的 runtime”混为一谈；runtime 任务需要把启动、环境、依赖、输出目录、版本归档一起固化。
- 如果 runtime 是从独立 bundle 解压目录启动，那么后续修改开发中的 `F-LMM/` 仓库代码，不应直接影响已部署 runtime；若 runtime 直接跑在当前仓库或继续引用开发仓库里的可变资源，则会受影响。
- 不要把纠错层和 runtime 打包同时大改；纠错层排在 runtime 稳定之后。

## 4. 开工前必做

1. 先看 `git -C /home/hechen/gyk/F-LMM status --short`
2. 只围绕当前任务读必要文件，不要大面积改 unrelated 文件
3. 识别当前是：
   - 网络/接入问题
   - runtime 打包/部署问题
   - explainable / correction 问题
4. 先写最小可验证改动，再扩展

## 5. 最小验证流程

### 5.1 网络与直连

如果 shell 环境带代理，验证内网 IP 时优先使用 `--noproxy '*'` 或设置 `NO_PROXY`。

```bash
curl --noproxy '*' -i http://10.208.40.11:9000/healthz

curl --noproxy '*' -i -X OPTIONS http://10.208.40.11:9000/ask \
  -H 'Origin: http://localhost:5173' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type'
```

通过标准：
- `/healthz` 返回 `200 OK`
- CORS 响应里有 `access-control-allow-origin`

### 5.2 backend / queue 最小自检

```bash
curl --noproxy '*' -sS -X POST http://10.208.40.11:9000/session/create \
  -H 'Content-Type: application/json' \
  -d '{}'
```

如果 `FLMM_WEB_NO_MODEL=1`，不要直接测 `/ask`，应改为 `/tasks` + 轮询 `/tasks/{task_id}`。

### 5.3 纠错层验证

- 必须基于固定 manifest 或固定样例集验证
- 至少保留：
  - 原始 answer
  - grounding record
  - ROI / bbox
  - correction 前后结果
  - 自动判定结果

不要只凭单个 demo case 判断“纠错有效”。

## 6. 提交纪律

- 非必要不要提交；只有在形成明确里程碑时才 commit 或 tag
- 提交前必须 review，最低要求：
  - `git diff --stat`
  - `git diff -- <relevant paths>`
  - 回读新增文档或关键改动
  - 运行最小验证流程
- 如果验证失败，先确认是代码问题还是环境问题
  - 例如当前 worker 可能因为 GPU 已占满而在 `ASK` 任务上报 `CUDA out of memory`
  - 这类情况不能直接归因到 API 逻辑错误

## 7. 适合放在 agent.md 的内容

- 当前项目真实约束
- 后续 agent 必须遵守的验证流程
- 哪个前端是废弃的，哪个后端才是真的
- commit / review 纪律
- 任务优先级和阅读入口

## 8. 不适合放在 agent.md、应放入项目文档的内容

- 需求背景
- 里程碑和交付物
- 每个阶段的验收标准
- correction layer 的设计方案
- runtime 打包结构和部署方案

这些内容统一放在 `doc/07-execution-plan/README.md`。
