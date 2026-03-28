# Web Demo Workspace

这个目录放的是 Web demo 的实际代码，而不是完整文档中心。

目录职责：

- `backend/`
  - FastAPI 服务入口
  - session 管理
  - 同步接口
  - SQLite task queue
  - worker
- `frontend/`
  - 仓库内旧的 React + Vite 前端原型目录
  - 现已从仓库移除
  - 相关接口与交互说明保留在 `doc/webdemo/`

推荐先看这些文档，而不是直接从代码猜：

- `../../../doc/webdemo/FRONTEND_AGENT_HANDOFF.md`
  - 给前端开发者 / agent 的交接文档
- `../../../doc/webdemo/README.md`
  - Web demo 端到端逻辑说明
- `./backend/README.md`
  - backend 和 worker 的启动命令
- `../../../doc/06-explainable-framework/DEMO_USAGE_AND_CODE_GUIDE.md`
  - token-to-region / region-to-token 代码逻辑

## 快速启动

### Backend / worker

直接看：

- `scripts/demo/web/backend/README.md`

### Frontend

仓库内旧前端原型已移除，本目录不再提供前端启动命令。

如需接入新的前端实现，请阅读：

- `../../../doc/webdemo/FRONTEND_AGENT_HANDOFF.md`
- `../../../doc/webdemo/README.md`

## 维护原则

这个目录只放 Web 代码和最短必要说明。

更完整的：

- 接口说明
- explainable 任务说明
- 前后端协作文档
- 部署与运行说明

统一放在 `doc/` 和 `backend/README.md`，避免这里再复制一份长期失真。
