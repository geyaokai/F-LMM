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
  - React + Vite 前端原型
  - 当前主要覆盖图片上传、问答、ground 展示
  - 还不是完整 explainable 前端

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

```bash
cd /home/hechen/gyk/F-LMM/scripts/demo/web/frontend
npm install
npm run dev -- --host --port 5173
```

如需指定后端：

```bash
export VITE_API_BASE=http://127.0.0.1:9000
```

默认情况下：

- 开发端口是 `5173` / `4173` 时，前端会默认请求 `http://127.0.0.1:9000`
- 结果资源通过 backend 挂载的 `/results/...` 访问

## 维护原则

这个目录只放 Web 代码和最短必要说明。

更完整的：

- 接口说明
- explainable 任务说明
- 前后端协作文档
- 部署与运行说明

统一放在 `doc/` 和 `backend/README.md`，避免这里再复制一份长期失真。
