# Web Demo 开发指南

本指南说明如何把当前 `FrozenQwenSAM` 交互逻辑迁移到 Web 形态，覆盖后端 API、前端 UI 与部署建议，便于快速搭建在线 Demo。

## 1. 目标能力

- 支持多轮问答（沿用 `SessionState.history`）
- 复用 `ask / ask --roi / ask --cot / ground / clear` 等命令
- Web 端可查看原图、mask overlay、ROI、CoT 结果

## 2. 后端方案

### 2.1 框架选择

- 推荐 FastAPI（或 Flask）。FastAPI 自带类型提示、OpenAPI 文档和 async 支持，便于后续扩展流式生成。
- 直接引用 `scripts/demo/interact.py` 中的 `SessionState`、`parse_ask_command`、`handle_*` 函数，避免重复实现。

### 2.2 Session 管理

1. 维护 `Dict[str, SessionState]`，`session_id` 建议使用 UUID。
2. `POST /session/create`：初始化 `SessionState`，返回 `session_id`。
3. `POST /session/reset` 或 `DELETE /session/<id>`：清理历史、删除结果目录。
4. 设置 TTL（如 60 分钟）并定时清理过期 session，避免磁盘堆积。

### 2.3 API 设计示例

| Method | Path | 请求体 | 说明 |
| --- | --- | --- | --- |
| `POST /load_image` | `{session_id, image_path/base64}` | 调 `handle_load`，返回图像信息与结果目录 |
| `POST /ask` | `{session_id, question, mode, index, reset_history}` | `mode` = `default / roi / cot`，内部映射到 `handle_ask / handle_inspect / handle_cot_resample` |
| `POST /ground` | `{session_id, indices}` | 返回 mask/overlay/ROI 的路径 |
| `POST /clear` | `{session_id}` | 等价 CLI `clear`，清空多轮上下文 |
| `GET /results/<path>` | - | 静态文件（overlay/mask/roi） |

统一返回格式：

```json
{
  "status": "ok",
  "data": {...},
  "message": ""
}
```

错误时置 `status="error"` 并附带可读消息。

### 2.4 结果文件与静态服务

- 仍写入 `scripts/demo/results/qwen/<timestamp>/round_xx`。
- FastAPI 可挂载 `StaticFiles(directory="scripts/demo/results")`，前端直接用 `/results/...` 访问。
- 定期清理旧 round（CRON 或后台线程）。

## 3. 前端方案

### 3.1 快速原型

- 使用 Streamlit / Gradio：可直接在 Python 中调用后端或本地模型，适合内部演示。

### 3.2 正式 Web UI

- 技术栈：React/Vite + Tailwind，或 Vue + Element。
- 核心模块：
  1. **聊天区**：展示多轮问答与 CoT 输出。
  2. **图像区**：显示原图，提供 overlay/mask 选择（可用 `<canvas>`、fabric.js 等）。
  3. **短语列表**：列出 `[Phrases]`，点击触发 `/ground`。
  4. **命令面板**：输入问题、选择模式（普通/ROI/CoT），勾选“重置历史”。
- ROI 预览：展示 `roi_XX.png` 缩略图，或在图像上高亮 bbox。

## 4. 部署建议

1. **后端**：编写 `serve_web.py`，用 Uvicorn/Gunicorn 启动；建议容器化以固定环境。
2. **GPU**：单进程占用一块卡；多用户需求可通过队列串行或多实例负载均衡。
3. **文件管理**：守护进程或定时任务删除过期 `results`，防止磁盘爆满。
4. **安全**：若对公网开放，需加鉴权（token/登录）、限流、CORS 控制与日志。

## 5. TODO 清单

- [ ] 实现 FastAPI 服务（封装 `SessionState`，暴露 `/ask`、`/ground` 等端点）。
- [ ] 提供简单 Streamlit/Gradio 前端示例。
- [ ] 完善静态结果目录清理脚本。
- [ ] 视需求加入 WebSocket 流式回答。

按以上步骤即可把 CLI Demo 平滑迁移到 Web 环境，保留多轮对话、Grounding 与 CoT 能力，同时获得更友好的可视化体验。

