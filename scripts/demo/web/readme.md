# Web Demo Workspace

该目录用于前后端协同开发，当前包含：

- `backend/`：FastAPI 服务，直接复用 `scripts/demo/interact.py` 的 `SessionState` 逻辑。
- `frontend/`：预留给 React/Vite 前端（尚未初始化，可运行 `npm create vite@latest frontend -- --template react-ts`）。

## 后端（FastAPI）

1. 激活已有的 FLMM 推理环境（建议 `pip install "fastapi[standard]>=0.115"` 以获得官方 CLI）。
2. 在仓库根目录运行（任选其一）：

   ```bash
   fastapi dev scripts/demo/web/backend/main.py --port 9000
   ```
   或

   ```bash
   uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000 --reload
   ```
   首次启动会根据 `configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py` 加载模型，约耗时数十秒。
3. 服务启动后默认暴露以下 REST 接口：
   - `POST /session`：创建新会话，可选 `image_path` 立即加载图像。
   - `POST /load_image`：加载/切换图像，重置多轮上下文与结果目录。
   - `POST /ask`：`mode=default|roi|cot` 对应 CLI 的 `ask/ask --roi/ask --cot`。
   - `POST /ground`：调用 grounding，返回 overlay/mask/ROI 可视化路径。
   - `POST /clear`：清空多轮对话上下文。
   - `POST /session/reset` 与 `DELETE /session/{id}`：彻底重置或销毁会话。
4. 遵循 `doc/webdemo.md` 的建议，所有遮罩/ROI 文件统一挂载在 `/results/...`，前端可直接通过静态路径访问。

### 日志控制

- 在启动前设置 `FLMM_WEB_LOG_LEVEL=WARNING` 或 `ERROR` 可降低后端日志噪声，`FLMM_WEB_MMENGINE_LOG_LEVEL=ERROR` 可单独抑制 `mmengine` 的初始化输出。
- FastAPI CLI 自带的服务器日志可通过 `fastapi dev ... --log-level warning` 或 `fastapi run ... --log-level error` 控制。

### 可配置项

`main.py` 读取以下环境变量（均可选）：

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `FLMM_WEB_CONFIG` | 模型配置路径 | `configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py` |
| `FLMM_WEB_CHECKPOINT` | 可选 checkpoint | `None` |
| `FLMM_WEB_DEVICE` | 推理设备 | `cuda` |
| `FLMM_WEB_DEVICE_MAP` | Qwen `device_map` | `auto` |
| `FLMM_WEB_RESULTS_DIR` | 结果根目录 | `scripts/demo/results/qwen` |
| `FLMM_WEB_RESULTS_MOUNT` | 静态资源挂载路径 | `/results` |
| `FLMM_WEB_MAX_NEW_TOKENS` 等 | 继承 CLI 中的生成、历史参数 | 见 `main.py` |

## 前端（React）

- 已初始化 Vite + React（TypeScript），目录 `scripts/demo/web/frontend`。
- 首次安装依赖：

  ```bash
  cd scripts/demo/web/frontend
  npm install   # 或 npm ci
  ```

- 启动前端：

  ```bash
  npm run dev -- --host --port 5173
  ```

  如需指向后端，启动前可设置：

  ```bash
  export VITE_API_BASE=http://127.0.0.1:9000
  ```

  若不设，前端在 5173/4173 端口开发时默认访问 `http://127.0.0.1:9000`；生产预览则使用当前页面域名。

- 结果资源（mask/overlay/roi）由后端挂载到 `/results`，前端会自动拼接 `apiBase + /results/...`。

后续可在此文件补充部署脚本、API 调试示例等内容。

后续可在此文件补充部署脚本、API 调试示例等内容。*** End Patch
