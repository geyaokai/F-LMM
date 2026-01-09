# Async Task Worker Quickstart

Run the FastAPI backend and the separate SQLite-backed worker so heavy ASK/GROUND/ATTN jobs stay off the request thread.

## 1) Environment
- `FLMM_WEB_CONFIG` — path to the model config (e.g. `configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py`).
- `FLMM_WEB_CHECKPOINT` — path to the checkpoint `.pth` (e.g. `checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth`).
- `FLMM_WEB_DEVICE` / `FLMM_WEB_DEVICE_MAP` / `FLMM_WEB_DEVICE_MAX_MEMORY` — device overrides.
- `FLMM_WEB_RESULTS_DIR` — root for outputs & task DB (default: `./results`).
- `FLMM_WEB_RESULTS_MOUNT` — URL mount for static files (default: `/results`).
- `FLMM_WEB_TASK_DB` — optional custom path to the task DB (defaults to `$FLMM_WEB_RESULTS_DIR/task_queue.db`).
- `FLMM_WEB_WORKER_ID` — optional identifier shown in logs.

Notes:
- `FLMM_WEB_CHECKPOINT` can be a relative path; backend resolves it relative to repo root.
- `FLMM_WEB_CONFIG` is read as a path (relative is fine as long as you run commands from repo root).

## 2) Start FastAPI (terminal A)
You can use either `uvicorn ...:app` or your previous `fastapi dev ...` workflow.

### Option A: uvicorn (recommended)

#### Linux/macOS (bash/zsh)
```bash
export HF_HUB_OFFLINE=1
export FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
export FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
export FLMM_WEB_RESULTS_DIR=./results

uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
```

#### Windows PowerShell
```powershell
$env:FLMM_WEB_CONFIG = "configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py"
$env:FLMM_WEB_CHECKPOINT = "checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth"
$env:FLMM_WEB_RESULTS_DIR = ".\results"

uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
```

#### Windows CMD
```bat
set FLMM_WEB_CONFIG=configs\qwenrozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
set FLMM_WEB_CHECKPOINT=checkpointsrozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
set FLMM_WEB_RESULTS_DIR=.
esults

uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
```

### Option B: fastapi dev (close to your old command)

#### Linux/macOS (bash/zsh)
```bash
export FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
export FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
export FLMM_WEB_RESULTS_DIR=./results

fastapi dev scripts/demo/web/backend/main.py --host 0.0.0.0 --port 9000
```

#### Windows PowerShell
```powershell
$env:FLMM_WEB_CONFIG = "configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py"
$env:FLMM_WEB_CHECKPOINT = "checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth"
$env:FLMM_WEB_RESULTS_DIR = ".\results"

fastapi dev scripts/demo/web/backend/main.py --host 0.0.0.0 --port 9000
```

- On startup the app initializes the SQLite queue and serves `$FLMM_WEB_RESULTS_DIR` at `$FLMM_WEB_RESULTS_MOUNT`.
- New tasks can be enqueued via `POST /tasks` and polled with `GET /tasks/{task_id}`.

## 3) Start the worker (terminal B)
Important: `FLMM_WEB_RESULTS_DIR` (and optionally `FLMM_WEB_TASK_DB`) must match the backend, otherwise the worker will read a different SQLite file.

#### Linux/macOS (bash/zsh)
```bash
export FLMM_WEB_RESULTS_DIR=./results
# optional: export FLMM_WEB_TASK_DB=./results/task_queue.db

python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5
# optional: --db /custom/path/task_queue.db
```

#### Windows PowerShell
```powershell
$env:FLMM_WEB_RESULTS_DIR = ".\results"
# optional: $env:FLMM_WEB_TASK_DB = ".\results\task_queue.db"

python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5
```

- The worker claims tasks atomically and updates statuses PENDING → RUNNING → DONE/FAILED.

## 4) Directory layout
- `results/sessions/{session_id}/turns/turn_{idx}/ground/ground_{id}/...`
- `results/sessions/{session_id}/turns/turn_{idx}/attn/{i2t|t2i}/attn_{id}/...`
- Uploaded/loaded images reside under `results/sessions/{session_id}/images/`.
- Queue DB: `results/task_queue.db` (or `FLMM_WEB_TASK_DB`).

## 5) Smoke test for the queue only
```bash
python -m scripts.demo.web.backend.task_queue.test_queue
```
This verifies enqueue → claim → done/failed transitions without loading the model.
