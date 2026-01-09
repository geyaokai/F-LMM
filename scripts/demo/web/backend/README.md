# Async Task Worker Quickstart

Run the FastAPI backend and the separate SQLite-backed worker so heavy ASK/GROUND/ATTN jobs stay off the request thread.

New (Jan 2026)
- `FLMM_WEB_NO_MODEL=1` lets the backend skip loading the model (no GPU). Use `/tasks` to enqueue ASK/GROUND and let the worker do inference.

## 1) Environment
- `FLMM_WEB_CONFIG` — model config path (e.g. `configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py`).
- `FLMM_WEB_CHECKPOINT` — checkpoint `.pth` path (e.g. `checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth`).
- `FLMM_WEB_DEVICE` / `FLMM_WEB_DEVICE_MAP` / `FLMM_WEB_DEVICE_MAX_MEMORY` — device overrides; clear `FLMM_WEB_DEVICE_MAP` when you want single-card.
- `FLMM_WEB_RESULTS_DIR` — root for outputs & task DB (default: `./results`).
- `FLMM_WEB_RESULTS_MOUNT` — URL mount for static files (default: `/results`).
- `FLMM_WEB_TASK_DB` — optional custom path to the task DB (defaults to `$FLMM_WEB_RESULTS_DIR/task_queue.db`).
- `FLMM_WEB_WORKER_ID` — optional identifier shown in logs.
- `FLMM_WEB_NO_MODEL` — set to `1` to run backend without loading the model/GPU (only queue + static serving).

Notes:
- `FLMM_WEB_CHECKPOINT` and `FLMM_WEB_CONFIG` may be relative; backend resolves them relative to repo root.

## 2) Start FastAPI (terminal A)
You can use either `uvicorn ...:app` or your previous `fastapi dev ...` workflow.

### Option A: uvicorn (recommended)
```bash
export HF_HUB_OFFLINE=1
export FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
export FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
export FLMM_WEB_RESULTS_DIR=./results
export FLMM_WEB_NO_MODEL=1 
uvicorn scripts.demo.web.backend.main:app --host 0.0.0.0 --port 9000
# optional: export FLMM_WEB_NO_MODEL=1   # backend skips model/GPU
```

### Option B: fastapi dev
```bash
export FLMM_WEB_CONFIG=configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py
export FLMM_WEB_CHECKPOINT=checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth
export FLMM_WEB_RESULTS_DIR=./results
# optional: export FLMM_WEB_NO_MODEL=1

fastapi dev scripts/demo/web/backend/main.py --host 0.0.0.0 --port 9000
```

- On startup the app initializes the SQLite queue and serves `$FLMM_WEB_RESULTS_DIR` at `$FLMM_WEB_RESULTS_MOUNT`.
- New tasks can be enqueued via `POST /tasks` and polled with `GET /tasks/{task_id}`.
- If you set `FLMM_WEB_NO_MODEL=1`, `/ask` and `/ground` return 503; enqueue ASK/GROUND via `/tasks` instead.

## 3) Start the worker (terminal B)
Important: `FLMM_WEB_RESULTS_DIR` (and optionally `FLMM_WEB_TASK_DB`) must match the backend, otherwise the worker will read a different SQLite file.

```bash
export FLMM_WEB_RESULTS_DIR=./results
# optional: export FLMM_WEB_TASK_DB=./results/task_queue.db
# optional: export CUDA_VISIBLE_DEVICES=0 FLMM_WEB_DEVICE=cuda:0 FLMM_WEB_DEVICE_MAP=  # single-GPU, avoid auto split

python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5
# optional: --db /custom/path/task_queue.db
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
