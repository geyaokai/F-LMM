#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for the runtime backend + worker task queue."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:9000",
        help="Backend base URL.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Image path visible to the runtime host. Required for ASK task validation.",
    )
    parser.add_argument(
        "--question",
        default="What is under the mirror?",
        help="Question used for the queued ASK task.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum time to wait for a terminal task state.",
    )
    parser.add_argument(
        "--allow-failed-task",
        action="store_true",
        help="Treat FAILED as a completed smoke test when the error is preserved.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for a JSON summary.",
    )
    parser.add_argument(
        "--cleanup-session",
        action="store_true",
        help="Delete the temporary session after the smoke test.",
    )
    return parser


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Tuple[int, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        status = resp.getcode()
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = body
    return status, parsed


def expect_status(step: str, status: int, payload: Any, expected: int = 200) -> None:
    if status != expected:
        raise RuntimeError(f"{step} HTTP {status}: {payload}")
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"{step} returned error payload: {payload}")


def classify_task_error(error_text: str) -> str:
    lowered = (error_text or "").lower()
    if "cuda out of memory" in lowered:
        return "environment_cuda_oom"
    if "no such file" in lowered or "not found" in lowered:
        return "runtime_path_or_asset_error"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "runtime_import_error"
    if "permission denied" in lowered:
        return "runtime_permission_error"
    if lowered:
        return "worker_or_model_error"
    return "unknown_error"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    summary: Dict[str, Any] = {
        "base_url": base_url,
        "steps": [],
        "session_id": None,
        "task_id": None,
        "task_status": None,
        "task_error": None,
        "task_error_classification": None,
    }

    try:
        status, payload = request_json(base_url, "/healthz", timeout=args.timeout)
        expect_status("healthz", status, payload)
        summary["steps"].append("healthz")

        create_payload: Dict[str, Any] = {}
        status, payload = request_json(
            base_url,
            "/session/create",
            method="POST",
            payload=create_payload,
            timeout=args.timeout,
        )
        expect_status("session/create", status, payload)
        session_id = (payload or {}).get("data", {}).get("session_id")
        if not session_id:
            raise RuntimeError(f"session/create did not return session_id: {payload}")
        summary["session_id"] = session_id
        summary["session_dir_url"] = (payload or {}).get("data", {}).get("session_dir_url")
        summary["steps"].append("session/create")

        if not args.image_path:
            raise RuntimeError("--image-path is required for /tasks ASK smoke test.")

        task_payload = {
            "type": "ASK",
            "session_id": session_id,
            "payload": {
                "question": args.question,
                "image_path": args.image_path,
            },
        }
        status, payload = request_json(
            base_url,
            "/tasks",
            method="POST",
            payload=task_payload,
            timeout=args.timeout,
        )
        expect_status("tasks enqueue", status, payload)
        task_id = (payload or {}).get("data", {}).get("task_id")
        if task_id is None:
            raise RuntimeError(f"/tasks did not return task_id: {payload}")
        summary["task_id"] = task_id
        summary["steps"].append("tasks")

        deadline = time.time() + args.timeout
        last_task_payload: Dict[str, Any] = {}
        while time.time() < deadline:
            status, payload = request_json(
                base_url,
                f"/tasks/{task_id}",
                timeout=args.timeout,
            )
            expect_status("tasks poll", status, payload)
            last_task_payload = (payload or {}).get("data", {}).get("task", {}) or {}
            task_status = last_task_payload.get("status")
            if task_status in {"DONE", "FAILED"}:
                summary["task_status"] = task_status
                summary["steps"].append("tasks/poll")
                break
            time.sleep(max(args.poll_interval, 0.1))
        else:
            raise RuntimeError(
                f"Task {task_id} did not reach DONE/FAILED within {args.timeout:.1f}s."
            )

        error_text = str(last_task_payload.get("error") or "")
        summary["task_error"] = error_text or None
        if error_text:
            summary["task_error_classification"] = classify_task_error(error_text)
        if summary["task_status"] == "FAILED" and not args.allow_failed_task:
            raise RuntimeError(
                f"Task {task_id} failed and --allow-failed-task was not set: {error_text}"
            )

        if args.cleanup_session and session_id:
            status, payload = request_json(
                base_url,
                f"/session/{session_id}",
                method="DELETE",
                timeout=args.timeout,
            )
            expect_status("session delete", status, payload)
            summary["steps"].append("session/delete")

        if args.json_output:
            write_json(Path(args.json_output), summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except urllib.error.URLError as exc:
        print(f"[runtime-smoke] request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        if args.json_output:
            summary["error"] = str(exc)
            write_json(Path(args.json_output), summary)
        print(f"[runtime-smoke] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
