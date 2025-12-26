#!/usr/bin/env python3
"""轻量本地自检脚本：依次调用 healthz -> create_session -> (可选)load_image -> ask -> clear -> delete_session。"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def request_json(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Tuple[int, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        status = resp.getcode()
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = body
    return status, parsed


def expect_ok(step: str, status: int, payload: Any):
    if status != 200:
        raise RuntimeError(f"{step} HTTP {status}: {payload}")
    if isinstance(payload, dict) and payload.get("status") not in (None, "ok"):
        raise RuntimeError(f"{step} error: {payload}")


def main():
    parser = argparse.ArgumentParser(description="F-LMM Web Backend 本地冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000", help="后端服务地址")
    parser.add_argument("--image-path", default=None, help="可选：指定本地图片路径，自动 load_image")
    parser.add_argument("--question", default="What is in the image?", help="提问内容，默认非空以避免后端报错")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    def log(msg: str):
        print(f"[test] {msg}")

    try:
        # 1) health check
        log("GET /healthz")
        status, payload = request_json(f"{base}/healthz")
        expect_ok("healthz", status, payload)
        log(f"healthz ok -> {payload}")

        # 2) create session
        create_body: Dict[str, Any] = {}
        if args.image_path:
            create_body["image_path"] = args.image_path
        log("POST /session/create")
        status, payload = request_json(f"{base}/session/create", method="POST", payload=create_body)
        expect_ok("create_session", status, payload)
        session_id = (payload or {}).get("data", {}).get("session_id")
        if not session_id:
            raise RuntimeError(f"create_session 未返回 session_id: {payload}")
        log(f"session created: {session_id}")

        # 3) optionally load image (if not loaded via create)
        if args.image_path and not create_body:
            # create_body 为空时才需要手动 load_image，这里保证 image_path 可以重复加载也没问题
            log("POST /load_image")
            load_body = {"session_id": session_id, "image_path": args.image_path}
            status, payload = request_json(f"{base}/load_image", method="POST", payload=load_body)
            expect_ok("load_image", status, payload)
            log("image loaded")

        # 4) ask
        log("POST /ask")
        ask_body = {"session_id": session_id, "question": args.question}
        status, payload = request_json(f"{base}/ask", method="POST", payload=ask_body)
        expect_ok("ask", status, payload)
        answer = (payload or {}).get("data", {}).get("answer")
        log(f"ask ok, answer: {answer}")

        # 5) clear history
        log("POST /clear")
        clear_body = {"session_id": session_id}
        status, payload = request_json(f"{base}/clear", method="POST", payload=clear_body)
        expect_ok("clear", status, payload)
        log("history cleared")

        # 6) delete session
        log("DELETE /session/{session_id}")
        status, payload = request_json(f"{base}/session/{session_id}", method="DELETE")
        expect_ok("delete_session", status, payload)
        log("session deleted, test completed")

    except urllib.error.URLError as exc:
        print(f"[error] 请求失败: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
