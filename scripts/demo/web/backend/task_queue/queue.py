"""Task queue operations on top of SQLite."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional


TASK_TYPES = {"ASK", "GROUND", "ATTN_I2T", "ATTN_T2I"}
TASK_STATUSES = {"PENDING", "RUNNING", "DONE", "FAILED"}


def _serialize(obj: Any) -> str:
    return json.dumps(obj or {}, separators=(",", ":"))


def _deserialize(blob: Optional[str]) -> Optional[Any]:
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "type": row["type"],
        "status": row["status"],
        "session_id": row["session_id"],
        "turn_idx": row["turn_idx"],
        "turn_uid": row["turn_uid"],
        "input_json": _deserialize(row["input_json"]),
        "output_json": _deserialize(row["output_json"]),
        "error": row["error"],
        "worker_id": row["worker_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def enqueue_task(
    conn: sqlite3.Connection,
    task_type: str,
    session_id: str,
    input_obj: Any,
    *,
    turn_idx: Optional[int] = None,
    turn_uid: Optional[str] = None,
) -> int:
    if task_type not in TASK_TYPES:
        raise ValueError(f"Invalid task type: {task_type}")
    payload = _serialize(input_obj)
    cur = conn.execute(
        """
        INSERT INTO tasks (type, status, session_id, turn_idx, turn_uid, input_json)
        VALUES (?, 'PENDING', ?, ?, ?, ?)
        """,
        (task_type, session_id, turn_idx, turn_uid, payload),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_task(conn: sqlite3.Connection, task_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def claim_next_task(conn: sqlite3.Connection, worker_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        row = cur.execute(
            "SELECT * FROM tasks WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        cur.execute(
            "UPDATE tasks SET status = 'RUNNING', worker_id = ? WHERE id = ? AND status = 'PENDING'",
            (worker_id, row["id"]),
        )
        conn.commit()
        return get_task(conn, row["id"])
    except Exception:
        conn.rollback()
        raise


def mark_done(conn: sqlite3.Connection, task_id: int, output_obj: Any) -> None:
    payload = _serialize(output_obj)
    conn.execute(
        "UPDATE tasks SET status = 'DONE', output_json = ?, error = NULL WHERE id = ?",
        (payload, task_id),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, task_id: int, error: str) -> None:
    conn.execute(
        "UPDATE tasks SET status = 'FAILED', error = ? WHERE id = ?",
        (error, task_id),
    )
    conn.commit()
