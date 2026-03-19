"""SQLite helpers for the task queue."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _tasks_table_sql(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    ).fetchone()
    if row is None:
        return None
    return row[0]


def _migrate_tasks_table_if_needed(conn: sqlite3.Connection) -> None:
    sql = _tasks_table_sql(conn)
    if sql is None:
        return
    if (
        "TOKEN_TO_REGION" in sql
        and "REGION_TO_TOKEN" in sql
        and "ATTN_I2T" not in sql
        and "ATTN_T2I" not in sql
    ):
        return
    conn.executescript(
        """
        PRAGMA foreign_keys=OFF;
        BEGIN IMMEDIATE;
        DROP TRIGGER IF EXISTS trg_tasks_updated_at;
        DROP INDEX IF EXISTS idx_tasks_status_created;
        DROP INDEX IF EXISTS idx_tasks_status;
        DROP INDEX IF EXISTS idx_tasks_turn_unique;
        DROP INDEX IF EXISTS idx_tasks_ask_turn_unique;
        CREATE TABLE tasks_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL CHECK (
                type IN (
                    'ASK',
                    'GROUND',
                    'TOKEN_TO_REGION',
                    'REGION_TO_TOKEN'
                )
            ),
            status TEXT NOT NULL CHECK (
                status IN ('PENDING','RUNNING','DONE','FAILED')
            ),
            session_id TEXT NOT NULL,
            turn_idx INTEGER,
            turn_uid TEXT,
            input_json TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            worker_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        INSERT INTO tasks_new (
            id,
            type,
            status,
            session_id,
            turn_idx,
            turn_uid,
            input_json,
            output_json,
            error,
            worker_id,
            created_at,
            updated_at
        )
        SELECT
            id,
            CASE
                WHEN type = 'ATTN_I2T' THEN 'TOKEN_TO_REGION'
                WHEN type = 'ATTN_T2I' THEN 'REGION_TO_TOKEN'
                ELSE type
            END,
            status,
            session_id,
            turn_idx,
            turn_uid,
            input_json,
            output_json,
            error,
            worker_id,
            created_at,
            updated_at
        FROM tasks;
        DROP TABLE tasks;
        ALTER TABLE tasks_new RENAME TO tasks;
        COMMIT;
        PRAGMA foreign_keys=ON;
        """
    )


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with queue-friendly pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection, schema_path: Optional[Path] = None) -> None:
    """Initialize the database using the bundled schema."""
    schema_file = schema_path or SCHEMA_PATH
    _migrate_tasks_table_if_needed(conn)
    sql = schema_file.read_text(encoding="utf-8")
    conn.executescript(sql)
