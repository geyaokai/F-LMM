"""SQLite helpers for the task queue."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with queue-friendly pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection, schema_path: Optional[Path] = None) -> None:
    """Initialize the database using the bundled schema."""
    schema_file = schema_path or SCHEMA_PATH
    sql = schema_file.read_text(encoding="utf-8")
    conn.executescript(sql)
