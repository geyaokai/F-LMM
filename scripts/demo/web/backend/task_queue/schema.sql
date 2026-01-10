-- SQLite schema for async task queue
-- Supports types ASK|GROUND|ATTN_I2T|ATTN_T2I and statuses PENDING|RUNNING|DONE|FAILED

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Ensure we recreate the unique index if the definition changes
DROP INDEX IF EXISTS idx_tasks_turn_unique;
DROP INDEX IF EXISTS idx_tasks_ask_turn_unique;

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK (type IN ('ASK','GROUND','ATTN_I2T','ATTN_T2I')),
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','DONE','FAILED')),
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

CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks (status, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
-- Only ASK should be unique per turn; other types may have multiple entries per turn
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_ask_turn_unique
ON tasks (session_id, turn_idx)
WHERE type = 'ASK';

CREATE TRIGGER IF NOT EXISTS trg_tasks_updated_at
AFTER UPDATE ON tasks
BEGIN
    UPDATE tasks SET updated_at = (strftime('%Y-%m-%dT%H:%M:%fZ','now')) WHERE id = NEW.id;
END;
