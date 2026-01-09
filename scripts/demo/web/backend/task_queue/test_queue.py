"""Minimal tests for SQLite task queue atomicity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .db import connect, init_db
from .queue import claim_next_task, enqueue_task, get_task, mark_done, mark_failed


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "queue.db"
        self.conn = connect(self.db_path)
        init_db(self.conn)

    def tearDown(self):
        try:
            self.conn.close()
        finally:
            self.tmp.cleanup()

    def test_enqueue_and_claim_once(self):
        task_id = enqueue_task(self.conn, "ASK", "sess1", {"q": "hello"}, turn_idx=0)
        pending = get_task(self.conn, task_id)
        self.assertEqual(pending["status"], "PENDING")

        claimed = claim_next_task(self.conn, "worker-a")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], task_id)
        self.assertEqual(claimed["status"], "RUNNING")
        self.assertEqual(claimed["worker_id"], "worker-a")

        # No second claim while running
        nothing = claim_next_task(self.conn, "worker-a")
        self.assertIsNone(nothing)

        mark_done(self.conn, task_id, {"answer": "ok"})
        done = get_task(self.conn, task_id)
        self.assertEqual(done["status"], "DONE")
        self.assertEqual(done["output_json"], {"answer": "ok"})
        self.assertIsNone(done["error"])

    def test_mark_failed(self):
        task_id = enqueue_task(self.conn, "GROUND", "sess2", {"indices": [0, 1]}, turn_idx=1)
        claim_next_task(self.conn, "worker-b")
        mark_failed(self.conn, task_id, "boom")
        failed = get_task(self.conn, task_id)
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error"], "boom")
        self.assertIsNone(failed["output_json"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
