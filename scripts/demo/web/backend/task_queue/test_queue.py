"""Minimal tests for SQLite task queue atomicity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    def test_json_serialization_numpy_scalars(self):
        task_id = enqueue_task(self.conn, "ASK", "sess3", {"foo": np.int64(7)}, turn_idx=2)
        claim_next_task(self.conn, "worker-c")
        mark_done(self.conn, task_id, {"value": np.int64(8), "vec": np.array([1, 2])})
        done = get_task(self.conn, task_id)
        self.assertEqual(done["output_json"], {"value": 8, "vec": [1, 2]})

    def test_requeue_same_turn_replaces_pending(self):
        # initial pending task
        first_id = enqueue_task(self.conn, "ASK", "sess4", {"q": "v1"}, turn_idx=0)
        # re-enqueue same turn should replace payload and reset status
        second_id = enqueue_task(self.conn, "ASK", "sess4", {"q": "v2"}, turn_idx=0)
        # IDs may or may not be the same depending on SQLite, but payload should reflect latest
        claimed = claim_next_task(self.conn, "worker-d")
        self.assertEqual(claimed["input_json"]["q"], "v2")
        # mark done to ensure no locking
        mark_done(self.conn, claimed["id"], {"answer": "ok"})

    def test_ground_same_turn_allows_multiple(self):
        enqueue_task(self.conn, "GROUND", "sess5", {"i": 1}, turn_idx=0)
        enqueue_task(self.conn, "GROUND", "sess5", {"i": 2}, turn_idx=0)
        first = claim_next_task(self.conn, "worker-e")
        second = claim_next_task(self.conn, "worker-e")
        ids = {first["id"], second["id"]}
        self.assertEqual(len(ids), 2)
        self.assertEqual({first["input_json"]["i"], second["input_json"]["i"]}, {1, 2})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
