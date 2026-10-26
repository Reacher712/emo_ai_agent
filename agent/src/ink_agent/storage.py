import json
import sqlite3
from threading import RLock
from time import time


class TaskStore:
    """Business lifecycle gate, independent of the graph's cached state."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = RLock()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                deadline REAL NOT NULL, blocked TEXT, published TEXT
            );
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, result TEXT NOT NULL
            );
        """)
        self.db.commit()

    def create(self, task_id, session_id, timeout):
        with self.lock, self.db:
            self.db.execute("INSERT INTO tasks VALUES (?, ?, ?, NULL, NULL)",
                            (task_id, session_id, time() + timeout))

    def _guard(self, task_id):
        row = self.db.execute("SELECT deadline, blocked, published FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return None if row[2] else row[1] or ("timed_out" if row[0] <= time() else None)

    def guard(self, task_id):
        with self.lock:
            return self._guard(task_id)

    def cancel(self, task_id):
        with self.lock, self.db:
            cursor = self.db.execute(
                "UPDATE tasks SET blocked='cancelled' WHERE id=? AND published IS NULL", (task_id,))
            if not cursor.rowcount:
                raise ValueError("Task does not exist or is already published")

    def block_session(self, session_id):
        with self.lock, self.db:
            self.db.execute("UPDATE tasks SET blocked='blocked' WHERE session_id=? AND published IS NULL",
                            (session_id,))

    def publish(self, task_id, artifact):
        with self.lock, self.db:
            # A transaction protects cancellation-vs-publication across processes too.
            self.db.execute("BEGIN IMMEDIATE")
            blocked = self._guard(task_id)
            if blocked:
                return blocked
            self.db.execute("UPDATE tasks SET published=? WHERE id=? AND published IS NULL",
                            (json.dumps(artifact, ensure_ascii=False), task_id))
        return None

    def operation(self, operation_id, factory):
        # Suitable for deterministic mock calls. Real remote backends must support
        # idempotency keys / submission lookup before this can be used for billing.
        with self.lock:
            row = self.db.execute("SELECT result FROM operations WHERE id=?", (operation_id,)).fetchone()
            if row:
                return json.loads(row[0])
        # Do not hold the business DB lock during slow image work: cancellation
        # and high-risk messages must be able to block publication immediately.
        result = factory()
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO operations VALUES (?, ?)",
                            (operation_id, json.dumps(result, ensure_ascii=False)))
            row = self.db.execute("SELECT result FROM operations WHERE id=?", (operation_id,)).fetchone()
            return json.loads(row[0])

    def owns_artifact(self, session_id, artifact_id):
        with self.lock:
            rows = self.db.execute("SELECT published FROM tasks WHERE session_id=? AND published IS NOT NULL",
                                   (session_id,)).fetchall()
            return any(json.loads(row[0])["artifact_id"] == artifact_id for row in rows)

    def close(self):
        self.db.close()

