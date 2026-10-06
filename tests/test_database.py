import os
import sqlite3
import tempfile
import unittest

from arxiv_daily import database as db


class DatabaseMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_migration_keeps_one_row_per_content_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            path = os.path.join(tmpdir, "arxiv_daily.db")
            try:
                connection = sqlite3.connect(path)
                connection.execute(
                    """
                    CREATE TABLE embedding_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        abstract TEXT,
                        content_key TEXT,
                        model TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TEXT,
                        UNIQUE(title, abstract, model)
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO embedding_cache (title, abstract, content_key, model, embedding) VALUES (?, ?, ?, ?, ?)",
                    ("Older title", "same abstract", "content-1", "model-a", b"old"),
                )
                connection.execute(
                    "INSERT INTO embedding_cache (title, abstract, content_key, model, embedding) VALUES (?, ?, ?, ?, ?)",
                    ("Newer title", "different abstract", "content-1", "model-a", b"new"),
                )
                connection.commit()
                connection.close()

                await db.init_db()

                connection = sqlite3.connect(path)
                rows = connection.execute(
                    "SELECT title, embedding FROM embedding_cache WHERE content_key = ?",
                    ("content-1",),
                ).fetchall()
                table_sql = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'embedding_cache'"
                ).fetchone()[0]
                connection.close()

                self.assertEqual(rows, [("Newer title", b"new")])
                normalized = "".join(table_sql.split()).lower()
                self.assertIn("unique(content_key,model)", normalized)
                self.assertNotIn("unique(title,abstract,model)", normalized)
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

    async def test_startup_fails_orphaned_running_tasks_and_purges_expired_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.create_task_run("daily recommendation", "manual")
                await db.save_candidate_cache("expired", [{"title": "old"}], ttl_minutes=1)
                path = os.path.join(tmpdir, "arxiv_daily.db")
                connection = sqlite3.connect(path)
                connection.execute(
                    "UPDATE candidate_cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE cache_key = 'expired'"
                )
                connection.commit()
                connection.close()

                deleted = await db.purge_expired_candidate_cache()
                failed = await db.fail_orphaned_running_tasks("Process restarted before the task finished")
                latest = await db.get_latest_task_run()

                self.assertEqual(deleted, 1)
                self.assertEqual(failed, 1)
                self.assertIsNotNone(latest)
                self.assertEqual(latest["status"], "failed")
                self.assertIn("restarted", latest["error"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

    async def test_feedback_vote_is_stored_once_per_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.upsert_feedback(
                    url="https://example.com/paper",
                    vote="relevant",
                    title="Useful paper",
                )
                await db.upsert_feedback(
                    url="https://example.com/paper",
                    vote="irrelevant",
                    title="Useful paper",
                )
                rows = await db.list_feedback()

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["vote"], "irrelevant")
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


if __name__ == "__main__":
    unittest.main()
