import asyncio
import unittest

from arxiv_daily import database as db
from arxiv_daily.task_runner import TaskRunner
from tests.helpers import temp_data_dir


class TaskRunnerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_start_allows_only_one_task(self):
        with temp_data_dir():
            await db.init_db()
            runner = TaskRunner()
            started = asyncio.Event()
            release = asyncio.Event()

            async def work() -> str:
                started.set()
                await release.wait()
                return "done"

            async def start_one() -> str:
                try:
                    await runner.start("test", work, trigger="test")
                    return "started"
                except RuntimeError:
                    return "rejected"

            results = await asyncio.gather(start_one(), start_one())
            await started.wait()
            release.set()
            await runner.wait()

            self.assertEqual(sorted(results), ["rejected", "started"])

    async def test_completed_task_status_remains_visible_after_done_callback(self):
        with temp_data_dir():
            await db.init_db()
            runner = TaskRunner()

            async def work() -> str:
                return "done"

            run_id = await runner.start("test", work, trigger="test")
            await runner.wait()
            await asyncio.sleep(0)

            status = await runner.get_status()

            self.assertFalse(status["running"])
            self.assertTrue(status["done"])
            self.assertEqual(status["task_name"], "test")
            self.assertEqual(status["run_id"], run_id)
            self.assertIsNone(status["error"])
            self.assertEqual(status["latest_run"]["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
