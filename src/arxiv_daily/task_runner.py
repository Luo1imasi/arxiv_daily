import asyncio
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from . import database as db


class TaskRunner:
    def __init__(self):
        self._task: asyncio.Task | None = None
        self._task_name: str | None = None
        self._last_error: str | None = None
        self._current_run_id: int | None = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(
        self,
        task_name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        trigger: str,
        metadata: dict[str, Any] | None = None,
        result_to_metrics: Callable[[Any], dict[str, Any]] | None = None,
    ) -> int:
        if self.is_running():
            raise RuntimeError("A task is already running")

        run_id = await db.create_task_run(task_name, trigger, metadata=metadata)
        self._task_name = task_name
        self._last_error = None
        self._current_run_id = run_id

        async def _wrapped() -> Any:
            try:
                result = await coro_factory()
            except asyncio.CancelledError:
                self._last_error = "Task cancelled"
                await db.complete_task_run(run_id, "cancelled", error=self._last_error)
                raise
            except Exception as exc:
                self._last_error = str(exc)
                await db.complete_task_run(run_id, "failed", error=self._last_error)
                raise
            else:
                metrics = result_to_metrics(result) if result_to_metrics else None
                await db.complete_task_run(run_id, "succeeded", metrics=metrics)
                self._last_error = None
                return result

        self._task = asyncio.create_task(_wrapped())
        self._task.add_done_callback(self._on_done)
        return run_id

    async def wait(self) -> Any:
        if self._task is None:
            return None
        return await self._task

    async def get_status(self) -> dict[str, Any]:
        latest_run = await db.get_latest_task_run()
        status = {
            "running": False,
            "done": False,
            "error": self._last_error,
            "task_name": self._task_name,
            "run_id": self._current_run_id,
            "latest_run": latest_run,
        }
        if self._task is None:
            return status

        if self._task.done():
            if self._task.cancelled():
                status["done"] = True
                return status

            exc = self._task.exception()
            if exc:
                self._last_error = str(exc)
                status["done"] = True
                status["error"] = self._last_error
                return status

            self._last_error = None
            status["done"] = True
            status["error"] = None
            return status

        status["running"] = True
        status["error"] = None
        return status

    def _on_done(self, task: asyncio.Task[Any]):
        task_name = self._task_name
        if task.cancelled():
            logger.warning(f"Task cancelled: {task_name}")
        else:
            exc = task.exception()
            if exc:
                logger.error(
                    f"Task failed: {exc}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )

        if self._task is task:
            self._task = None
            self._task_name = None
            self._current_run_id = None
