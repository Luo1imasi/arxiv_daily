import asyncio
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from . import database as db


@dataclass
class _LiveTaskState:
    task: asyncio.Task[Any] | None = None
    task_name: str | None = None
    run_id: int | None = None


@dataclass
class _LastTaskState:
    done: bool = False
    task_name: str | None = None
    run_id: int | None = None
    error: str | None = None


class TaskRunner:
    def __init__(self):
        self._live = _LiveTaskState()
        self._last = _LastTaskState()
        self._start_lock = asyncio.Lock()

    def is_running(self) -> bool:
        return self._live.task is not None and not self._live.task.done()

    async def start(
        self,
        task_name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        trigger: str,
        metadata: dict[str, Any] | None = None,
        result_to_metrics: Callable[[Any], dict[str, Any]] | None = None,
    ) -> int:
        async with self._start_lock:
            if self.is_running():
                raise RuntimeError("A task is already running")

            run_id = await db.create_task_run(task_name, trigger, metadata=metadata)
            self._live = _LiveTaskState(task_name=task_name, run_id=run_id)
            self._last = _LastTaskState()

            def _remember_done(error: str | None) -> None:
                self._last = _LastTaskState(
                    done=True,
                    task_name=task_name,
                    run_id=run_id,
                    error=error,
                )

            async def _wrapped() -> Any:
                try:
                    result = await coro_factory()
                except asyncio.CancelledError:
                    error = "Task cancelled"
                    _remember_done(error)
                    await db.complete_task_run(run_id, "cancelled", error=error)
                    raise
                except Exception as exc:
                    error = str(exc)
                    _remember_done(error)
                    await db.complete_task_run(run_id, "failed", error=error)
                    raise
                else:
                    metrics = result_to_metrics(result) if result_to_metrics else None
                    await db.complete_task_run(run_id, "succeeded", metrics=metrics)
                    _remember_done(None)
                    return result

            self._live.task = asyncio.create_task(_wrapped())
            self._live.task.add_done_callback(self._on_done)
            return run_id

    async def wait(self) -> Any:
        if self._live.task is None:
            return None
        return await self._live.task

    async def get_status(self) -> dict[str, Any]:
        latest_run = await db.get_latest_task_run()
        status = {
            "running": False,
            "done": self._last.done,
            "error": self._last.error,
            "task_name": self._live.task_name or self._last.task_name,
            "run_id": self._live.run_id or self._last.run_id,
            "latest_run": latest_run,
        }
        if self._live.task is None:
            return status

        if self._live.task.done():
            if self._live.task.cancelled():
                status["done"] = True
                return status

            exc = self._live.task.exception()
            if exc:
                status["done"] = True
                status["error"] = str(exc)
                return status

            status["done"] = True
            status["error"] = None
            return status

        status["running"] = True
        status["error"] = None
        return status

    def _on_done(self, task: asyncio.Task[Any]):
        task_name = self._live.task_name
        if task.cancelled():
            logger.warning(f"Task cancelled: {task_name}")
        else:
            exc = task.exception()
            if exc:
                logger.error(
                    f"Task failed: {exc}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )

        if self._live.task is task:
            self._live = _LiveTaskState()
