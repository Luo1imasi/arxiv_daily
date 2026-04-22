"""通用工具函数"""

import hashlib
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ParamSpec, TypeVar

from loguru import logger
from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


def parallel_execute(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 8,
    desc: str = "Processing",
) -> list[R]:
    """并发执行函数并收集结果

    Args:
        func: 要执行的函数
        items: 输入项列表
        max_workers: 最大并发数
        desc: 进度条描述

    Returns:
        结果列表（顺序可能不同）
    """
    items_list = list(items)
    if not items_list:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"parallel_execute task failed: {e}")
    return results


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def make_content_key(title: str, abstract: str) -> str:
    payload = f"{normalize_text(title)}\n{normalize_text(abstract)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def retry_call(
    func: Callable[P, R],
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> R:
    """直接调用带重试的函数

    Args:
        func: 要执行的函数
        args: 函数参数
        kwargs: 函数关键字参数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exceptions: 需要捕获的异常类型

    Returns:
        函数返回值
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if not exceptions:
        raise ValueError("exceptions must contain at least one exception type")

    if kwargs is None:
        kwargs = {}
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
    logger.error(f"{func.__name__} failed after {max_retries} attempts")
    if last_exception is None:
        raise RuntimeError("retry_call exhausted without capturing an exception")
    raise last_exception
