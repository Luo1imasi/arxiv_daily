import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def temp_data_dir() -> Iterator[str]:
    old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["ARXIV_DAILY_DATA"] = tmpdir
        try:
            yield tmpdir
        finally:
            if old_data_dir is None:
                os.environ.pop("ARXIV_DAILY_DATA", None)
            else:
                os.environ["ARXIV_DAILY_DATA"] = old_data_dir


@contextmanager
def patched_attr(target: object, name: str, value: Any) -> Iterator[None]:
    old_value = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, old_value)
