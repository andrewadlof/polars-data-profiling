"""Collecting lazy queries with the engine chosen in the settings.

Every query of a profile is collected through :func:`collect` and
:func:`collect_all`, which use the Polars engine of ``engine.collect_engine``
("auto", "in-memory", "streaming" or "gpu"). :func:`describe` sets it for the
duration of a profile with :func:`use_engine`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import polars as pl

_ENGINE: ContextVar[Any] = ContextVar("polars_data_profiling_engine", default="auto")


@contextmanager
def use_engine(engine: Any) -> Iterator[None]:
    """Collect with ``engine`` inside the block."""
    token = _ENGINE.set(engine)
    try:
        yield
    finally:
        _ENGINE.reset(token)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect one query."""
    return lf.collect(engine=_ENGINE.get())


def collect_all(lfs: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Collect several queries together, sharing common sub-plans."""
    lfs = list(lfs)
    if not lfs:
        return []
    return pl.collect_all(lfs, engine=_ENGINE.get())
