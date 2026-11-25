"""Samples of rows shown in the report."""

from __future__ import annotations

import random
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect, collect_all


class Sample(BaseModel):
    """A sample of rows.

    Attributes:
        id: ``head``, ``tail``, ``random`` or ``custom``.
        data: the rows, as a ``pl.DataFrame``.
        name: title shown in the report.
        caption: optional caption.
        index: position of each row in the profiled frame, when known.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    data: Any
    name: str | None
    caption: str | None = None
    index: list[int] | None = None


def get_sample(config: Settings, df: pl.LazyFrame, n_rows: int) -> list[Sample]:
    """First, last and random rows (``samples.head``, ``.tail``, ``.random``).

    All samples are taken in one ``collect_all``; on a LazyFrame only the needed
    rows are materialised.
    """
    if n_rows == 0:
        return []
    queries: list[tuple[str, str, pl.LazyFrame, list[int]]] = []
    if config.samples.head > 0:
        k = min(config.samples.head, n_rows)
        queries.append(("head", "First rows", df.head(k), list(range(k))))
    if config.samples.tail > 0:
        k = min(config.samples.tail, n_rows)
        queries.append(
            ("tail", "Last rows", df.tail(k), list(range(n_rows - k, n_rows)))
        )
    if config.samples.random > 0:
        k = min(config.samples.random, n_rows)
        rows = sorted(random.sample(range(n_rows), k))
        query = (
            df.with_row_index("__row").filter(pl.col("__row").is_in(rows)).drop("__row")
        )
        queries.append(("random", "Random sample", query, rows))
    frames = collect_all([q for _, _, q, _ in queries])
    return [
        Sample(id=sample_id, data=frame, name=name, index=index)
        for (sample_id, name, _, index), frame in zip(queries, frames)
    ]


def get_custom_sample(sample: dict) -> list[Sample]:
    data = sample["data"]
    if isinstance(data, pl.LazyFrame):
        data = collect(data)
    return [
        Sample(
            id="custom",
            data=data,
            name=sample.get("name"),
            caption=sample.get("caption"),
        )
    ]
