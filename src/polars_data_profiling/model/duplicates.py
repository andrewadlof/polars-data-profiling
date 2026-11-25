"""Duplicate rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect_all


def get_duplicates(
    config: Settings,
    df: pl.LazyFrame,
    supported_columns: Sequence[str],
    n_rows: int,
) -> tuple[dict[str, Any], pl.DataFrame | None]:
    """The most frequent duplicated rows, computed with one lazy group-by.

    Args:
        config: report settings.
        df: the frame.
        supported_columns: the columns that identify a row.
        n_rows: number of rows of the frame.

    Returns:
        ``n_duplicates`` (the number of distinct rows that occur more than once)
        and ``p_duplicates``, and the ``duplicates.head`` most frequent of these
        rows with their number of occurrences.
    """
    n_head = config.duplicates.head
    metrics: dict[str, Any] = {}
    if n_head <= 0:
        return metrics, None
    schema = df.collect_schema()
    key = config.duplicates.key
    if key in schema:
        raise ValueError(
            f"Duplicates key ({key}) may not be part of the DataFrame. Either change the "
            f" column name in the DataFrame or change the 'duplicates.key' parameter."
        )
    columns = list(supported_columns)
    if not columns:
        metrics["n_duplicates"] = 0
        metrics["p_duplicates"] = 0.0
        return metrics, None

    groups = df.group_by(columns).len(name=key).filter(pl.col(key) > 1)
    if n_rows == 0:
        metrics["n_duplicates"] = 0
        metrics["p_duplicates"] = 0.0
        return metrics, None
    counts, top = collect_all(
        [
            groups.select(pl.len()),
            groups.sort(key, descending=True, maintain_order=True).head(n_head),
        ]
    )
    n_duplicates = counts.item()
    metrics["n_duplicates"] = int(n_duplicates)
    metrics["p_duplicates"] = n_duplicates / n_rows
    return metrics, top
