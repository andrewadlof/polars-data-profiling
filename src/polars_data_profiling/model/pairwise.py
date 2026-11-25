"""Scatter plots of pairs of numeric variables (interactions)."""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect_all
from polars_data_profiling.visualisation.plot import scatter_pairwise

# Points drawn per scatter plot. Larger samples are drawn as hexbins, whose
# density a uniform sample preserves.
MAX_SCATTER_POINTS = 10_000


def get_scatter_tasks(
    config: Settings, continuous_variables: list
) -> list[tuple[Any, Any]]:
    if not config.interactions.continuous:
        return []
    targets = config.interactions.targets
    if len(targets) == 0:
        targets = continuous_variables
    return [(x, y) for y in continuous_variables for x in targets]


def get_scatter_data(
    df: pl.LazyFrame,
    tasks: list[tuple[Any, Any]],
    continuous_variables: list,
    n_rows: int,
) -> dict[tuple[Any, Any], pl.DataFrame]:
    """The complete (x, y) pairs of every task, collected together.

    Frames with more than ``MAX_SCATTER_POINTS`` rows are thinned lazily to
    every k-th row, so a large LazyFrame is never collected whole.
    """
    stride = max(1, -(-n_rows // MAX_SCATTER_POINTS))
    queries = {}
    for x, y in tasks:
        if x not in continuous_variables:
            continue
        cols = [x] if x == y else [x, y]
        query = df.select(pl.col(c).cast(pl.Float64) for c in cols)
        if stride > 1:
            query = query.gather_every(stride)
        queries[(x, y)] = query.drop_nulls()
    frames = collect_all(list(queries.values())) if queries else []
    return dict(zip(queries, frames))


def get_scatter_plot(
    config: Settings, data: pl.DataFrame | None, x: Any, y: Any
) -> str:
    if data is None:
        return ""
    return scatter_pairwise(config, data.get_column(x), data.get_column(y), x, y)
