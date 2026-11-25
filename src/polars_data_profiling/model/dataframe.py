"""Preprocessing of the profiled frame."""

from __future__ import annotations

import polars as pl

from polars_data_profiling.config import Settings


def preprocess(config: Settings, df: pl.LazyFrame) -> pl.LazyFrame:
    """Prepare a frame for profiling (lazily).

    - NaN in float columns becomes null, so it counts as a missing value (as in
      pandas, where NaN is the missing value of floats).
    - In time-series mode, rows are sorted by ``vars.timeseries.sortby``.
    """
    schema = df.collect_schema()
    floats = [name for name, dtype in schema.items() if dtype.is_float()]
    if floats:
        df = df.with_columns(pl.col(floats).fill_nan(None))
    timeseries = config.vars.timeseries
    if timeseries.active and timeseries.sortby:
        df = df.sort(timeseries.sortby, nulls_last=True, maintain_order=True)
    return df
