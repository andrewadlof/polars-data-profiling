"""Description of the time index in time-series mode."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect


def get_time_index_description(
    config: Settings, df: pl.LazyFrame, table_stats: dict
) -> dict:
    """Start, end and average period of the time index.

    The index is the ``vars.timeseries.sortby`` column, or the row number.
    """
    sortby = config.vars.timeseries.sortby
    if sortby:
        dtype = df.collect_schema()[sortby]
        if not (dtype.is_numeric() or dtype.is_temporal()):
            return {}
        index = pl.col(sortby)
    else:
        dtype = pl.Int64
        index = pl.int_range(pl.len(), dtype=pl.Int64)
    if dtype.is_temporal():
        steps = index.cast(pl.Datetime("us")).diff().abs().dt.total_microseconds()
    else:
        steps = index.cast(pl.Float64).diff().abs()
    start, end, period = collect(
        df.select(
            index.min().alias("start"),
            index.max().alias("end"),
            steps.mean().alias("p"),
        )
    ).row(0)
    if dtype.is_temporal() and period is not None:
        period = timedelta(microseconds=period)
    return {
        "n_series": table_stats["types"].get("TimeSeries", 0),
        "length": table_stats["n"],
        "start": start,
        "end": end,
        "period": period,
    }
