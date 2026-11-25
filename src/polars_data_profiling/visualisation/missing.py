"""Missing-value charts."""

from __future__ import annotations

from typing import Any

from polars_data_profiling.config import Settings
from polars_data_profiling.visualisation import charts
from polars_data_profiling.visualisation.charts import builders


def plot_missing_matrix(
    config: Settings, notnull: Any, columns: list[str], nrows: int
) -> str:
    """Share of present values per column in bands of rows.

    Args:
        notnull: ``(bands, columns)`` array of the share of non-null values.
    """
    return charts.render(builders.missing_matrix(config, notnull, columns, nrows))


def plot_missing_bar(
    config: Settings, notnull_counts: list, nrows: int, columns: list[str]
) -> str:
    """Non-missing values per column."""
    return charts.render(builders.missing_bar(config, notnull_counts, nrows, columns))


def plot_missing_heatmap(
    config: Settings, corr_mat: Any, mask: Any, columns: list[str]
) -> str:
    """Nullity correlation heatmap."""
    return charts.render(builders.missing_heatmap(config, corr_mat, mask, columns))
