"""Missing-value diagrams: counts, matrix and nullity correlation."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect
from polars_data_profiling.visualisation.charts.builders import MAX_MATRIX_BANDS
from polars_data_profiling.visualisation.missing import (
    plot_missing_bar,
    plot_missing_heatmap,
    plot_missing_matrix,
)


def missing_bar(
    config: Settings, df: pl.LazyFrame, n_rows: int, variables: dict
) -> str:
    """Non-missing values per column (from the column descriptions)."""
    columns = list(variables)
    notnull = [n_rows - variables[c].get("n_missing", 0) for c in columns]
    return plot_missing_bar(
        config, notnull_counts=notnull, nrows=n_rows, columns=columns
    )


def missing_matrix(
    config: Settings, df: pl.LazyFrame, n_rows: int, variables: dict
) -> str:
    """Share of non-missing values per column in horizontal bands of rows.

    The rows are split into at most ``MAX_MATRIX_BANDS`` bands and the shares
    are computed with one lazy group-by, so only a small matrix is collected.
    """
    columns = list(variables)
    n_bands = max(1, min(MAX_MATRIX_BANDS, n_rows))
    band = (pl.int_range(pl.len(), dtype=pl.Int64) * n_bands // max(n_rows, 1)).alias(
        "__band"
    )
    shares = collect(
        df.select(band, *[pl.col(c).is_not_null().cast(pl.Float64) for c in columns])
        .group_by("__band")
        .mean()
        .sort("__band")
    )
    bands = shares.get_column("__band").to_numpy()
    fractions = np.ones((n_bands, len(columns)))
    fractions[bands] = shares.select(columns).to_numpy()
    return plot_missing_matrix(config, notnull=fractions, columns=columns, nrows=n_rows)


def missing_heatmap(
    config: Settings, df: pl.LazyFrame, n_rows: int, variables: dict
) -> str:
    """Nullity correlation between the partially missing columns."""
    columns = [
        name
        for name, summary in variables.items()
        if 0 < summary.get("n_missing", 0) < n_rows
    ]
    k = len(columns)
    pairs = list(itertools.combinations(range(k), 2))
    exprs = [
        pl.corr(
            pl.col(columns[i]).is_null().cast(pl.Float64),
            pl.col(columns[j]).is_null().cast(pl.Float64),
        ).alias(f"__c{n}")
        for n, (i, j) in enumerate(pairs)
    ]
    matrix = np.eye(k)
    if exprs:
        row = collect(df.select(exprs)).row(0)
        for (i, j), value in zip(pairs, row):
            value = np.nan if value is None else value
            matrix[i, j] = matrix[j, i] = value
    corr = pl.DataFrame(matrix, schema=columns, orient="row")
    mask = np.zeros_like(matrix)
    mask[np.triu_indices_from(mask)] = True
    return plot_missing_heatmap(config, corr_mat=corr, mask=mask, columns=columns)


def get_missing_active(config: Settings, table_stats: dict) -> dict[str, Any]:
    """The missing-value diagrams to draw, given the table statistics."""
    missing_map = {
        "bar": {
            "min_missing": 0,
            "name": "Count",
            "caption": "A simple visualization of nullity by column.",
            "function": missing_bar,
        },
        "matrix": {
            "min_missing": 0,
            "name": "Matrix",
            "caption": "Nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.",
            "function": missing_matrix,
        },
        "heatmap": {
            "min_missing": 2,
            "name": "Heatmap",
            "caption": "The correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another.",
            "function": missing_heatmap,
        },
    }

    return {
        name: settings
        for name, settings in missing_map.items()
        if (
            config.missing_diagrams[name]
            and table_stats["n_vars_with_missing"] >= settings["min_missing"]
        )
        and (
            name != "heatmap"
            or (
                table_stats["n_vars_with_missing"] - table_stats["n_vars_all_missing"]
                >= settings["min_missing"]
            )
        )
    }


def get_missing_diagram(
    config: Settings,
    df: pl.LazyFrame,
    n_rows: int,
    variables: dict,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Draw one missing-value diagram.

    Returns:
        The diagram's name, caption and rendered chart, or None if it failed.
    """
    function: Callable | None = settings.get("function")
    if function is None:
        return None
    try:
        result = function(config, df, n_rows, variables)
    except (ValueError, pl.exceptions.PolarsError) as e:
        warnings.warn(
            f"""There was an attempt to generate the {settings["name"]} missing values diagrams, but this failed.
To hide this warning, disable the calculation
(using `ProfileReport(df, missing_diagrams={{"{settings["name"].lower()}": False}})`)
If this is problematic for your use case, please report this as an issue:
https://github.com/andrewadlof/polars-data-profiling/issues
(include the error message: '{e}')""",
            stacklevel=2,
        )
        return None
    return {
        "name": settings["name"],
        "caption": settings["caption"],
        "matrix": result,
    }
