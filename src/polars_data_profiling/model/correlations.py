"""Correlations between variables.

Correlation matrices are square ``polars.DataFrame``s: one column per variable,
rows in the same order as the columns.

- ``pearson`` and ``spearman``: every pair in one lazy select, each on the rows
  where both values are present (pandas' pairwise-complete semantics).
- ``kendall``: scipy's tau-b on the materialised numeric columns.
- ``cramers``: Cramér's V (bias corrected) from lazy cross-tabulations.
- ``auto``: Spearman between numeric variables, Cramér's V when a categorical
  variable is involved (numeric variables are first cut into
  ``correlations.auto.n_bins`` equal-width bins).

``phi_k`` from the pandas version is not available: the phik package requires
pandas.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
from scipy import stats

from polars_data_profiling.config import Settings
from polars_data_profiling.utils.collect import collect, collect_all

NUMERIC_TYPES = {"Numeric", "TimeSeries"}
CATEGORICAL_TYPES = {"Categorical", "Boolean"}


def _matrix_frame(columns: Sequence[str], matrix: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(matrix, schema=list(columns), orient="row")


def _pairwise(
    df: pl.LazyFrame,
    columns: Sequence[str],
    pair_expr: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> pl.DataFrame:
    """A symmetric matrix of a pairwise statistic, in one select."""
    pairs = list(itertools.combinations(range(len(columns)), 2))
    exprs = []
    for n, (i, j) in enumerate(pairs):
        a, b = pl.col(columns[i]), pl.col(columns[j])
        complete = a.is_not_null() & b.is_not_null()
        exprs.append(pair_expr(a.filter(complete), b.filter(complete)).alias(f"__p{n}"))
    # A constant column has no correlation, not even with itself (as pandas)
    exprs += [
        pl.col(c).cast(pl.Float64).std().alias(f"__s{n}") for n, c in enumerate(columns)
    ]
    row = collect(df.select(exprs)).row(0)
    matrix = np.eye(len(columns))
    for (i, j), value in zip(pairs, row):
        matrix[i, j] = matrix[j, i] = np.nan if value is None else value
    stds = row[len(pairs) :]
    for i, std in enumerate(stds):
        if not std:
            matrix[i, :] = matrix[:, i] = np.nan
    return _matrix_frame(columns, matrix)


def _numeric_columns(summary: dict, min_distinct: int = 0) -> list[str]:
    return [
        name
        for name, value in summary.items()
        if value["type"] in NUMERIC_TYPES and value.get("n_distinct", 0) > min_distinct
    ]


def _numeric_frame(df: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
    return df.select(pl.col(c).cast(pl.Float64) for c in columns)


def pearson_compute(
    config: Settings, df: pl.LazyFrame, summary: dict
) -> pl.DataFrame | None:
    columns = _numeric_columns(summary)
    if not columns:
        return None
    return _pairwise(
        _numeric_frame(df, columns),
        columns,
        lambda a, b: pl.corr(a, b, method="pearson"),
    )


def spearman_compute(
    config: Settings, df: pl.LazyFrame, summary: dict
) -> pl.DataFrame | None:
    columns = _numeric_columns(summary)
    if not columns:
        return None
    return _pairwise(
        _numeric_frame(df, columns),
        columns,
        lambda a, b: pl.corr(a, b, method="spearman"),
    )


def kendall_compute(
    config: Settings, df: pl.LazyFrame, summary: dict
) -> pl.DataFrame | None:
    columns = _numeric_columns(summary)
    if not columns:
        return None
    data = collect(_numeric_frame(df, columns))
    matrix = np.eye(len(columns))
    for i, j in itertools.combinations(range(len(columns)), 2):
        pair = data.select(columns[i], columns[j]).drop_nulls()
        if pair.height < 2:
            value = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = stats.kendalltau(
                    pair.get_column(columns[i]).to_numpy(),
                    pair.get_column(columns[j]).to_numpy(),
                ).statistic
        matrix[i, j] = matrix[j, i] = value
    for i, name in enumerate(columns):
        if summary[name].get("n_distinct", 0) <= 1:
            matrix[i, :] = matrix[:, i] = np.nan
    return _matrix_frame(columns, matrix)


def cramers_corrected_stat(table: np.ndarray, correction: bool = True) -> float:
    """Cramér's V with the bias correction of Bergsma (2013).

    Args:
        table: contingency table of two variables.
        correction: Yates' continuity correction of the chi-squared statistic.
    """
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.size == 0:
        return 0
    chi2 = stats.chi2_contingency(table, correction=correction)[0]
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
        rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
        kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
        rkcorr = min((kcorr - 1.0), (rcorr - 1.0))
        if rkcorr == 0.0:
            return 1.0
        return float(np.sqrt(phi2corr / rkcorr))


def _crosstabs(df: pl.LazyFrame, pairs: Sequence[tuple[str, str]]) -> list[np.ndarray]:
    """Dense contingency tables of several column pairs, collected together."""
    queries = [
        df.select(pl.col(a).alias("a"), pl.col(b).alias("b"))
        .drop_nulls()
        .group_by("a", "b")
        .len()
        for a, b in pairs
    ]
    tables = []
    for counts in collect_all(queries):
        rows = {v: i for i, v in enumerate(counts.get_column("a").unique().to_list())}
        cols = {v: i for i, v in enumerate(counts.get_column("b").unique().to_list())}
        table = np.zeros((len(rows), len(cols)))
        for a, b, n in counts.iter_rows():
            table[rows[a], cols[b]] = n
        tables.append(table)
    return tables


def _categorical_columns(config: Settings, summary: dict) -> list[str]:
    threshold = config.categorical_maximum_correlation_distinct
    return [
        name
        for name, value in summary.items()
        if value["type"] in CATEGORICAL_TYPES
        and 1 < value.get("n_distinct", 0) <= threshold
    ]


def cramers_compute(
    config: Settings, df: pl.LazyFrame, summary: dict
) -> pl.DataFrame | None:
    columns = sorted(_categorical_columns(config, summary))
    if len(columns) <= 1:
        return None
    pairs = list(itertools.combinations(range(len(columns)), 2))
    tables = _crosstabs(df, [(columns[i], columns[j]) for i, j in pairs])
    matrix = np.eye(len(columns))
    for (i, j), table in zip(pairs, tables):
        value = cramers_corrected_stat(table) if table.size else np.nan
        matrix[i, j] = matrix[j, i] = value
    return _matrix_frame(columns, matrix)


def _discretized(df: pl.LazyFrame, column: str, n_bins: int) -> pl.Expr:
    """Equal-width bins over [min, max], labelled 0..n_bins-1 (as pd.cut)."""
    x = pl.col(column).cast(pl.Float64)
    low, high = x.min(), x.max()
    width = (high - low) / n_bins
    return (
        pl.when(width > 0)
        .then(((x - low) / width).floor().clip(0, n_bins - 1))
        .otherwise(0)
        .cast(pl.Int64)
        .alias(column)
    )


def auto_compute(
    config: Settings, df: pl.LazyFrame, summary: dict
) -> pl.DataFrame | None:
    numeric = _numeric_columns(summary, min_distinct=1)
    categorical = _categorical_columns(config, summary)
    if len(numeric + categorical) <= 1:
        return None
    columns = sorted(numeric + categorical)
    n_bins = config.correlations["auto"].n_bins
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.eye(len(columns))

    numeric_pairs = [
        (a, b)
        for a, b in itertools.combinations(columns, 2)
        if a in numeric and b in numeric
    ]
    if numeric_pairs:
        spearman = _pairwise(
            _numeric_frame(df, numeric),
            numeric,
            lambda a, b: pl.corr(a, b, method="spearman"),
        )
        for a, b in numeric_pairs:
            value = spearman.get_column(b)[numeric.index(a)]
            matrix[index[a], index[b]] = matrix[index[b], index[a]] = value

    mixed_pairs = [
        (a, b)
        for a, b in itertools.combinations(columns, 2)
        if a in categorical or b in categorical
    ]
    if mixed_pairs:
        binned = df.select(
            [_discretized(df, c, n_bins) for c in numeric]
            + [pl.col(c) for c in categorical]
        )
        for (a, b), table in zip(mixed_pairs, _crosstabs(binned, mixed_pairs)):
            value = cramers_corrected_stat(table) if table.size else np.nan
            matrix[index[a], index[b]] = matrix[index[b], index[a]] = value
    return _matrix_frame(columns, matrix)


CORRELATION_MEASURES: dict[str, Callable] = {
    "auto": auto_compute,
    "pearson": pearson_compute,
    "spearman": spearman_compute,
    "kendall": kendall_compute,
    "cramers": cramers_compute,
}


def warn_correlation(correlation_name: str, error: str) -> None:
    warnings.warn(
        f"""There was an attempt to calculate the {correlation_name} correlation, but this failed.
To hide this warning, disable the calculation
(using `ProfileReport(df, correlations={{"{correlation_name}": {{"calculate": False}}}})`
If this is problematic for your use case, please report this as an issue:
https://github.com/andrewadlof/polars-data-profiling/issues
(include the error message: '{error}')""",
        stacklevel=2,
    )


def calculate_correlation(
    config: Settings, df: pl.LazyFrame, correlation_name: str, summary: dict
) -> pl.DataFrame | None:
    """One correlation matrix, or None when it is empty or fails.

    Args:
        config: report settings.
        df: the typed frame (columns cast to their inferred types).
        correlation_name: auto, pearson, spearman, kendall or cramers.
        summary: the description of every column.
    """
    if correlation_name not in CORRELATION_MEASURES:
        if correlation_name == "phi_k":
            warn_correlation(
                correlation_name,
                "phi_k is not available in polars-data-profiling (the phik package requires pandas)",
            )
            return None
        raise ValueError(f"Unknown correlation: {correlation_name}")
    correlation = None
    try:
        correlation = CORRELATION_MEASURES[correlation_name](config, df, summary)
    except (
        ValueError,
        AssertionError,
        TypeError,
        IndexError,
        pl.exceptions.PolarsError,
    ) as e:
        warn_correlation(correlation_name, str(e))
    return correlation if correlation is not None and correlation.width > 0 else None


def perform_check_correlation(
    correlation_matrix: pl.DataFrame, threshold: float
) -> dict[str, list[str]]:
    """The variables highly correlated with each variable.

    Args:
        correlation_matrix: a square correlation matrix.
        threshold: the absolute correlation above which variables are flagged.
    """
    cols = correlation_matrix.columns
    values = correlation_matrix.to_numpy().astype(float)
    with np.errstate(invalid="ignore"):
        high = np.abs(values) >= threshold
    np.fill_diagonal(high, False)
    return {
        col: [cols[j] for j in np.flatnonzero(high[i])]
        for i, col in enumerate(cols)
        if high[i].any()
    }


def get_active_correlations(config: Settings) -> list[str]:
    return [name for name in config.correlations if config.correlations[name].calculate]
