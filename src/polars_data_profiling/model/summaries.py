"""Summary steps: the statistics computed for each variable type.

Each step is a plan (see :mod:`.engine`): a generator that yields the lazy
aggregations and queries it needs and fills ``summary`` with the results. The
summarizer chains the steps of a type after those of its parent type (every
type starts with :func:`describe_counts`, :func:`describe_generic` and
:func:`describe_supported`).

Value counts are kept as plain ``dict``s, most frequent first, and truncated:
``value_counts_without_nan`` holds the most frequent values and
``value_counts_index_sorted`` the smallest and largest. The totals
(``n_distinct``, ``count``) are exact.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.model import expressions as ex
from polars_data_profiling.model.engine import ColumnRef, Request, Response
from polars_data_profiling.model.summary_algorithms import (
    ValueCounts,
    bin_counts,
    chi_square,
    histogram_bin_edges,
    imbalance_from_entropy,
    length_summary,
    named_aggregate_summary,
    unicode_summary,
    weighted_histogram,
)

Step = Generator[Request, Response, None]

# Distinct words kept for the word cloud and the word frequency table.
MAX_WORDS = 1000
# Distinct values kept per URL / path component.
MAX_COMPONENT_VALUES = 100


def _n_top(config: Settings) -> int:
    """How many of the most frequent values to keep."""
    return (
        max(
            config.n_freq_table_max,
            config.plot.cat_freq.max_unique,
            config.vars.cat.n_obs,
            config.vars.bool.n_obs,
            config.n_obs_unique,
        )
        + 1
    )


def _counts_dict(
    df: pl.DataFrame, value: str = "value", count: str = "count"
) -> ValueCounts:
    return ValueCounts(
        zip(df.get_column(value).to_list(), df.get_column(count).to_list())
    )


def top_counts(lf: pl.LazyFrame, expr: pl.Expr, k: int) -> pl.LazyFrame:
    """The ``k`` most frequent non-null values of ``expr`` as (value, count)."""
    return (
        lf.select(expr.alias("value"))
        .drop_nulls()
        .group_by("value")
        .len(name="count")
        .sort(["count", "value"], descending=[True, False], nulls_last=True)
        .head(k)
    )


# ---------------------------------------------------------------------------
# Every type
# ---------------------------------------------------------------------------


def describe_counts(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Missing values, distinct values and the value counts."""
    hashable = ex.is_hashable_dtype(column.dtype) and column.dtype != pl.Null
    summary["hashable"] = hashable
    request = Request(aggs={"n_missing": column.expr.null_count()})
    if hashable:
        col = column.name
        counts = (
            column.frame.select(column.expr)
            .drop_nulls()
            .group_by(col)
            .len(name="__count")
        )
        orderable = ex.is_orderable_dtype(column.dtype)
        sort_by = ["__count", col] if orderable else ["__count"]
        request.frames["top"] = counts.sort(
            sort_by, descending=[True, False][: len(sort_by)]
        ).head(_n_top(config))
        count = pl.col("__count")
        mean = count.mean()
        request.frames["stats"] = counts.select(
            pl.len().alias("n_distinct"),
            (count == 1).sum().alias("n_unique"),
            count.entropy(base=2).alias("entropy"),
            ((count - mean) ** 2 / mean).sum().alias("chi_statistic"),
        )
        if orderable:
            n_extreme = config.n_extreme_obs
            ordered = counts.sort(col)
            request.frames["first"] = ordered.head(n_extreme)
            request.frames["last"] = ordered.tail(n_extreme)

    response = yield request
    summary["n_missing"] = int(response.values["n_missing"])
    if not hashable:
        summary["ordering"] = False
        return

    stats = response.frames["stats"].row(0, named=True)
    summary["n_distinct"] = int(stats["n_distinct"])
    summary["n_unique"] = int(stats["n_unique"] or 0)
    summary["_entropy"] = stats["entropy"]
    summary["_counts_chi_statistic"] = stats["chi_statistic"]
    summary["value_counts_without_nan"] = _counts_dict(
        response.frames["top"], column.name, "__count"
    )
    if "first" in response.frames:
        first = _counts_dict(response.frames["first"], column.name, "__count")
        last = _counts_dict(response.frames["last"], column.name, "__count")
        summary["value_counts_index_sorted"] = ValueCounts({**first, **last})
        summary["ordering"] = True
    else:
        summary["ordering"] = False


def describe_generic(config: Settings, column: ColumnRef, summary: dict) -> None:
    """Number of values and the share of missing values."""
    n = summary["n"]
    summary["p_missing"] = summary["n_missing"] / n if n > 0 else 0
    summary["count"] = n - summary["n_missing"]
    summary.setdefault("memory_size", None)


def describe_supported(config: Settings, column: ColumnRef, summary: dict) -> None:
    """Shares of distinct and unique values."""
    if not summary.get("hashable"):
        return
    count = summary["count"]
    distinct = summary["n_distinct"]
    unique = summary["n_unique"]
    summary.update(
        {
            "p_distinct": distinct / count if count > 0 else 0,
            "is_unique": unique == count and count > 0,
            "p_unique": unique / count if count > 0 else 0,
        }
    )


# ---------------------------------------------------------------------------
# Histograms of numbers
# ---------------------------------------------------------------------------


def _bin_index(values: pl.Expr, bins: np.ndarray) -> pl.Expr:
    """Index of the histogram bin of each value (the last bin is closed)."""
    n_bins = len(bins) - 1
    vmin, vmax = float(bins[0]), float(bins[-1])
    width = (vmax - vmin) / n_bins
    return (
        ((values - vmin) / width)
        .floor()
        .clip(0, n_bins - 1)
        .cast(pl.Int64)
        .alias("bin")
    )


def _histogram_query(
    lf: pl.LazyFrame, values: pl.Expr, bins: np.ndarray
) -> pl.LazyFrame:
    return (
        lf.select(values.alias("v"))
        .filter(pl.col("v").is_not_null() & pl.col("v").is_finite())
        .select(_bin_index(pl.col("v"), bins))
        .group_by("bin")
        .len(name="count")
    )


def _finite_aggs(values: pl.Expr) -> dict[str, pl.Expr]:
    finite = values.filter(values.is_finite())
    return {
        "f_min": finite.min(),
        "f_max": finite.max(),
        "f_count": finite.count(),
        "f_q25": finite.quantile(0.25, "linear"),
        "f_q75": finite.quantile(0.75, "linear"),
    }


def _histograms(
    config: Settings,
    column: ColumnRef,
    values: pl.Expr,
    stats: dict,
    n_unique: int,
    summary: dict,
    name: str = "histogram",
) -> Step:
    """The display histogram and the chi-squared test of the finite values."""
    if not stats["f_count"]:
        summary[name] = []
        if config.vars.num.chi_squared_threshold > 0.0:
            summary["chi_squared"] = chi_square(None)
        return
    vmin, vmax = float(stats["f_min"]), float(stats["f_max"])
    n = int(stats["f_count"])
    iqr = float(stats["f_q75"] - stats["f_q25"])
    bins = histogram_bin_edges(config, vmin, vmax, n_unique=n_unique, n=n, iqr=iqr)
    request = Request(frames={"hist": _histogram_query(column.frame, values, bins)})
    chi_bins = None
    if config.vars.num.chi_squared_threshold > 0.0:
        # numpy's "auto" bins, as scipy users would compute the test
        chi_config = config.model_copy(deep=True)
        chi_config.plot.histogram.bins = 0
        chi_config.plot.histogram.max_bins = 10**6
        chi_bins = histogram_bin_edges(chi_config, vmin, vmax, n_unique, n, iqr)
        request.frames["chi"] = _histogram_query(column.frame, values, chi_bins)

    response = yield request
    hist = response.frames["hist"]
    counts = bin_counts(bins, hist.iter_rows())
    if config.plot.histogram.density:
        widths = np.diff(bins)
        counts = counts / (counts.sum() * widths) if counts.sum() else counts
    summary[name] = (counts, bins)
    if chi_bins is not None:
        summary["chi_squared"] = chi_square(
            bin_counts(chi_bins, response.frames["chi"].iter_rows())
        )


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def describe_numeric(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Moments, quantiles, extremes, monotonicity and the histogram."""
    if not summary.get("hashable"):
        return
    raw = column.expr.drop_nulls()
    x = raw.cast(pl.Float64)
    quantiles = config.vars.num.quantiles
    aggs: dict[str, pl.Expr] = {
        "mean": x.mean(),
        "std": x.std(),
        "variance": x.var(),
        "min": raw.min(),
        "max": raw.max(),
        "kurtosis": x.kurtosis(bias=False),
        "skewness": x.skew(bias=False),
        "sum": raw.sum(),
        "n_negative": (x < 0).sum(),
        "n_zeros": (x == 0).sum(),
        "n_infinite": x.is_infinite().sum(),
        "mad": (x - x.median()).abs().median(),
        "monotonic_increase": raw.diff().drop_nulls().ge(0).all(),
        "monotonic_decrease": raw.diff().drop_nulls().le(0).all(),
        **_finite_aggs(x),
    }
    for q in quantiles:
        aggs[f"q_{q}"] = x.quantile(q, "linear")

    response = yield Request(aggs=aggs)
    v = response.values
    n = summary["n"]
    count = summary["count"]

    def num(key: str, default: Any = np.nan) -> Any:
        value = v.get(key)
        return default if value is None else value

    stats = {
        "mean": num("mean"),
        "std": num("std", 0.0) if count > 1 else (np.nan if count == 0 else np.nan),
        "variance": num("variance", 0.0) if count > 1 else np.nan,
        "min": num("min"),
        "max": num("max"),
        "kurtosis": num("kurtosis"),
        "skewness": num("skewness"),
        "sum": num("sum", 0),
        "n_negative": int(num("n_negative", 0)),
        "n_zeros": int(num("n_zeros", 0)),
        "n_infinite": int(num("n_infinite", 0)),
        "mad": num("mad"),
    }
    if count == 0:
        stats.update({"std": 0.0, "variance": 0.0, "kurtosis": 0.0, "skewness": 0.0})
    for q in quantiles:
        stats[f"{q:.0%}"] = num(f"q_{q}")
    summary.update(stats)
    summary["p_negative"] = summary["n_negative"] / n if n else 0
    summary["p_zeros"] = summary["n_zeros"] / n if n else 0
    summary["p_infinite"] = summary["n_infinite"] / n if n else 0
    try:
        summary["range"] = summary["max"] - summary["min"]
    except TypeError:
        summary["range"] = np.nan
    if "75%" in summary and "25%" in summary:
        summary["iqr"] = summary["75%"] - summary["25%"]
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else np.nan

    is_unique = summary.get("n_distinct") == count
    increase = bool(v["monotonic_increase"])
    decrease = bool(v["monotonic_decrease"])
    summary["monotonic_increase"] = increase
    summary["monotonic_decrease"] = decrease
    summary["monotonic_increase_strict"] = increase and is_unique
    summary["monotonic_decrease_strict"] = decrease and is_unique
    if summary["monotonic_increase_strict"]:
        summary["monotonic"] = 2
    elif summary["monotonic_decrease_strict"]:
        summary["monotonic"] = -2
    elif increase:
        summary["monotonic"] = 1
    elif decrease:
        summary["monotonic"] = -1
    else:
        summary["monotonic"] = 0

    yield from _histograms(
        config,
        column,
        column.expr.cast(pl.Float64),
        v,
        summary.get("n_distinct", 0),
        summary,
    )


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def _to_python_datetime(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # a date: promote, as pandas Timestamps did
    return datetime(value.year, value.month, value.day)


def describe_date(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Range of the dates and the histogram of their timestamps."""
    if not summary.get("hashable"):
        return
    dates = column.expr.cast(pl.Datetime("us"))
    seconds = dates.dt.epoch("s").cast(pl.Float64)
    response = yield Request(
        aggs={"min": dates.min(), "max": dates.max(), **_finite_aggs(seconds)}
    )
    v = response.values
    if v["min"] is None:
        summary.update({"min": None, "max": None, "range": 0})
    else:
        summary["min"] = _to_python_datetime(v["min"])
        summary["max"] = _to_python_datetime(v["max"])
        summary["range"] = summary["max"] - summary["min"]
    yield from _histograms(
        config, column, seconds, v, summary.get("n_distinct", 0), summary
    )
    summary.update({"invalid_dates": 0, "n_invalid_dates": 0, "p_invalid_dates": 0})


# ---------------------------------------------------------------------------
# Booleans and categories
# ---------------------------------------------------------------------------


def describe_boolean(config: Settings, column: ColumnRef, summary: dict) -> None:
    """Most frequent value and the class imbalance."""
    if not summary.get("hashable"):
        return
    value_counts = summary.get("value_counts_without_nan", {})
    if value_counts:
        top, freq = next(iter(value_counts.items()))
        summary.update({"top": top, "freq": freq})
        summary["imbalance"] = imbalance_from_entropy(
            summary.get("_entropy"), summary["n_distinct"]
        )
    else:
        summary.update({"top": None, "freq": 0, "imbalance": 0})


def _string_requests(
    config: Settings, column: ColumnRef, length: bool, characters: bool, words: bool
) -> Request:
    lf = column.frame
    s = column.expr.cast(pl.String)
    request = Request()
    if length:
        request.frames["lengths"] = (
            lf.select(s.drop_nulls().str.len_chars().alias("length"))
            .group_by("length")
            .len(name="count")
        )
    if characters:
        request.frames["characters"] = (
            lf.select(s.drop_nulls().str.split("").explode().alias("char"))
            .filter(pl.col("char").is_not_null() & (pl.col("char") != ""))
            .group_by("char")
            .len(name="count")
            .sort(["count", "char"], descending=[True, False])
        )
    if words:
        stop_words = [w.lower() for w in config.vars.cat.stop_words]
        word = pl.col("word")
        words_lf = lf.select(
            s.drop_nulls()
            .str.to_lowercase()
            .str.extract_all(r"\S+")
            .explode()
            .str.strip_chars(ex.WORD_STRIP_CHARACTERS)
            .alias("word")
        ).filter(word.is_not_null() & (word != ""))
        if stop_words:
            words_lf = words_lf.filter(~word.is_in(stop_words))
        request.frames["words"] = (
            words_lf.group_by("word")
            .len(name="count")
            .sort(["count", "word"], descending=[True, False])
            .head(MAX_WORDS)
        )
    request.frames["first_rows"] = lf.select(s.drop_nulls().head(5).alias("v"))
    return request


def _string_results(config: Settings, response: Response, summary: dict) -> None:
    frames = response.frames
    if "lengths" in frames:
        lengths = frames["lengths"]
        lens = lengths.get_column("length").to_list()
        counts = lengths.get_column("count").to_list()
        summary.update(length_summary(lens, counts))
        summary.update(
            weighted_histogram(config, lens, counts, name="histogram_length")
        )
    if "characters" in frames:
        summary.update(
            unicode_summary(_counts_dict(frames["characters"], "char", "count"))
        )
    if "words" in frames:
        word_counts = _counts_dict(frames["words"], "word", "count")
        if word_counts:
            summary["word_counts"] = word_counts


def describe_categorical(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Imbalance, goodness of fit, and length, character and word statistics."""
    if not summary.get("hashable"):
        return
    cat = config.vars.cat
    summary["imbalance"] = imbalance_from_entropy(
        summary.get("_entropy"), summary["n_distinct"]
    )
    if config.vars.num.chi_squared_threshold > 0.0:
        statistic = summary.get("_counts_chi_statistic")
        dof = summary["n_distinct"] - 1
        if statistic is None or summary["count"] == 0:
            summary["chi_squared"] = chi_square(None)
        else:
            from scipy.stats import chi2

            summary["chi_squared"] = {
                "statistic": float(statistic),
                "pvalue": float(chi2.sf(statistic, dof)) if dof > 0 else float("nan"),
            }
    response = yield _string_requests(
        config, column, cat.length, cat.characters, cat.words
    )
    if not cat.redact:
        summary["first_rows"] = response.frames["first_rows"].get_column("v")
    _string_results(config, response, summary)


def describe_text(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Length, character and word statistics of free text."""
    if not summary.get("hashable"):
        return
    text = config.vars.text
    response = yield _string_requests(
        config, column, text.length, text.characters, text.words
    )
    summary["first_rows"] = response.frames["first_rows"].get_column("v")
    _string_results(config, response, summary)


# ---------------------------------------------------------------------------
# URLs, paths, files and images
# ---------------------------------------------------------------------------

URL_COMPONENTS = ("scheme", "netloc", "path", "query", "fragment")


def describe_url(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Counts of each URL component (scheme, host, path, query, fragment)."""
    parts = column.expr.str.extract_groups(ex.URL_PATTERN)
    request = Request()
    for index, name in enumerate(URL_COMPONENTS, start=1):
        component = parts.struct.field(str(index)).fill_null("")
        part = pl.when(column.expr.is_not_null()).then(component)
        request.frames[name] = top_counts(column.frame, part, MAX_COMPONENT_VALUES)
        request.aggs[f"n_{name}"] = part.drop_nulls().n_unique()
    response = yield request
    for name in URL_COMPONENTS:
        summary[f"{name}_counts"] = _counts_dict(response.frames[name])
        summary[f"n_{name}_unique"] = response.values[f"n_{name}"]


def _path_components(expr: pl.Expr) -> dict[str, pl.Expr]:
    sep = r"[\\/]" if os.name == "nt" else "/"
    not_sep = r"[^\\/]" if os.name == "nt" else "[^/]"
    name = expr.str.extract(rf"({not_sep}*)$", 1)
    parent = expr.str.extract(rf"^(.*){sep}{not_sep}*$", 1).fill_null("")
    # os.path.dirname keeps the root ("/" or "C:\\")
    parent = (
        pl.when((parent == "") & expr.str.contains(rf"^{sep}"))
        .then(expr.str.slice(0, 1))
        .otherwise(parent)
    )
    suffix = name.str.extract(r"[^.](\.[^.]*)$", 1).fill_null("")
    stem = (
        pl.when(suffix == "")
        .then(expr)
        .otherwise(expr.str.slice(0, expr.str.len_chars() - suffix.str.len_chars()))
    )
    anchor = (
        expr.str.extract(r"^([A-Za-z]:)", 1).fill_null("")
        if os.name == "nt"
        else pl.lit("")
    )
    return {
        "stem": stem,
        "suffix": suffix,
        "name": name,
        "parent": parent,
        "anchor": anchor,
    }


def describe_path(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Counts of each path component, and the prefix shared by all paths."""
    request = Request(
        aggs={"min": column.expr.min(), "max": column.expr.max()},
    )
    for name, expr in _path_components(column.expr).items():
        part = pl.when(column.expr.is_not_null()).then(expr)
        request.frames[name] = top_counts(column.frame, part, MAX_COMPONENT_VALUES)
        request.aggs[f"n_{name}"] = part.drop_nulls().n_unique()
    response = yield request
    # The prefix common to all strings is the one of the smallest and largest.
    low, high = response.values["min"] or "", response.values["max"] or ""
    summary["common_prefix"] = os.path.commonprefix([low, high]) or "No common prefix"
    for name in ("stem", "suffix", "name", "parent", "anchor"):
        summary[f"{name}_counts"] = _counts_dict(response.frames[name])
        summary[f"n_{name}_unique"] = response.values[f"n_{name}"]


def describe_file(config: Settings, column: ColumnRef, summary: dict) -> None:
    """File sizes and timestamps (reads the file system)."""
    paths = column.series().drop_nulls().to_list()
    stats = [os.stat(p) for p in paths]

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    sizes = [s.st_size for s in stats]
    summary["file_size"] = sizes
    summary["file_created_time"] = [fmt(s.st_ctime) for s in stats]
    summary["file_accessed_time"] = [fmt(s.st_atime) for s in stats]
    summary["file_modified_time"] = [fmt(s.st_mtime) for s in stats]
    size_counts: dict[int, int] = {}
    for size in sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    summary.update(
        weighted_histogram(
            config,
            list(size_counts),
            list(size_counts.values()),
            name="histogram_file_size",
        )
    )


def describe_image(config: Settings, column: ColumnRef, summary: dict) -> None:
    """Image dimensions, truncation, EXIF data and duplicate hashes."""
    from polars_data_profiling.model.images import image_summary

    paths = column.series().drop_nulls().to_list()
    summary.update(
        image_summary(paths, exif=config.vars.image.exif, hash=config.vars.image.hash)
    )


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def describe_timeseries(config: Settings, column: ColumnRef, summary: dict) -> Step:
    """Seasonality, stationarity and gaps of a numeric time series."""
    from polars_data_profiling.model.timeseries import timeseries_summary

    index = (
        pl.col(config.vars.timeseries.sortby)
        if config.vars.timeseries.sortby
        else pl.int_range(pl.len(), dtype=pl.Int64)
    )
    response = yield Request(
        frames={
            "series": column.frame.select(
                index.alias("index"), column.expr.cast(pl.Float64).alias("value")
            )
        }
    )
    data = response.frames["series"]
    summary.update(timeseries_summary(config, data))


def named_summary(values: list[float], key: str) -> dict:  # pragma: no cover
    return named_aggregate_summary(values, key)
