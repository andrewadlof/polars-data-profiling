"""Statistics helpers shared by the summary steps.

Everything here works on small, already-aggregated inputs (value counts, bin
counts, distinct characters) produced by lazy Polars queries; nothing touches
the full column.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import chi2

from polars_data_profiling.config import Settings


class ValueCounts(dict):
    """Value -> count, most frequent first (or sorted by value).

    A plain ``dict``, marked so that comparison reports pair two value counts
    instead of merging them key by key.
    """


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def auto_bin_count(n: int, data_range: float, iqr: float | None) -> int:
    """The number of bins of ``numpy.histogram_bin_edges(bins="auto")``.

    numpy's "auto" takes the smaller of the Sturges and Freedman-Diaconis bin
    widths; both depend only on the number of values, their range and their
    interquartile range, which are computed lazily.
    """
    if n <= 1 or data_range <= 0:
        return 1
    sturges = data_range / (math.log2(n) + 1.0)
    fd = 2.0 * iqr * n ** (-1.0 / 3.0) if iqr and iqr > 0 else 0.0
    if fd > 0:
        # numpy limits the number of bins: FD is never narrower than half the
        # square-root rule's width
        fd = max(fd, data_range / math.sqrt(n) / 2.0)
        width = min(fd, sturges)
    else:
        width = sturges
    return max(1, int(math.ceil(data_range / width)))


def histogram_bin_edges(
    config: Settings,
    vmin: float,
    vmax: float,
    n_unique: int,
    n: int,
    iqr: float | None = None,
) -> np.ndarray:
    """Bin edges for a histogram of values in ``[vmin, vmax]``."""
    hist_config = config.plot.histogram
    data_range = vmax - vmin
    if data_range == 0:
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.1
        return np.array([vmin - eps, vmin + eps])
    if hist_config.bins > 0:
        n_bins = max(1, min(hist_config.bins, n_unique, hist_config.max_bins))
    else:
        n_bins = min(auto_bin_count(n, data_range, iqr), hist_config.max_bins)
    return np.linspace(vmin, vmax, n_bins + 1)


def bin_counts(
    bins: np.ndarray, index_counts: Mapping[int, int] | Sequence[tuple[int, int]]
) -> np.ndarray:
    """Dense bin counts from sparse ``bin index -> count`` pairs."""
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    items = index_counts.items() if isinstance(index_counts, Mapping) else index_counts
    for index, count in items:
        if index is not None and 0 <= index < len(counts):
            counts[int(index)] += int(count)
    return counts


def weighted_histogram(
    config: Settings,
    values: Sequence[float],
    weights: Sequence[float],
    name: str = "histogram",
) -> dict:
    """A histogram of a small set of distinct values with their counts.

    Used for string lengths and file sizes, whose distinct values are few.
    """
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    finite = np.isfinite(values_arr)
    values_arr, weights_arr = values_arr[finite], weights_arr[finite]
    if values_arr.size == 0:
        return {name: []}
    vmin, vmax = float(values_arr.min()), float(values_arr.max())
    bins = histogram_bin_edges(
        config,
        vmin,
        vmax,
        n_unique=len(np.unique(values_arr)),
        n=int(weights_arr.sum()),
    )
    counts, edges = np.histogram(
        values_arr,
        bins=bins,
        weights=weights_arr,
        density=config.plot.histogram.density,
    )
    return {name: (counts, edges)}


# ---------------------------------------------------------------------------
# Tests and scores
# ---------------------------------------------------------------------------


def chi_square(observed: Sequence[float] | np.ndarray | None) -> dict:
    """Chi-squared goodness-of-fit test against the uniform distribution.

    Same result as ``scipy.stats.chisquare(observed)``.
    """
    if observed is None:
        return {"statistic": 0, "pvalue": 0}
    obs = np.asarray(observed, dtype=float)
    if obs.size == 0 or obs.sum() == 0:
        return {"statistic": 0, "pvalue": 0}
    expected = obs.mean()
    statistic = float(((obs - expected) ** 2 / expected).sum())
    dof = obs.size - 1
    pvalue = float(chi2.sf(statistic, dof)) if dof > 0 else float("nan")
    return {"statistic": statistic, "pvalue": pvalue}


def imbalance_score(counts: Sequence[float], n_classes: int) -> float:
    """Class imbalance: 1 - entropy / log2(n_classes).

    A uniform distribution scores 0, a single dominant class approaches 1. A
    single class is balanced (0).
    """
    if n_classes <= 1:
        return 0
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0
    p = arr / arr.sum()
    entropy = float(-(p * np.log2(p)).sum())
    return 1 - entropy / math.log2(n_classes)


def imbalance_from_entropy(entropy: float | None, n_classes: int) -> float:
    """Class imbalance from an entropy (base 2) computed lazily."""
    if n_classes <= 1 or entropy is None:
        return 0
    return 1 - entropy / math.log2(n_classes)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Median of values repeated ``weights`` times (lower median on ties,
    averaged when the cumulative weight lands exactly on the half)."""
    order = np.argsort(values)
    vals = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(w)
    half = cumulative[-1] / 2.0
    idx = int(np.searchsorted(cumulative, half))
    if cumulative[idx] == half and idx + 1 < len(vals):
        return float((vals[idx] + vals[idx + 1]) / 2.0)
    return float(vals[idx])


def length_summary(lengths: Sequence[int], counts: Sequence[int]) -> dict:
    """Min, max, mean and median string length from a length histogram.

    Args:
        lengths: distinct lengths.
        counts: number of values with each length.
    """
    if len(lengths) == 0:
        return {
            "max_length": np.nan,
            "mean_length": np.nan,
            "median_length": np.nan,
            "min_length": np.nan,
            "length_histogram": ValueCounts(),
        }
    order = sorted(range(len(lengths)), key=lambda i: -counts[i])
    return {
        "max_length": int(max(lengths)),
        "mean_length": float(np.average(lengths, weights=counts)),
        "median_length": weighted_median(lengths, counts),
        "min_length": int(min(lengths)),
        "length_histogram": ValueCounts(
            (int(lengths[i]), int(counts[i])) for i in order
        ),
    }


def named_aggregate_summary(values: Sequence[float], key: str) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"{k}_{key}": np.nan for k in ("max", "mean", "median", "min")}
    return {
        f"max_{key}": float(np.max(arr)),
        f"mean_{key}": float(np.mean(arr)),
        f"median_{key}": float(np.median(arr)),
        f"min_{key}": float(np.min(arr)),
    }


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------


def _sorted_counts(counter: Mapping[Any, int]) -> ValueCounts:
    return ValueCounts(sorted(counter.items(), key=lambda kv: -kv[1]))


def unicode_summary(character_counts: Mapping[str, int]) -> dict:
    """Unicode categories, blocks and scripts of a column's characters.

    Args:
        character_counts: character -> occurrences, most frequent first.

    Returns:
        The character, category, block and script summaries.
    """
    try:
        from tangled_up_in_unicode import (  # type: ignore[import-not-found]
            block,
            block_abbr,
            category,
            category_long,
            script,
        )
    except ImportError:
        from unicodedata import category as _category

        category = _category  # type: ignore[assignment]

        def unknown(_: str) -> str:
            return "(unknown)"

        block = block_abbr = category_long = script = unknown  # type: ignore[assignment]

    summary: dict[str, Any] = {
        "n_characters_distinct": len(character_counts),
        "n_characters": int(sum(character_counts.values())),
        "character_counts": ValueCounts(character_counts),
    }
    char_to_block = {char: block(char) for char in character_counts}
    char_to_category = {char: category(char) for char in character_counts}
    char_to_script = {char: script(char) for char in character_counts}

    summary["category_alias_values"] = {
        char: category_long(value) for char, value in char_to_category.items()
    }
    summary["block_alias_values"] = {
        char: block_abbr(value) for char, value in char_to_block.items()
    }

    def group(
        mapping: Mapping[str, str],
    ) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        totals: Counter = Counter()
        per_group: dict[str, Counter] = {}
        for char, n_char in character_counts.items():
            name = mapping[char]
            totals[name] += n_char
            per_group.setdefault(name, Counter())[char] += n_char
        return _sorted_counts(totals), {
            k: _sorted_counts(v) for k, v in per_group.items()
        }

    block_counts, block_chars = group(summary["block_alias_values"])
    summary["block_alias_counts"] = block_counts
    summary["n_block_alias"] = len(block_counts)
    summary["block_alias_char_counts"] = block_chars

    script_counts, script_chars = group(char_to_script)
    summary["script_counts"] = script_counts
    summary["n_scripts"] = len(script_counts)
    summary["script_char_counts"] = script_chars

    category_counts, category_chars = group(summary["category_alias_values"])
    summary["category_alias_counts"] = ValueCounts(
        (str(k).replace("_", " "), v) for k, v in category_counts.items()
    )
    summary["n_category"] = len(category_counts)
    summary["category_alias_char_counts"] = category_chars
    return summary
