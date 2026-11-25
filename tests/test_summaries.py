"""Summary statistics against numpy/scipy reference implementations."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

from polars_data_profiling.model.summary_algorithms import (
    auto_bin_count,
    chi_square,
    imbalance_score,
    weighted_median,
)

from .conftest import run_describe


@pytest.fixture
def numbers():
    rng = np.random.default_rng(1)
    values = rng.gamma(2.0, 3.0, 1000) - 2
    values[::17] = np.nan
    values[5] = 0.0
    values[6] = np.inf
    return values


def test_numeric_statistics(config, numbers):
    d = run_describe(pl.DataFrame({"x": numbers}), config)
    s = d.variables["x"]
    present = numbers[~np.isnan(numbers)]
    finite = present[np.isfinite(present)]

    assert s["type"] == "Numeric"
    assert s["n"] == len(numbers)
    assert s["n_missing"] == np.isnan(numbers).sum()
    assert s["count"] == len(present)
    assert s["n_infinite"] == 1
    assert s["n_zeros"] == 1
    assert s["n_negative"] == (present < 0).sum()
    assert s["n_distinct"] == len(np.unique(present))
    assert s["min"] == present.min()
    assert s["max"] == np.inf

    # moments over the finite values (inf makes them inf/nan in both)
    d2 = run_describe(pl.DataFrame({"x": finite}), config)
    s2 = d2.variables["x"]
    assert s2["mean"] == pytest.approx(finite.mean())
    assert s2["std"] == pytest.approx(finite.std(ddof=1))
    assert s2["variance"] == pytest.approx(finite.var(ddof=1))
    assert s2["sum"] == pytest.approx(finite.sum())
    assert s2["skewness"] == pytest.approx(stats.skew(finite, bias=False))
    assert s2["kurtosis"] == pytest.approx(stats.kurtosis(finite, bias=False))
    assert s2["mad"] == pytest.approx(np.median(np.abs(finite - np.median(finite))))
    for q in config.vars.num.quantiles:
        assert s2[f"{q:.0%}"] == pytest.approx(np.quantile(finite, q))
    assert s2["iqr"] == pytest.approx(
        np.quantile(finite, 0.75) - np.quantile(finite, 0.25)
    )
    assert s2["cv"] == pytest.approx(finite.std(ddof=1) / finite.mean())

    counts, edges = s2["histogram"]
    ref_counts, ref_edges = np.histogram(
        finite, bins=np.linspace(finite.min(), finite.max(), 51)
    )
    np.testing.assert_allclose(edges, ref_edges)
    np.testing.assert_array_equal(counts, ref_counts)


def test_chi_squared_uses_numpy_auto_bins(config):
    values = np.random.default_rng(2).normal(size=777)
    s = run_describe(pl.DataFrame({"x": values}), config).variables["x"]
    reference = stats.chisquare(np.histogram(values, bins="auto")[0])
    assert s["chi_squared"]["statistic"] == pytest.approx(reference.statistic)
    assert s["chi_squared"]["pvalue"] == pytest.approx(reference.pvalue)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (list(range(10)), 2),
        ([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1),
        (list(range(10, 0, -1)), -2),
        ([9, 9, 8, 7, 6, 5, 4, 3, 2, 1], -1),
        ([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], 0),
        ([0, None, 1, 2, 3, 4, 5, 6, 7, 8], 2),
    ],
)
def test_monotonicity(config, values, expected):
    s = run_describe(pl.DataFrame({"x": values}), config).variables["x"]
    assert s["monotonic"] == expected


def test_value_counts_are_truncated_with_exact_totals(config):
    values = [i % 40 for i in range(4000)] + list(range(1000, 1100))
    s = run_describe(pl.DataFrame({"x": values}), config).variables["x"]
    assert s["n_distinct"] == 140
    assert s["n_unique"] == 100
    assert s["p_unique"] == pytest.approx(100 / 4100)
    counts = s["value_counts_without_nan"]
    assert len(counts) < 140
    assert list(counts.values())[0] == 100
    # smallest and largest values, with their counts
    sorted_counts = s["value_counts_index_sorted"]
    assert list(sorted_counts)[:3] == [0, 1, 2]
    assert list(sorted_counts)[-1] == 1099


def test_categorical_statistics(config):
    values = ["apple"] * 50 + ["banana"] * 30 + ["kiwi"] * 20 + [None] * 10
    s = run_describe(pl.DataFrame({"c": values}), config).variables["c"]
    assert s["type"] == "Categorical"
    assert s["value_counts_without_nan"] == {"apple": 50, "banana": 30, "kiwi": 20}
    assert s["n_missing"] == 10
    assert s["imbalance"] == pytest.approx(imbalance_score([50, 30, 20], 3))
    assert s["imbalance"] == pytest.approx(
        1 - stats.entropy([50, 30, 20], base=2) / math.log2(3)
    )
    assert s["chi_squared"]["statistic"] == pytest.approx(
        stats.chisquare([50, 30, 20]).statistic
    )
    assert s["max_length"] == 6 and s["min_length"] == 4
    assert s["mean_length"] == pytest.approx((50 * 5 + 30 * 6 + 20 * 4) / 100)
    assert s["median_length"] == 5
    assert s["n_characters_distinct"] == len(set("applebananakiwi"))
    assert s["character_counts"]["a"] == 50 + 30 * 3
    assert list(s["first_rows"]) == ["apple"] * 5


def test_text_word_counts_and_stop_words(config):
    df = pl.DataFrame({"t": [f"The cat, the dog! item{i}" for i in range(60)]})
    s = run_describe(df, config).variables["t"]
    assert s["type"] == "Text"
    assert s["word_counts"]["the"] == 120
    assert s["word_counts"]["cat"] == 60
    assert "cat," not in s["word_counts"]
    config.vars.cat.stop_words = ["the"]
    s = run_describe(df, config).variables["t"]
    assert "the" not in s["word_counts"]


def test_boolean_statistics(config):
    s = run_describe(pl.DataFrame({"b": [True] * 80 + [False] * 20}), config).variables[
        "b"
    ]
    assert s["type"] == "Boolean"
    assert (s["top"], s["freq"]) == (True, 80)
    assert s["imbalance"] == pytest.approx(imbalance_score([80, 20], 2))


def test_date_statistics(config):
    dates = [dt.datetime(2020, 1, 1) + dt.timedelta(days=3 * i) for i in range(100)]
    s = run_describe(pl.DataFrame({"d": dates}), config).variables["d"]
    assert s["type"] == "DateTime"
    assert s["min"] == dates[0] and s["max"] == dates[-1]
    assert s["range"] == dates[-1] - dates[0]
    counts, edges = s["histogram"]
    assert counts.sum() == 100
    assert edges[0] == dates[0].replace(tzinfo=dt.UTC).timestamp()


def test_string_dates_are_parsed(config):
    values = ["2021-03-04", "03/05/2021", "2021-03-06T10:00:00Z", ""] * 10
    s = run_describe(pl.DataFrame({"d": values}), config).variables["d"]
    assert s["type"] == "DateTime"
    assert s["n_missing"] == 10
    assert s["min"] == dt.datetime(2021, 3, 4)


def test_url_components(config):
    config.vars.url.active = True
    values = ["https://a.com/x?q=1#top", "http://b.org/y", "https://a.com/z?q=2"] * 5
    s = run_describe(pl.DataFrame({"u": values}), config).variables["u"]
    assert s["type"] == "URL"
    assert s["scheme_counts"] == {"https": 10, "http": 5}
    assert s["netloc_counts"] == {"a.com": 10, "b.org": 5}
    assert s["fragment_counts"]["top"] == 5


def test_path_components(config, tmp_path):
    config.vars.path.active = True
    base = tmp_path.as_posix() if tmp_path.anchor == "/" else str(tmp_path)
    sep = "/" if tmp_path.anchor == "/" else "\\"
    values = [f"{base}{sep}data{sep}file{i % 3}.csv" for i in range(30)]
    s = run_describe(pl.DataFrame({"p": values}), config).variables["p"]
    assert s["type"] == "Path"
    assert s["suffix_counts"] == {".csv": 30}
    assert s["n_name_unique"] == 3
    assert s["common_prefix"].endswith(f"data{sep}file")


def test_unsupported_columns_are_counted_not_described(config):
    s = run_describe(pl.DataFrame({"l": [[1], None, [2, 3]]}), config).variables["l"]
    assert s["type"] == "Unsupported"
    assert s["n_missing"] == 1
    assert s["hashable"] is False
    assert s["memory_size"] > 0


def test_table_statistics(config, mixed_df):
    d = run_describe(mixed_df, config)
    assert d.table["n"] == mixed_df.height
    assert d.table["n_var"] == mixed_df.width
    missing = sum(v["n_missing"] for v in d.variables.values())
    assert d.table["n_cells_missing"] == missing
    assert d.table["types"]["Numeric"] >= 3


def test_helpers():
    for size in (10, 1000, 50_000):
        data = np.random.default_rng(size).standard_t(3, size)
        iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
        expected = len(np.histogram_bin_edges(data, bins="auto")) - 1
        assert auto_bin_count(size, data.max() - data.min(), iqr) == expected
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert weighted_median([1, 2], [1, 1]) == 1.5
    assert chi_square([]) == {"statistic": 0, "pvalue": 0}
    assert imbalance_score([10], 1) == 0
