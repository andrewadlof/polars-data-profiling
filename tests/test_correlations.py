import itertools

import numpy as np
import polars as pl
import pytest
from scipy import stats

from polars_data_profiling.model.alerts import AlertType
from polars_data_profiling.model.correlations import (
    cramers_corrected_stat,
    perform_check_correlation,
)

from .conftest import run_describe


@pytest.fixture
def data() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    z = rng.normal(size=n)
    x[::13] = np.nan
    y[::7] = np.nan
    group = np.where(x > 0, "pos", "neg")
    return pl.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "group": group.tolist(),
            "noise": rng.choice(["a", "b", "c"], n).tolist(),
        }
    )


def matrix_value(matrix: pl.DataFrame, a: str, b: str) -> float:
    return matrix.get_column(b)[matrix.columns.index(a)]


def pairwise(data: pl.DataFrame, a: str, b: str):
    pair = data.select(a, b).drop_nans().drop_nulls()
    return pair.get_column(a).to_numpy(), pair.get_column(b).to_numpy()


@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("pearson", lambda a, b: stats.pearsonr(a, b).statistic),
        ("spearman", lambda a, b: stats.spearmanr(a, b).statistic),
        ("kendall", lambda a, b: stats.kendalltau(a, b).statistic),
    ],
)
def test_numeric_correlations_are_pairwise_complete(config, data, method, reference):
    config.correlations[method].calculate = True
    matrix = run_describe(data, config).correlations[method]
    assert matrix.columns == ["x", "y", "z"]
    for a, b in itertools.combinations(["x", "y", "z"], 2):
        assert matrix_value(matrix, a, b) == pytest.approx(
            reference(*pairwise(data, a, b))
        )
    assert matrix_value(matrix, "x", "x") == 1


def test_cramers_v(config, data):
    config.correlations["cramers"].calculate = True
    matrix = run_describe(data, config).correlations["cramers"]
    table = (
        data.select("group", "noise")
        .group_by("group", "noise")
        .len()
        .pivot(on="noise", index="group", values="len")
        .fill_null(0)
        .drop("group")
        .to_numpy()
    )
    assert matrix_value(matrix, "group", "noise") == pytest.approx(
        cramers_corrected_stat(table)
    )


def test_cramers_v_of_identical_variables_is_one():
    table = np.array([[30, 0], [0, 40]])
    assert cramers_corrected_stat(table) == pytest.approx(1.0, abs=0.05)


def test_auto_mixes_spearman_and_cramers(config, data):
    matrix = run_describe(data, config).correlations["auto"]
    assert set(matrix.columns) == {"x", "y", "z", "group", "noise"}
    assert matrix_value(matrix, "x", "y") == pytest.approx(
        stats.spearmanr(*pairwise(data, "x", "y")).statistic
    )
    # group is the sign of x: strongly associated
    assert matrix_value(matrix, "x", "group") > 0.8
    assert abs(matrix_value(matrix, "z", "noise")) < 0.3


def test_high_correlation_alert(config, data):
    d = run_describe(data, config)
    alerts = {(a.alert_type, a.column_name) for a in d.alerts}
    assert (AlertType.HIGH_CORRELATION, "x") in alerts
    high = perform_check_correlation(d.correlations["auto"], 0.9)
    assert "y" in high["x"]


def test_constant_columns_are_not_correlated(config):
    """A constant number column is categorical (one distinct value), so it is
    left out of the numeric correlations, and of auto (n_distinct <= 1)."""
    config.correlations["pearson"].calculate = True
    df = pl.DataFrame(
        {"a": [1.0, 2, 3, 4, 5, 6, 7], "b": [5.0] * 7, "c": [3.0, 1, 4, 1.5, 5, 9, 2]}
    )
    d = run_describe(df, config)
    assert d.variables["b"]["type"] == "Categorical"
    assert d.correlations["pearson"].columns == ["a", "c"]
    assert "b" not in d.correlations["auto"].columns


def test_phi_k_is_reported_as_unavailable(config, data):
    config.update({"correlations": {"phi_k": {"calculate": True}}})
    from polars_data_profiling.model.correlations import calculate_correlation

    with pytest.warns(UserWarning, match="phik package requires pandas"):
        assert calculate_correlation(config, data.lazy(), "phi_k", {}) is None
