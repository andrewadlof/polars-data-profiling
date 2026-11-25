import datetime as dt

import numpy as np
import polars as pl
import pytest

from polars_data_profiling import ProfileReport
from polars_data_profiling.model.timeseries import (
    acf,
    adfuller,
    compute_gap_stats,
    mackinnon_pvalue,
    pacf,
    seasonality_test,
)


def reference_series() -> dict[str, np.ndarray]:
    """The series whose statsmodels results are hard-coded below."""
    rng = np.random.default_rng(0)
    series = {
        "noise": rng.normal(size=300),
        "walk": np.cumsum(rng.normal(size=300)),
        "ar1": np.zeros(500),
        "seasonal": np.sin(np.arange(400) * 2 * np.pi / 12)
        + rng.normal(scale=0.3, size=400),
        "short": rng.normal(size=25),
    }
    for i in range(1, 500):
        series["ar1"][i] = 0.7 * series["ar1"][i - 1] + rng.normal()
    return series


# statsmodels.tsa.stattools.adfuller(x, autolag="AIC", regression="c")
STATSMODELS_ADF = {
    "noise": (-16.712316, 0.0, 0),
    "walk": (-2.362907, 0.152480, 0),
    "ar1": (-9.188047, 0.0, 0),
    "seasonal": (-4.504704, 0.000193, 12),
    "short": (-4.075854, 0.001061, 3),
}


@pytest.mark.parametrize("name", list(STATSMODELS_ADF))
def test_adfuller_matches_statsmodels(name):
    stat, pvalue, lag = adfuller(reference_series()[name])
    ref_stat, ref_p, ref_lag = STATSMODELS_ADF[name]
    assert stat == pytest.approx(ref_stat, abs=1e-6)
    assert pvalue == pytest.approx(ref_p, abs=1e-6)
    assert lag == ref_lag


def test_mackinnon_bounds():
    assert mackinnon_pvalue(5.0) == 1.0
    assert mackinnon_pvalue(-30.0) == 0.0
    assert 0 < mackinnon_pvalue(-2.0) < 1


def test_acf_is_normalised_autocovariance():
    x = reference_series()["ar1"]
    values, conf = acf(x, nlags=5)
    centred = x - x.mean()
    expected = [
        np.dot(centred[: len(x) - k], centred[k:]) / np.dot(centred, centred)
        for k in range(6)
    ]
    np.testing.assert_allclose(values, expected)
    assert values[1] == pytest.approx(0.7, abs=0.1)
    # Bartlett bands widen with the lag
    widths = conf[:, 1] - conf[:, 0]
    assert widths[0] == 0 and np.all(np.diff(widths[1:]) >= 0)


def test_pacf_of_ar1_cuts_off_after_lag_one():
    values, _ = pacf(reference_series()["ar1"], nlags=5)
    assert values[1] == pytest.approx(0.7, abs=0.1)
    assert np.all(np.abs(values[2:]) < 0.15)


def test_seasonality():
    series = reference_series()
    assert seasonality_test(series["seasonal"])["seasonality_presence"]
    assert not seasonality_test(series["noise"])["seasonality_presence"]
    assert not seasonality_test(series["ar1"])["seasonality_presence"]


def test_gap_stats_with_datetime_index():
    times = [
        dt.datetime(2021, 1, 1) + dt.timedelta(hours=h)
        for h in range(50)
        if h not in (20, 21, 22)
    ]
    data = pl.DataFrame({"index": times, "value": np.arange(len(times), dtype=float)})
    gaps = compute_gap_stats(data)
    assert gaps["n_gaps"] == 1
    assert gaps["gaps"][0] == [dt.datetime(2021, 1, 1, 19), dt.datetime(2021, 1, 1, 23)]
    assert gaps["max"] == dt.timedelta(hours=4)


def test_timeseries_report():
    # three missing hours every 50: a 4-hour step, more than twice the mean step
    times = [
        dt.datetime(2021, 1, 1) + dt.timedelta(hours=h)
        for h in range(400)
        if h % 50 not in (7, 8, 9)
    ]
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"t": times}).with_columns(
        seasonal=pl.Series(
            np.sin(np.arange(len(times)) * 2 * np.pi / 24)
            + rng.normal(scale=0.1, size=len(times))
        ),
        trend=pl.int_range(pl.len()).cast(pl.Float64),
    )
    report = ProfileReport(
        df.sample(fraction=1.0, shuffle=True, seed=1),
        tsmode=True,
        sortby="t",
        progress_bar=False,
    )
    d = report.get_description()
    assert d.variables["trend"]["type"] == "TimeSeries"
    assert d.variables["seasonal"]["seasonal"]
    assert not d.variables["trend"]["stationary"]
    assert d.variables["trend"]["gap_stats"]["n_gaps"] == 8  # hours 7, 57, ..., 357
    assert d.time_index_analysis.start == times[0]
    assert (
        dt.timedelta(hours=1) < d.time_index_analysis.period < dt.timedelta(hours=1.1)
    )
    html = report.to_html()
    assert "Autocorrelation" in html and "Gap analysis" in html
