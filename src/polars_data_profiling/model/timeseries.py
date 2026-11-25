"""Time-series statistics: stationarity, seasonality, gaps and autocorrelation.

These need the whole series in order, so they run on a materialised column
(numpy arrays). They replace the statsmodels functions the pandas backend used
(``adfuller``, ``acf``, ``pacf``), which would pull in pandas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.stats import norm

from polars_data_profiling.config import Settings

# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller test (constant, no trend)
# ---------------------------------------------------------------------------

# MacKinnon (1994) response-surface coefficients for the p-value of the ADF
# statistic with a constant and one series (statsmodels.tsa.adfvalues, scaled).
_TAU_MAX_C = 2.74
_TAU_MIN_C = -18.83
_TAU_STAR_C = -1.61
_TAU_SMALLP_C = (2.1659, 1.4412, 0.038269)
_TAU_LARGEP_C = (1.7339, 0.93202, -0.12745, -0.010368)


def mackinnon_pvalue(stat: float) -> float:
    """Approximate p-value of an ADF statistic (regression with a constant)."""
    if stat > _TAU_MAX_C:
        return 1.0
    if stat < _TAU_MIN_C:
        return 0.0
    coef = _TAU_SMALLP_C if stat <= _TAU_STAR_C else _TAU_LARGEP_C
    return float(norm.cdf(np.polyval(coef[::-1], stat)))


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, their t-values and the AIC of an OLS fit."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    nobs, k = x.shape
    ssr = float(resid @ resid)
    llf = -nobs / 2.0 * (math.log(2 * math.pi) + math.log(ssr / nobs) + 1.0)
    aic = -2.0 * llf + 2.0 * k
    dof = max(nobs - rank, 1)
    sigma2 = ssr / dof
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = beta / np.sqrt(np.diag(cov))
    return beta, tvalues, aic


def _lagged_design(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Response and regressors of the ADF regression with ``lags`` lags.

    Regressors are the lagged level followed by the lagged differences.
    """
    xdiff = np.diff(x)
    nobs = len(xdiff) - lags
    y = xdiff[lags:]
    columns = [x[lags:-1]]
    for lag in range(1, lags + 1):
        columns.append(xdiff[lags - lag : len(xdiff) - lag])
    return y, np.column_stack(columns)[:nobs]


def adfuller(
    x: np.ndarray, maxlag: int | None = None, autolag: str | None = "AIC"
) -> tuple[float, float, int]:
    """Augmented Dickey-Fuller unit-root test with a constant.

    Matches ``statsmodels.tsa.stattools.adfuller(x, regression="c")`` for the
    statistic and the MacKinnon p-value.

    Returns:
        The test statistic, its p-value and the number of lags used.
    """
    x = np.asarray(x, dtype=float)
    nobs = len(x)
    ntrend = 1
    if maxlag is None:
        maxlag = int(math.ceil(12.0 * (nobs / 100.0) ** 0.25))
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError(
                "sample size is too short to use selected regression component"
            )
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError("maxlag must be less than (nobs/2 - 1 - ntrend)")

    if autolag is not None and autolag.upper() == "AIC" and maxlag > 0:
        y, design = _lagged_design(x, maxlag)
        full = np.column_stack([np.ones(len(y)), design])
        best_aic, best_lag = math.inf, 0
        # one, two, ... regressors after the constant: lagged level + lags
        for n_cols in range(2, full.shape[1] + 1):
            _, _, aic = _ols(y, full[:, :n_cols])
            if aic < best_aic:
                best_aic, best_lag = aic, n_cols - 2
        usedlag = best_lag
    else:
        usedlag = maxlag

    y, design = _lagged_design(x, usedlag)
    regressors = np.column_stack([design, np.ones(len(y))])
    _, tvalues, _ = _ols(y, regressors)
    stat = float(tvalues[0])
    return stat, mackinnon_pvalue(stat), usedlag


def stationarity_test(config: Settings, values: np.ndarray) -> tuple[bool, float]:
    _, p_value, _ = adfuller(
        values,
        maxlag=config.vars.timeseries.maxlag,
        autolag=config.vars.timeseries.autolag,
    )
    return p_value < config.vars.timeseries.significance, p_value


# ---------------------------------------------------------------------------
# Seasonality (FFT peaks, after Kats)
# ---------------------------------------------------------------------------


def get_fft(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive frequencies and their power (dB)."""
    psd = np.abs(np.fft.fft(values)) ** 2
    freq = np.fft.fftfreq(len(psd), 1.0)
    positive = freq > 0
    with np.errstate(divide="ignore"):
        ampl = 10 * np.log10(psd[positive])
    return freq[positive], ampl


def seasonality_test(values: np.ndarray, mad_threshold: float = 6.0) -> dict[str, Any]:
    """Detect seasonality from the outlying peaks of the power spectrum.

    Source: https://github.com/facebookresearch/Kats/blob/main/kats/detectors/seasonality.py
    """
    freq, ampl = get_fft(values)
    positive = ampl > 0
    if not positive.any():
        return {"seasonality_presence": False, "seasonalities": []}
    pos = ampl[positive & np.isfinite(ampl)]
    median = float(np.median(pos))
    above = pos[pos > median]
    mad = float(np.abs(above - above.mean()).mean()) if above.size else 0.0
    threshold = median + mad * mad_threshold

    finite = np.where(np.isfinite(ampl), ampl, -np.inf)
    peak_idx, _ = find_peaks(finite, threshold=0.1)
    peaks = [float(freq[i]) for i in peak_idx if ampl[i] > threshold]

    # drop harmonics of the base frequencies
    keep: list[float] = []
    removed = [False] * len(peaks)
    for i, current in enumerate(peaks):
        if removed[i]:
            continue
        keep.append(current)
        for j in range(i + 1, len(peaks)):
            if removed[j]:
                continue
            fraction = (peaks[j] / current) % 1
            if fraction < 0.01 or fraction > 0.99:
                removed[j] = True
    return {
        "seasonality_presence": len(keep) > 0,
        "seasonalities": [1 / f for f in keep],
    }


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def compute_gap_stats(data: pl.DataFrame, gap_tolerance: int = 2) -> dict:
    """Intervals between observations that are much longer than usual.

    Args:
        data: ``index`` and ``value`` columns, non-null values only.
    """
    index = data.get_column("index")
    is_date = index.dtype.is_temporal()
    if is_date:
        raw = index.cast(pl.Datetime("us")).to_numpy()
        diffs = np.diff(raw).astype("timedelta64[us]").astype(np.int64).astype(float)
    else:
        raw = index.cast(pl.Float64).to_numpy()
        diffs = np.diff(raw)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        gap_mask = np.zeros_like(diffs, dtype=bool)
        gap_sizes = positive
    else:
        min_gap = gap_tolerance * positive.mean()
        gap_mask = diffs > min_gap
        gap_sizes = positive[positive > min_gap]

    values = index.to_list()
    gaps = [[values[i], values[i + 1]] for i in np.flatnonzero(gap_mask)]

    def as_interval(v: float) -> Any:
        if is_date:
            from datetime import timedelta

            return timedelta(microseconds=float(v))
        return float(v)

    has_gaps = gap_sizes.size > 0
    return {
        "min": as_interval(gap_sizes.min()) if has_gaps else 0,
        "max": as_interval(gap_sizes.max()) if has_gaps else 0,
        "mean": as_interval(gap_sizes.mean()) if has_gaps else 0,
        "std": as_interval(gap_sizes.std(ddof=1)) if gap_sizes.size > 1 else 0,
        "series": data,
        "gaps": gaps,
        "n_gaps": len(gaps),
    }


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def _autocovariance(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(x, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: nlags + 1] / n
    return acov


def acf(
    x: np.ndarray, nlags: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation with Bartlett confidence intervals (as statsmodels)."""
    n = len(x)
    acov = _autocovariance(x, nlags)
    values = acov / acov[0] if acov[0] else np.zeros_like(acov)
    var = np.ones(nlags + 1) / n
    var[0] = 0
    var[1] = 1.0 / n
    var[2:] *= 1 + 2 * np.cumsum(values[1:-1] ** 2)
    interval = norm.ppf(1 - alpha / 2.0) * np.sqrt(var)
    return values, np.column_stack([values - interval, values + interval])


def pacf(
    x: np.ndarray, nlags: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Partial autocorrelation by Levinson-Durbin on the biased autocovariance
    (statsmodels' ``method="ywm"``), with normal confidence intervals."""
    n = len(x)
    acov = _autocovariance(x, nlags)
    values = np.zeros(nlags + 1)
    values[0] = 1.0
    if acov[0] == 0:
        conf = np.column_stack([values, values])
        return values, conf
    phi = np.zeros(nlags + 1)
    error = acov[0]
    for k in range(1, nlags + 1):
        reflection = (acov[k] - np.dot(phi[1:k], acov[k - 1 : 0 : -1])) / error
        new_phi = phi.copy()
        new_phi[k] = reflection
        new_phi[1:k] = phi[1:k] - reflection * phi[k - 1 : 0 : -1]
        phi = new_phi
        error *= 1 - reflection**2
        values[k] = reflection
    interval = norm.ppf(1 - alpha / 2.0) * np.sqrt(1.0 / n)
    conf = np.column_stack([values - interval, values + interval])
    conf[0] = values[0]
    return values, conf


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def timeseries_summary(config: Settings, data: pl.DataFrame) -> dict:
    """Seasonality, stationarity and gaps of a series.

    Args:
        data: ``index`` and ``value`` columns, in time order.
    """
    data = data.filter(pl.col("value").is_not_null())
    values = data.get_column("value").to_numpy()
    seasonal = (
        seasonality_test(values)["seasonality_presence"] if values.size else False
    )
    try:
        stationary, p_value = stationarity_test(config, values)
    except (ValueError, np.linalg.LinAlgError):
        stationary, p_value = False, float("nan")
    return {
        "seasonal": seasonal,
        "stationary": stationary and not seasonal,
        "addfuller": p_value,
        "series": data,
        "gap_stats": compute_gap_stats(data),
    }
