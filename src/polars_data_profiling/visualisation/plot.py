"""Plot functions used by the report: each returns inline SVG markup.

The functions take the summary values (histogram counts and edges, value-count
dicts, correlation matrices as ``polars.DataFrame``, time series as frames with
``index`` and ``value`` columns) and hand a chart spec to the native renderer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polars_data_profiling.config import Settings
from polars_data_profiling.visualisation import charts
from polars_data_profiling.visualisation.charts import builders


def _is_valid_hist_data(series: Any, bins: Any) -> bool:
    """Whether histogram counts and bin edges can be drawn."""
    if series is None or bins is None:
        return False
    if isinstance(bins, list):
        return any(_is_valid_hist_data(s, b) for s, b in zip(series, bins))
    try:
        counts = np.asarray(series, dtype=float)
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        counts.size > 0
        and np.isfinite(counts).any()
        and edges.size >= 2
        and bool(np.isfinite(edges).all())
    )


def plot_word_cloud(config: Settings, word_counts: dict | list[dict]) -> str:
    """Word cloud drawn as SVG text (needs the ``wordcloud`` extra), or the
    most frequent words as bars without it."""
    try:
        import wordcloud  # noqa: F401
    except ImportError:
        return charts.render(builders.word_frequency(config, word_counts))
    return charts.render(builders.word_cloud(config, word_counts))


def histogram(
    config: Settings, series: Any, bins: Any, date: bool = False
) -> str | None:
    """Histogram from counts and bin edges (lists of them when comparing).

    Returns:
        The chart, or None without usable data.
    """
    if not _is_valid_hist_data(series, bins):
        return None
    return charts.render(builders.histogram(config, series, bins, date=date))


def mini_histogram(
    config: Settings, series: Any, bins: Any, date: bool = False
) -> str | None:
    """A small histogram for the variable overview."""
    if not _is_valid_hist_data(series, bins):
        return None
    return charts.render(builders.histogram(config, series, bins, date=date, mini=True))


def correlation_matrix(config: Settings, data: Any, vmin: int = -1) -> str:
    """A correlation matrix (a square ``polars.DataFrame``) as a heatmap."""
    return charts.render(builders.correlation_matrix(config, data, vmin=vmin))


def scatter_complex(config: Settings, series: Any) -> str:
    """Scatter (or hexbin) plot of complex values."""
    values = np.asarray(series)
    return charts.render(
        builders.scatter(
            config, values.real, values.imag, "Real", "Imaginary", len(values)
        )
    )


def scatter_series(
    config: Settings, series: Any, x_label: str = "Width", y_label: str = "Height"
) -> str:
    """Scatter (or hexbin) plot of pairs."""
    pairs = list(series)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return charts.render(builders.scatter(config, xs, ys, x_label, y_label, len(pairs)))


def scatter_pairwise(
    config: Settings, series1: Any, series2: Any, x_label: str, y_label: str
) -> str:
    """Scatter (or hexbin) plot of two variables (complete pairs)."""
    xs = np.asarray(series1, dtype=float)
    ys = np.asarray(series2, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    return charts.render(
        builders.scatter(
            config, xs[keep], ys[keep], x_label, y_label, len(xs), gridsize=15
        )
    )


def cat_frequency_plot(config: Settings, data: dict | list[dict]) -> str:
    """Category frequencies as a stacked bar or a pie (``plot.cat_freq.type``)."""
    return charts.render(builders.cat_frequency(config, data))


def plot_timeseries_gap_analysis(
    config: Settings, series: Any, gaps: Any, figsize: tuple = (6, 3)
) -> str:
    """Line plot of the series with the detected gaps highlighted."""
    return charts.render(builders.timeseries_gaps(config, series, gaps))


def plot_overview_timeseries(
    config: Settings, variables: Any, figsize: tuple = (6, 4), scale: bool = False
) -> str:
    """Line plot of all time-series variables (optionally scaled to [0, 1])."""
    return charts.render(builders.overview_timeseries(config, variables, scale))


def mini_ts_plot(
    config: Settings, series: Any, figsize: tuple[float, float] = (3, 2.25)
) -> str:
    """Time-series line plot (mini by default; larger figsize = full width)."""
    return charts.render(builders.timeseries(config, series, mini=figsize[0] <= 4))


def plot_acf_pacf(config: Settings, series: Any, figsize: tuple = (15, 5)) -> str:
    """Autocorrelation and partial autocorrelation plots."""
    return charts.render(builders.acf_pacf(config, series))


def get_ts_lag(config: Settings, n_values: int) -> int:
    """Number of lags of the ACF/PACF plots."""
    return int(min(config.vars.timeseries.pacf_acf_lag, n_values // 2 - 1))
