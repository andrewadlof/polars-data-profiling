"""Compute the full description of a frame."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from polars_data_profiling.config import Settings
from polars_data_profiling.model import BaseAnalysis, BaseDescription
from polars_data_profiling.model.alerts import get_alerts
from polars_data_profiling.model.correlations import (
    calculate_correlation,
    get_active_correlations,
)
from polars_data_profiling.model.dataframe import preprocess
from polars_data_profiling.model.description import TimeIndexAnalysis
from polars_data_profiling.model.duplicates import get_duplicates
from polars_data_profiling.model.missing import get_missing_active, get_missing_diagram
from polars_data_profiling.model.pairwise import (
    get_scatter_data,
    get_scatter_plot,
    get_scatter_tasks,
)
from polars_data_profiling.model.sample import get_custom_sample, get_sample
from polars_data_profiling.model.summarizer import BaseSummarizer
from polars_data_profiling.model.summary import get_series_descriptions
from polars_data_profiling.model.table import get_table_stats
from polars_data_profiling.model.timeseries_index import get_time_index_description
from polars_data_profiling.model.typeset_base import BaseTypeSet
from polars_data_profiling.utils.collect import use_engine
from polars_data_profiling.utils.dataframe import to_lazyframe
from polars_data_profiling.utils.progress_bar import progress
from polars_data_profiling.version import __version__


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def describe(
    config: Settings,
    df: pl.LazyFrame | pl.DataFrame,
    summarizer: BaseSummarizer,
    typeset: BaseTypeSet,
    sample: dict | None = None,
) -> BaseDescription:
    """Calculate the statistics of every column and of the table.

    Args:
        config: report settings.
        df: the frame to profile. A LazyFrame is never collected whole: every
            statistic is computed by a lazy query.
        summarizer: summarizer object.
        typeset: the typeset used to infer variable types.
        sample: optional, dict with a custom sample.

    Returns:
        The description: table statistics, variables, correlations, missing
        values diagrams, alerts, samples, duplicates and package details.
    """
    if not isinstance(config, Settings):
        raise TypeError(f"`config` must be of type `Settings`, got {type(config)}")
    with use_engine(config.engine.collect_engine):
        return _describe(config, df, summarizer, typeset, sample)


def _describe(
    config: Settings,
    df: pl.LazyFrame | pl.DataFrame,
    summarizer: BaseSummarizer,
    typeset: BaseTypeSet,
    sample: dict | None,
) -> BaseDescription:
    lf = preprocess(config, to_lazyframe(df))
    if config.engine.share_scans:
        # Queries collected together read the source once
        lf = lf.cache()

    number_of_tasks = 5
    with tqdm(
        total=number_of_tasks,
        desc="Summarize dataset",
        disable=not config.progress_bar,
        position=0,
    ) as pbar:
        date_start = _now()

        # Variables: type inference (one pass) and summaries (batched passes)
        n_columns = len(lf.collect_schema())
        pbar.total += n_columns
        series_description, typed = get_series_descriptions(
            config, lf, summarizer, typeset, pbar
        )
        variables = typed.types
        n_rows = typed.n_rows

        supported_columns = [c for c, t in variables.items() if t != "Unsupported"]
        interval_columns = [
            c for c, t in variables.items() if t in {"Numeric", "TimeSeries"}
        ]
        pbar.update()

        # Table statistics
        table_stats = progress(get_table_stats, pbar, "Get dataframe statistics")(
            config, n_rows, series_description
        )

        # Correlations, on the typed frame (numbers parsed from strings, ...)
        correlations: dict[str, Any] = {}
        if n_rows != 0:
            correlation_names = get_active_correlations(config)
            pbar.total += len(correlation_names)
            for name in correlation_names:
                matrix = progress(
                    calculate_correlation, pbar, f"Calculate {name} correlation"
                )(config, typed.frame, name, series_description)
                if matrix is not None:
                    correlations[name] = matrix

        # Scatter matrix
        pbar.set_postfix_str("Get scatter matrix")
        scatter_tasks = get_scatter_tasks(config, interval_columns)
        pbar.total += len(scatter_tasks)
        scatter_data = get_scatter_data(
            typed.frame, scatter_tasks, interval_columns, n_rows
        )
        scatter_matrix: dict[Any, dict[Any, Any]] = {
            x: {y: None} for x, y in scatter_tasks
        }
        for x, y in scatter_tasks:
            scatter_matrix[x][y] = progress(
                get_scatter_plot, pbar, f"scatter {x}, {y}"
            )(config, scatter_data.get((x, y)), x, y)

        # Missing values diagrams
        missing_map = get_missing_active(config, table_stats)
        pbar.total += len(missing_map)
        missing = {
            name: progress(get_missing_diagram, pbar, f"Missing diagram {name}")(
                config, lf, n_rows, series_description, settings
            )
            for name, settings in missing_map.items()
        }
        missing = {name: value for name, value in missing.items() if value is not None}

        # Sample
        pbar.set_postfix_str("Take sample")
        samples = (
            get_sample(config, lf, n_rows)
            if sample is None
            else get_custom_sample(sample)
        )
        pbar.update()

        # Duplicates
        metrics, duplicates = progress(get_duplicates, pbar, "Detecting duplicates")(
            config, lf, supported_columns, n_rows
        )
        table_stats.update(metrics)

        alerts = progress(get_alerts, pbar, "Get alerts")(
            config, table_stats, series_description, correlations
        )

        tsindex_description = None
        if config.vars.timeseries.active:
            tsindex_description = get_time_index_description(config, lf, table_stats)

        pbar.set_postfix_str("Get reproduction details")
        package = {
            "polars_data_profiling_version": __version__,
            "polars_data_profiling_config": config.model_dump_json(),
        }
        pbar.update()
        pbar.set_postfix_str("Completed")
        date_end = _now()

    analysis = BaseAnalysis(config.title, date_start, date_end)
    time_index_analysis = (
        TimeIndexAnalysis(**tsindex_description) if tsindex_description else None
    )
    return BaseDescription(
        analysis=analysis,
        time_index_analysis=time_index_analysis,
        table=table_stats,
        variables=series_description,
        scatter=scatter_matrix,
        correlations=correlations,
        missing=missing,
        alerts=alerts,
        package=package,
        sample=samples,
        duplicates=duplicates,
    )
