"""Summarizers: which summary steps run for each variable type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.model import BaseDescription
from polars_data_profiling.model.engine import ColumnRef, Plan
from polars_data_profiling.model.handler import Handler
from polars_data_profiling.model.summaries import (
    describe_boolean,
    describe_categorical,
    describe_counts,
    describe_date,
    describe_file,
    describe_generic,
    describe_image,
    describe_numeric,
    describe_path,
    describe_supported,
    describe_text,
    describe_timeseries,
    describe_url,
)
from polars_data_profiling.model.typeset_base import BaseTypeSet, ProfilingBaseType


class BaseSummarizer(Handler):
    """A base summarizer.

    Can be used to define custom summarizations: map type names to lists of
    summary steps (see :func:`polars_data_profiling.model.handler.compose`).
    """

    def summarize(
        self,
        config: Settings,
        column: ColumnRef,
        dtype: str | type[ProfilingBaseType],
        summary: dict | None = None,
    ) -> Plan:
        """The plan that summarises ``column`` as a ``dtype`` variable."""
        summary = {} if summary is None else summary
        summary["type"] = str(dtype)
        return self.handle(str(dtype), config, column, summary)


class ProfilingSummarizer(BaseSummarizer):
    """The default summarizer."""

    def __init__(self, typeset: BaseTypeSet, *args: Any, **kwargs: Any):
        self._summary_map = self._create_summary_map()
        super().__init__(self._summary_map, typeset)

    @property
    def summary_map(self) -> dict[str, list[Callable]]:
        """Allows users to modify the summary map after initialization."""
        return self._summary_map

    def _create_summary_map(self) -> dict[str, list[Callable]]:
        return {
            "Unsupported": [describe_counts, describe_generic, describe_supported],
            "Numeric": [describe_numeric],
            "DateTime": [describe_date],
            "Text": [describe_text],
            "Categorical": [describe_categorical],
            "Boolean": [describe_boolean],
            "URL": [describe_url],
            "Path": [describe_path],
            "File": [describe_file],
            "Image": [describe_image],
            "TimeSeries": [describe_timeseries],
        }


def format_summary(summary: BaseDescription | dict) -> dict:
    """Prepare summary for export to json file.

    Args:
        summary: summary to export

    Returns:
        dict: summary as dict
    """

    def fmt(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: fmt(va) for k, va in v.items()}
        else:
            if isinstance(v, pl.Series):
                return v.to_list()
            elif isinstance(v, pl.DataFrame):
                return v.to_dicts()
            elif (
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, np.ndarray) for x in v)
            ):
                return {"counts": v[0].tolist(), "bin_edges": v[1].tolist()}
            else:
                return v

    if isinstance(summary, BaseDescription):
        summary = asdict(summary)

    summary = {k: fmt(v) for k, v in summary.items()}
    return summary


def _redact_column(column: dict[str, Any]) -> dict[str, Any]:
    def redact_key(data: dict[str, Any]) -> dict[str, Any]:
        return {f"REDACTED_{i}": v for i, (_, v) in enumerate(data.items())}

    def redact_value(data: Any) -> Any:
        if isinstance(data, list):
            return [f"REDACTED_{i}" for i in range(len(data))]
        return {k: f"REDACTED_{i}" for i, (k, _) in enumerate(data.items())}

    keys_to_redact = [
        "block_alias_char_counts",
        "block_alias_values",
        "category_alias_char_counts",
        "category_alias_values",
        "character_counts",
        "script_char_counts",
        "value_counts_index_sorted",
        "value_counts_without_nan",
        "word_counts",
    ]

    values_to_redact = ["first_rows"]

    for field in keys_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_key(v) for k, v in column[field].items()}
        else:
            column[field] = redact_key(column[field])

    for field in values_to_redact:
        if field not in column:
            continue
        values = column[field]
        is_dict = isinstance(values, dict) and any(
            isinstance(v, dict) for v in values.values()
        )
        if is_dict:
            column[field] = {k: redact_value(v) for k, v in column[field].items()}
        else:
            column[field] = redact_value(column[field])

    return column


def redact_summary(summary: dict, config: Settings) -> dict:
    """Redact summary to export to json file.

    Args:
        summary (dict): summary to redact

    Returns:
        dict: redacted summary
    """
    for _, col in summary["variables"].items():
        if (config.vars.cat.redact and col["type"] == "Categorical") or (
            config.vars.text.redact and col["type"] == "Text"
        ):
            col = _redact_column(col)
    return summary
