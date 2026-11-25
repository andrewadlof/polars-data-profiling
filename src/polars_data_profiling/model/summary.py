"""Infer the type of every column and compute its summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from polars_data_profiling.config import Settings
from polars_data_profiling.model import expressions as ex
from polars_data_profiling.model.engine import ColumnRef, run_plans
from polars_data_profiling.model.probe import InferenceColumn, probe_frame
from polars_data_profiling.model.summarizer import BaseSummarizer
from polars_data_profiling.model.typeset_base import BaseTypeSet
from polars_data_profiling.utils.dataframe import sort_column_names

# Summary entries used between steps only.
_INTERNAL_KEYS = ("_entropy", "_counts_chi_statistic")


@dataclass
class TypedFrame:
    """The profiled frame with every column cast to its inferred type.

    Attributes:
        frame: the typed LazyFrame (strings of numbers parsed, categories as
            strings, ...).
        types: inferred type name per column.
        n_rows: number of rows.
    """

    frame: pl.LazyFrame
    types: dict[str, str]
    n_rows: int


def infer_types(
    config: Settings, df: pl.LazyFrame, typeset: BaseTypeSet
) -> tuple[TypedFrame, dict[str, InferenceColumn]]:
    """Infer the type of each column (one lazy pass) and build the typed frame."""
    schema = df.collect_schema()
    n_rows, columns = probe_frame(config, df, schema)

    types: dict[str, str] = {}
    exprs: list[pl.Expr] = []
    for name, column in columns.items():
        has_cast_type = name in typeset.type_schema
        if has_cast_type and column.facts.get("count"):
            vtype = typeset.type_schema[name]
            final = _cast_to_declared(column, str(vtype))
        elif config.infer_dtypes:
            final, path, _ = typeset.infer(column)
            vtype = path[-1]
        else:
            final, path, _ = typeset.detect(column)
            vtype = path[-1]
        typeset.type_schema[name] = vtype
        types[name] = str(vtype)
        expr = final.expr
        # Categories are profiled as strings
        if str(vtype) in ("Categorical", "Text") and (
            ex.is_categorical_dtype(final.dtype) or final.dtype.is_numeric()
        ):
            expr = expr.cast(pl.String)
        exprs.append(expr.alias(name))
    typed = df.select(exprs) if exprs else df
    if config.engine.share_scans:
        # Queries collected together read the source once (see engine.Engine)
        typed = typed.cache()
    return TypedFrame(frame=typed, types=types, n_rows=n_rows), columns


def _cast_to_declared(column: InferenceColumn, type_name: str) -> InferenceColumn:
    """A column under the representation of a type the user declared."""
    from polars_data_profiling.model.typeset_relations import (
        string_to_datetime,
        string_to_numeric,
        to_category,
    )

    if ex.is_string_dtype(column.dtype):
        if type_name in ("Numeric", "TimeSeries"):
            return string_to_numeric(column, {})
        if type_name == "DateTime":
            return string_to_datetime(column, {})
    if type_name == "Categorical":
        return to_category(column, {})
    return column


def get_series_descriptions(
    config: Settings,
    df: pl.LazyFrame,
    summarizer: BaseSummarizer,
    typeset: BaseTypeSet,
    pbar: tqdm,
) -> tuple[dict[str, dict[str, Any]], TypedFrame]:
    """Describe every column.

    Returns:
        The description of each column (sorted as configured) and the typed
        frame, which correlations and interactions read.
    """
    pbar.set_postfix_str("Infer variable types")
    # Types declared by the user (``type_schema``), before inference adds its own
    declared = set(typeset.type_schema)
    typed, columns = infer_types(config, df, typeset)
    schema = typed.frame.collect_schema()

    plans = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, type_name in typed.types.items():
        source = columns[name]
        summary: dict[str, Any] = {
            "n": typed.n_rows,
            "memory_size": source.source_facts.get("memory_size"),
            # Shown for unsupported columns when inference is not possible
            "cast_type": str(typeset.type_schema[name]) if name in declared else None,
        }
        column = ColumnRef(
            name=name,
            frame=typed.frame,
            dtype=schema[name],
            source_dtype=source.dtype,
        )
        summaries[name] = summary
        plans[name] = summarizer.summarize(config, column, type_name, summary)

    pbar.set_postfix_str("Describe variables")

    def done(_: Any) -> None:
        pbar.update()

    results = run_plans(typed.frame, plans, on_done=done)
    descriptions = {}
    for name in typed.types:
        summary = results.get(name) or summaries[name]
        for key in _INTERNAL_KEYS:
            summary.pop(key, None)
        descriptions[name] = summary
    return sort_column_names(descriptions, config.sort), typed
