"""Predicates and transformers of the inference relations.

Predicates decide from the facts that :func:`~.probe.probe_frame` computed for
every column in one pass, so inference never scans the data itself.
Transformers return the column under a new representation: an expression
(parsing strings as numbers, dates or booleans) and the facts that hold for it.
"""

from __future__ import annotations

import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.model import expressions as ex
from polars_data_profiling.model.probe import InferenceColumn
from polars_data_profiling.model.type_rules import (
    is_categorical_cardinality,
    is_few_numbers,
)
from polars_data_profiling.model.typeset_base import has_values


def _source(column: InferenceColumn, key: str, default=None):  # noqa: ANN001, ANN202
    return column.source_facts.get(key, default)


@has_values
def string_is_bool(column: InferenceColumn, state: dict, k: dict[str, bool]) -> bool:
    """Every non-null value is a key of the boolean mapping (any case)."""
    return ex.is_string_dtype(column.dtype) and bool(_source(column, "is_bool"))


def string_to_bool(
    column: InferenceColumn, state: dict, k: dict[str, bool]
) -> InferenceColumn:
    mapping = {key.lower(): value for key, value in k.items()}
    return column.converted(
        ex.parse_bool(column.expr, mapping),
        pl.Boolean,
        {**column.facts, "n_unique": min(column.facts.get("n_unique") or 0, 2)},
    )


def numeric_is_category(column: InferenceColumn, state: dict, k: Settings) -> bool:
    """A number column with few distinct values is categorical."""
    n_unique = column.facts.get("n_unique")
    return n_unique is not None and is_few_numbers(n_unique, k)


def to_category(column: InferenceColumn, state: dict) -> InferenceColumn:
    """Categories are profiled as strings."""
    if ex.is_string_dtype(column.dtype):
        return column
    return column.converted(column.expr.cast(pl.String), pl.String)


@has_values
def string_is_category(column: InferenceColumn, state: dict, k: Settings) -> bool:
    """A string column is categorical when

    - it has between one and ``vars.cat.cardinality_threshold`` distinct values,
    - the share of distinct values is below ``vars.cat.percentage_cat_threshold``,
    - and it is not a boolean column.
    """
    n_unique = column.facts.get("n_unique")
    if n_unique is None:
        return False
    return is_categorical_cardinality(
        n_unique, column.facts["count"], k
    ) and not string_is_bool(column, state, k.vars.bool.mappings)


@has_values
def string_is_datetime(column: InferenceColumn, state: dict) -> bool:
    """Every non-empty value parses as a date and at least one does.

    Empty strings are allowed: they become missing values of the date column.
    Among the values that parse, two kinds are not dates:

    - values without a digit (bare month and weekday names are categories);
    - columns of plain numbers: "2024" or "20240101" are read as numbers, so a
      year column gets the same type whatever its number of distinct values.
    """
    if not ex.is_string_dtype(column.dtype):
        return False
    return bool(
        _source(column, "dt_sample_ok")
        and _source(column, "dt_ok")
        and (_source(column, "dt_count") or 0) > 0
        and _source(column, "dt_digits")
        and not _source(column, "dt_plain_numbers")
    )


def string_to_datetime(column: InferenceColumn, state: dict) -> InferenceColumn:
    return column.converted(
        ex.parse_datetime(column.expr),
        pl.Datetime("us"),
        {
            **column.facts,
            "count": _source(column, "dt_count", 0),
            "n_unique": _source(column, "dt_n_unique"),
        },
    )


@has_values
def string_is_numeric(column: InferenceColumn, state: dict, k: Settings) -> bool:
    """Every value converts to a number, at least one is not missing, and the
    column has more distinct numbers than ``vars.num.low_categorical_threshold``
    (fewer makes it categorical)."""
    if not ex.is_string_dtype(column.dtype):
        return False
    if string_is_bool(column, state, k.vars.bool.mappings):
        return False
    if not _source(column, "num_ok") or not _source(column, "num_count"):
        return False
    n_unique = _source(column, "num_n_unique")
    return not (n_unique is not None and is_few_numbers(n_unique, k))


def string_to_numeric(column: InferenceColumn, state: dict) -> InferenceColumn:
    return column.converted(
        ex.parse_numeric(column.expr),
        pl.Float64,
        {
            **column.facts,
            "count": _source(column, "num_count", 0),
            "n_unique": _source(column, "num_n_unique"),
            "autocorr": _source(column, "num_autocorr", []),
        },
    )
