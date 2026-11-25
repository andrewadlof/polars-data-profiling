"""Facts about every column, computed in one lazy pass, for type inference.

Type inference asks questions such as "do all non-null values of this string
column parse as numbers?" or "how many distinct values does it have?". Rather
than evaluating each question on a materialised series (one pass over the data
per question and per column), :func:`probe_frame` builds the expressions that
answer every question the typeset may ask, for every column, and evaluates them
in a single ``LazyFrame.select``. The typeset then walks its relations on
:class:`InferenceColumn` objects that carry these facts, and transformers
return new *expressions* (e.g. the parse of a string column as numbers) rather
than converted data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.model import expressions as ex
from polars_data_profiling.utils.collect import collect


@dataclass(frozen=True)
class InferenceColumn:
    """A column as the typeset sees it during inference.

    Attributes:
        name: the column name.
        expr: expression producing the column's current representation from
            the preprocessed frame (``pl.col(name)`` before any conversion).
        dtype: the dtype ``expr`` evaluates to.
        facts: facts about the current representation: ``count`` (non-null
            values), ``n_unique`` (distinct non-null values) and the
            dtype-specific facts computed by :func:`probe_frame`.
        source_facts: all facts computed for the column, as stored before any
            conversion.
        frame: the preprocessed frame, for the rare checks that cannot be
            expressed in Polars (files on disk, image headers).
    """

    name: str
    expr: pl.Expr
    dtype: pl.DataType
    facts: dict[str, Any] = field(default_factory=dict)
    source_facts: dict[str, Any] = field(default_factory=dict)
    frame: pl.LazyFrame | None = None

    def converted(
        self, expr: pl.Expr, dtype: pl.DataType, facts: dict[str, Any] | None = None
    ) -> InferenceColumn:
        """This column under a new representation."""
        return replace(
            self, expr=expr, dtype=dtype, facts=self.facts if facts is None else facts
        )

    def distinct_values(self) -> list[Any]:
        """The distinct non-null values (materialised; used by File and Image)."""
        if self.frame is None:
            return []
        unique = self.frame.select(self.expr.drop_nulls().unique())
        return collect(unique).to_series().to_list()


def _autocorrelation(expr: pl.Expr, lag: int) -> pl.Expr:
    """Pearson correlation of the values with their lag, on complete pairs."""
    lagged = expr.shift(lag)
    complete = expr.is_not_null() & lagged.is_not_null()
    return pl.corr(expr.filter(complete), lagged.filter(complete))


def _column_facts(
    config: Settings, name: str, dtype: pl.DataType
) -> dict[str, pl.Expr]:
    """The fact expressions of one column."""
    col = pl.col(name)
    facts: dict[str, pl.Expr] = {"count": col.count()}
    if ex.is_hashable_dtype(dtype) and dtype != pl.Null:
        facts["n_unique"] = col.drop_nulls().n_unique()
    size = ex.estimated_size(col, dtype)
    if size is not None:
        facts["memory_size"] = size

    timeseries = config.vars.timeseries
    if ex.is_numeric_dtype(dtype) and timeseries.active:
        values = col.cast(pl.Float64)
        for lag in timeseries.lags:
            facts[f"autocorr_{lag}"] = _autocorrelation(values, lag)

    if ex.is_string_dtype(dtype):
        stripped = col.str.strip_chars()
        mapping = {k.lower(): v for k, v in config.vars.bool.mappings.items()}
        facts["is_bool"] = col.str.to_lowercase().is_in(list(mapping)).all()

        # Numbers: every value must convert (as pandas' astype(float) requires)
        raw_number = stripped.cast(pl.Float64, strict=False)
        number = ex.parse_numeric(col)
        facts["num_ok"] = (raw_number.is_not_null() | col.is_null()).all()
        facts["num_count"] = number.count()
        facts["num_n_unique"] = number.drop_nulls().n_unique()
        if timeseries.active:
            for lag in timeseries.lags:
                facts[f"num_autocorr_{lag}"] = _autocorrelation(number, lag)

        # Dates are expensive to parse (one attempt per format), so this pass
        # only parses the first non-empty values; the columns whose sample
        # parses get the full date facts in a second pass (_date_facts).
        sample = col.filter(stripped != "").head(DATE_SAMPLE_SIZE)
        facts["dt_sample_ok"] = ex.parse_datetime(sample).is_not_null().all()

        if config.vars.url.active:
            facts["is_url"] = col.str.contains(ex.URL_LIKE_PATTERN).all()
        if config.vars.path.active:
            facts["is_abspath"] = col.str.contains(ex.ABSOLUTE_PATH_PATTERN).all()
    return facts


# Non-empty values of a string column parsed as dates in the first pass.
DATE_SAMPLE_SIZE = 1000


def _date_facts(name: str) -> dict[str, pl.Expr]:
    """Full-column date facts of a string column whose sample parsed as dates.

    Every non-empty value must parse; empty strings become missing values.
    """
    col = pl.col(name)
    date = ex.parse_datetime(col)
    parsed = date.is_not_null()
    number = ex.parse_numeric(col)
    return {
        "dt_ok": (parsed | col.is_null() | (col.str.strip_chars() == "")).all(),
        "dt_count": date.count(),
        "dt_n_unique": date.drop_nulls().n_unique(),
        "dt_digits": col.filter(parsed).str.contains(r"\d").all(),
        "dt_plain_numbers": number.filter(parsed).is_not_null().all(),
    }


def _evaluate_facts(
    frame: pl.LazyFrame, facts: dict[str, dict[str, pl.Expr]]
) -> dict[str, dict[str, Any]]:
    """Evaluate the fact expressions of several columns in one select."""
    exprs: list[pl.Expr] = []
    keys: list[tuple[str, str]] = []
    for name, column_facts in facts.items():
        for fact, expr in column_facts.items():
            keys.append((name, fact))
            exprs.append(expr.alias(f"__fact_{len(keys)}"))
    values: dict[str, dict[str, Any]] = {name: {} for name in facts}
    if exprs:
        row = collect(frame.select(exprs)).row(0)
        for (name, fact), value in zip(keys, row):
            values[name][fact] = value
    return values


def probe_frame(
    config: Settings, frame: pl.LazyFrame, schema: pl.Schema | None = None
) -> tuple[int, dict[str, InferenceColumn]]:
    """Compute the inference facts of every column in one pass.

    A second pass computes the date facts of the string columns whose first
    values parse as dates (none in most frames).

    Args:
        config: report settings.
        frame: the preprocessed LazyFrame.
        schema: its schema, if already known.

    Returns:
        The number of rows and an :class:`InferenceColumn` per column.
    """
    schema = schema if schema is not None else frame.collect_schema()
    facts_exprs = {
        name: _column_facts(config, name, dtype) for name, dtype in schema.items()
    }
    facts_exprs["__table"] = {"n": pl.len()}
    by_column = _evaluate_facts(frame, facts_exprs)
    n_rows = int(by_column.pop("__table")["n"])

    date_candidates = [
        name for name, facts in by_column.items() if facts.get("dt_sample_ok")
    ]
    for name, facts in _evaluate_facts(
        frame, {name: _date_facts(name) for name in date_candidates}
    ).items():
        by_column[name].update(facts)

    columns = {}
    for name, dtype in schema.items():
        facts = by_column[name]
        facts["n"] = n_rows
        if config.vars.timeseries.active:
            facts["autocorr"] = [
                facts.get(f"autocorr_{lag}") for lag in config.vars.timeseries.lags
            ]
            facts["num_autocorr"] = [
                facts.get(f"num_autocorr_{lag}") for lag in config.vars.timeseries.lags
            ]
        columns[name] = InferenceColumn(
            name=name,
            expr=pl.col(name),
            dtype=dtype,
            facts={
                "n": n_rows,
                "count": facts.get("count", 0),
                "n_unique": facts.get("n_unique"),
                "autocorr": facts.get("autocorr", []),
                **{k: v for k, v in facts.items() if k.startswith("is_")},
            },
            source_facts=facts,
            frame=frame,
        )
    return n_rows, columns
