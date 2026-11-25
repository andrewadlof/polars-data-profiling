"""The default typeset: the variable types of a profile and how they are inferred."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial

import filetype
import polars as pl

from polars_data_profiling.config import Settings
from polars_data_profiling.model import expressions as ex
from polars_data_profiling.model.probe import InferenceColumn
from polars_data_profiling.model.typeset_base import (
    BaseTypeSet,
    Generic,
    IdentityRelation,
    InferenceRelation,
    ProfilingBaseType,
    TypeRelation,
    has_values,
)
from polars_data_profiling.model.typeset_relations import (
    numeric_is_category,
    string_is_bool,
    string_is_category,
    string_is_datetime,
    string_is_numeric,
    string_to_bool,
    string_to_datetime,
    string_to_numeric,
    to_category,
)


def typeset_types(config: Settings) -> list[type[ProfilingBaseType]]:
    """Define the active types, in precedence order, based on the config.

    When more than one relation leaving a type matches a column, the target
    listed first wins:

    - Boolean before Numeric: a string listed in ``vars.bool.mappings`` is a
      flag, even when it also parses as a number (e.g. a user maps "1"/"0").
    - DateTime, URL and Path before Categorical: a date, URL or path column
      keeps its type even when it has few distinct values.
    - Categorical before TimeSeries: a numeric column with at most
      ``vars.num.low_categorical_threshold`` distinct values is categorical in
      every mode, as it is when the same values arrive as strings.
    - Categorical before Text: the low-cardinality reading wins over free text.
    """

    class Unsupported(Generic):
        """Base type. All other types have a relationship with this type."""

    class Numeric(ProfilingBaseType):
        """Integer, float and decimal columns, and strings of numbers."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=partial(string_is_numeric, k=config),
                    transformer=string_to_numeric,
                ),
            ]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return ex.is_numeric_dtype(column.dtype)

    class Text(ProfilingBaseType):
        """Free text: names, notes, identifiers."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Unsupported)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return ex.is_string_dtype(column.dtype)

    class DateTime(ProfilingBaseType):
        """Date and datetime columns, and strings of dates."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=string_is_datetime,
                    transformer=string_to_datetime,
                ),
            ]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return ex.is_temporal_dtype(column.dtype)

    class Categorical(ProfilingBaseType):
        """Categorical and Enum columns, and numbers or strings with few
        distinct values."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Numeric,
                    relationship=partial(numeric_is_category, k=config),
                    transformer=to_category,
                ),
                InferenceRelation(
                    Text,
                    relationship=partial(string_is_category, k=config),
                    transformer=to_category,
                ),
            ]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return ex.is_categorical_dtype(column.dtype)

    class Boolean(ProfilingBaseType):
        """Boolean columns, and strings that map to booleans."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            # Numeric [0, 1] goes via Categorical (few distinct numbers)
            mapping = config.vars.bool.mappings
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=partial(string_is_bool, k=mapping),
                    transformer=partial(string_to_bool, k=mapping),
                ),
            ]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return column.dtype == pl.Boolean

    class URL(ProfilingBaseType):
        """Text whose every value has a scheme and a host."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Text)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return bool(column.facts.get("is_url"))

    class Path(ProfilingBaseType):
        """Text whose every value is an absolute path."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Text)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            return bool(column.facts.get("is_abspath"))

    class File(ProfilingBaseType):
        """Paths that all exist on disk."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Path)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            paths = state.setdefault("distinct_values", column.distinct_values())
            return all(os.path.exists(p) for p in paths)

    class Image(ProfilingBaseType):
        """Files that are all images."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(File)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            paths = state.setdefault("distinct_values", column.distinct_values())

            def is_image(path: str) -> bool:
                guess = filetype.guess(str(path))
                return guess is not None and guess.mime.startswith("image/")

            return all(is_image(p) for p in paths)

    class TimeSeries(ProfilingBaseType):
        """Numeric columns that correlate with their own past values."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Numeric)]

        @staticmethod
        @has_values
        def contains_op(column: InferenceColumn, state: dict) -> bool:
            if not ex.is_numeric_dtype(column.dtype):
                return False
            threshold = config.vars.timeseries.autocorrelation
            return any(
                value is not None and value == value and value >= threshold
                for value in column.facts.get("autocorr", [])
            )

    active = {Unsupported, Boolean, Numeric, Text, Categorical, DateTime}
    if config.vars.path.active:
        active.add(Path)
        if config.vars.file.active:
            active.add(File)
            if config.vars.image.active:
                active.add(Image)
    if config.vars.url.active:
        active.add(URL)
    if config.vars.timeseries.active:
        active.add(TimeSeries)

    precedence = [
        Unsupported,
        Boolean,
        Numeric,
        DateTime,
        URL,
        Path,
        File,
        Image,
        Categorical,
        TimeSeries,
        Text,
    ]
    return [t for t in precedence if t in active]


class ProfilingTypeSet(BaseTypeSet):
    """The default typeset, configured by the report settings."""

    def __init__(self, config: Settings, type_schema: dict | None = None):
        self.config = config
        super().__init__(typeset_types(config))
        self.type_schema = self._init_type_schema(type_schema or {})

    def _init_type_schema(self, type_schema: dict) -> dict:
        return {k: self._get_type(v) for k, v in type_schema.items()}
