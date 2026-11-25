"""Mapping from variable types to the summary steps that describe them."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from polars_data_profiling.model.engine import Plan
from polars_data_profiling.model.typeset_base import BaseTypeSet


def compose(functions: Sequence[Callable]) -> Callable[..., Plan]:
    """Chain summary steps into one plan.

    A step is called as ``step(config, column, summary)``. Generator steps are
    run as part of the plan (their lazy requests are batched with those of
    other columns). A plain function may update ``summary`` in place, return a
    new summary dict, or return a ``(config, column, summary)`` tuple as the
    pandas-era steps did.
    """

    def composed(config: Any, column: Any, summary: dict) -> Plan:
        for func in functions:
            result = func(config, column, summary)
            if inspect.isgenerator(result):
                result = yield from result
            if isinstance(result, tuple) and result:
                result = result[-1]
            if isinstance(result, dict) and result is not summary:
                summary.clear()
                summary.update(result)
        return summary

    return composed


class Handler:
    """A mapping from type names to summary steps.

    A type runs the steps of its parent type first (e.g. ``Numeric`` gets the
    counts of ``Unsupported`` before its own statistics).
    """

    def __init__(
        self, mapping: dict[str, list[Callable]], typeset: BaseTypeSet, *args, **kwargs
    ):
        self.mapping = mapping
        self.typeset = typeset
        self._complete_dag()

    def _complete_dag(self) -> None:
        # Parents come first in the topological order, so their lists are
        # already complete.
        for type_ in self.typeset.topological_order():
            parent = self.typeset.parent_of(type_)
            if parent is None:
                continue
            self.mapping[str(type_)] = self.mapping.get(
                str(parent), []
            ) + self.mapping.get(str(type_), [])

    def handle(self, dtype: str, config: Any, column: Any, summary: dict) -> Plan:
        """The plan that summarises a column of type ``dtype``."""
        return compose(self.mapping.get(dtype, []))(config, column, summary)


def get_render_map() -> dict[str, Callable]:
    import polars_data_profiling.report.structure.variables as render_algorithms

    return {
        "Boolean": render_algorithms.render_boolean,
        "Numeric": render_algorithms.render_real,
        "Complex": render_algorithms.render_complex,
        "Text": render_algorithms.render_text,
        "DateTime": render_algorithms.render_date,
        "Categorical": render_algorithms.render_categorical,
        "URL": render_algorithms.render_url,
        "Path": render_algorithms.render_path,
        "File": render_algorithms.render_file,
        "Image": render_algorithms.render_image,
        "Unsupported": render_algorithms.render_generic,
        "TimeSeries": render_algorithms.render_timeseries,
    }
