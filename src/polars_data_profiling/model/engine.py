"""Batched lazy execution of column summaries.

Every statistic in a profile is a Polars expression or a small LazyFrame query.
Running them column by column would scan the data once per column and per step.
Instead each summary step is written as a *generator* that yields a
:class:`Request` (the scalar aggregations and LazyFrame queries it needs) and
receives a :class:`Response` with the results::

    def describe_numeric(config, column, summary):
        response = yield Request(aggs={"mean": column.expr.mean()})
        summary["mean"] = response.values["mean"]

:func:`run_plans` advances the generators of all columns in lock step. In every
round it gathers the requests of every column that is waiting, evaluates all
scalar aggregations in a single ``LazyFrame.select`` and all queries in a single
``polars.collect_all`` call (which shares common sub-plans such as the scan of
the source), and sends each generator its own slice of the results. A profile
of ``k`` columns therefore costs a handful of passes over the data instead of
``k`` times that.

Plain (non-generator) functions are accepted wherever a step is expected, so a
custom summarizer can still be written as an ordinary function that
materialises what it needs with :meth:`ColumnRef.series`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from polars_data_profiling.utils.collect import collect, collect_all


@dataclass
class Request:
    """What a summary step needs evaluated.

    Attributes:
        aggs: named expressions evaluated together over the profiled frame. Each
            must reduce to a single value (an aggregation).
        frames: named LazyFrames, collected together.
    """

    aggs: dict[str, pl.Expr] = field(default_factory=dict)
    frames: dict[str, pl.LazyFrame] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.aggs or self.frames)


@dataclass
class Response:
    """Results of a :class:`Request`, keyed like the request."""

    values: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)


Plan = Generator[Request, Response, Any]


@dataclass(frozen=True)
class ColumnRef:
    """A column of the profiled frame, as summary steps see it.

    Attributes:
        name: the column name.
        frame: the typed LazyFrame (columns cast to their inferred type).
        dtype: the Polars dtype of the column in ``frame``.
        source_dtype: the dtype of the column in the input data.
    """

    name: str
    frame: pl.LazyFrame
    dtype: pl.DataType
    source_dtype: pl.DataType

    @property
    def expr(self) -> pl.Expr:
        """The column as an expression on :attr:`frame`."""
        return pl.col(self.name)

    def lazy(self) -> pl.LazyFrame:
        """A single-column LazyFrame."""
        return self.frame.select(self.expr)

    def series(self) -> pl.Series:
        """Materialise the column (for steps that cannot stay lazy)."""
        return collect(self.lazy()).to_series()


def run_plans(
    frame: pl.LazyFrame,
    plans: Mapping[Hashable, Plan | Any],
    on_done: Callable[[Hashable], None] | None = None,
) -> dict[Hashable, Any]:
    """Drive several plans to completion, batching their requests.

    Args:
        frame: the LazyFrame the scalar aggregations are evaluated on.
        plans: plans keyed by an identifier. A value that is not a generator is
            taken as an already-computed result.
        on_done: called with the key of each plan as it finishes.

    Returns:
        The return value of every plan, by key.
    """
    results: dict[Hashable, Any] = {}
    waiting: dict[Hashable, tuple[Plan, Request]] = {}

    def advance(key: Hashable, plan: Plan, response: Response | None) -> None:
        try:
            request = plan.send(response) if response is not None else next(plan)
        except StopIteration as stop:
            results[key] = stop.value
            if on_done is not None:
                on_done(key)
            return
        if not isinstance(request, Request):
            raise TypeError(
                f"A summary plan must yield Request objects, got {type(request)!r}"
            )
        waiting[key] = (plan, request)

    for key, plan in plans.items():
        if inspect.isgenerator(plan):
            advance(key, plan, None)
        else:
            results[key] = plan
            if on_done is not None:
                on_done(key)

    while waiting:
        batch = dict(waiting)
        waiting.clear()
        responses = _evaluate(
            frame, {key: request for key, (_, request) in batch.items()}
        )
        for key, (plan, _) in batch.items():
            advance(key, plan, responses[key])

    return results


def _evaluate(
    frame: pl.LazyFrame, requests: Mapping[Hashable, Request]
) -> dict[Hashable, Response]:
    """Evaluate a batch of requests with one select and one collect_all."""
    responses = {key: Response() for key in requests}

    # Scalar aggregations of every request go into one select. Output names are
    # positional, so arbitrary column and statistic names cannot collide.
    agg_exprs: list[pl.Expr] = []
    agg_targets: list[tuple[Hashable, str]] = []
    for key, request in requests.items():
        for name, expr in request.aggs.items():
            agg_targets.append((key, name))
            agg_exprs.append(expr.alias(f"__agg_{len(agg_exprs)}"))

    frame_targets: list[tuple[Hashable, str]] = []
    queries: list[pl.LazyFrame] = []
    for key, request in requests.items():
        for name, query in request.frames.items():
            frame_targets.append((key, name))
            queries.append(query)

    if agg_exprs:
        queries.append(frame.select(agg_exprs))

    collected = collect_all(queries) if queries else []

    if agg_exprs:
        row = collected.pop().row(0)
        for (key, name), value in zip(agg_targets, row):
            responses[key].values[name] = value

    for (key, name), df in zip(frame_targets, collected):
        responses[key].frames[name] = df

    return responses


def chain(*steps: Iterable[Any] | Plan | None) -> Plan:
    """Run plans one after the other as a single plan."""
    for step in steps:
        if inspect.isgenerator(step):
            yield from step
    return None
