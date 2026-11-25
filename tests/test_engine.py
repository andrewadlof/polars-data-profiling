"""Batched lazy execution: how often a profile reads its source."""

import numpy as np
import polars as pl
import pytest

from polars_data_profiling.config import Settings
from polars_data_profiling.model.engine import Request, run_plans
from polars_data_profiling.model.summarizer import ProfilingSummarizer
from polars_data_profiling.model.typeset import ProfilingTypeSet

from .conftest import run_describe


def test_run_plans_batches_requests_of_all_plans():
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    rounds = []

    def plan(col):
        r = yield Request(aggs={"sum": pl.col(col).sum()})
        rounds.append(1)
        r2 = yield Request(
            frames={"head": lf.select(pl.col(col).head(r.values["sum"] // 10))}
        )
        return r2.frames["head"].height

    results = run_plans(lf, {"a": plan("a"), "b": plan("b"), "c": 42})
    assert results == {"a": 0, "b": 3, "c": 42}
    assert len(rounds) == 2


def frame(n_cols: int, n_rows: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(n_cols):
        kind = i % 4
        if kind == 0:
            cols[f"num{i}"] = rng.normal(size=n_rows)
        elif kind == 1:
            cols[f"int{i}"] = rng.integers(0, 50, n_rows)
        elif kind == 2:
            cols[f"cat{i}"] = rng.choice(["a", "b", "c"], n_rows).tolist()
        else:
            cols[f"txt{i}"] = [f"value {j} {i}" for j in range(n_rows)]
    return pl.DataFrame(cols)


@pytest.mark.parametrize("n_cols", [4, 16])
def test_source_reads_do_not_grow_with_columns(config, counting_scan, n_cols):
    lf, reads = counting_scan(frame(n_cols))
    run_describe(lf, config)
    # probe + summary rounds + correlations, scatter, missing, sample, duplicates
    assert len(reads) <= 12


def test_without_shared_scans_queries_read_their_own_columns(config, counting_scan):
    config.engine.share_scans = False
    data = frame(8)
    lf, reads = counting_scan(data)
    d = run_describe(lf, config)
    assert len(reads) > 12
    assert any(r is not None and len(r) < data.width for r in reads)
    shared = run_describe(data, Settings(progress_bar=False))
    assert d.variables["num0"]["mean"] == pytest.approx(
        shared.variables["num0"]["mean"]
    )


def test_streaming_engine(config):
    config.engine.collect_engine = "streaming"
    d = run_describe(frame(8), config)
    assert d.variables["num0"]["type"] == "Numeric"


def test_custom_summarizer_with_plain_and_lazy_steps(config):
    typeset = ProfilingTypeSet(config)
    summarizer = ProfilingSummarizer(typeset)

    def plain(config, column, summary):
        summary["my_max"] = column.series().max()
        return config, column, summary

    def lazy(config, column, summary):
        response = yield Request(aggs={"n_big": (column.expr > 1).sum()})
        summary["n_big"] = response.values["n_big"]

    summarizer.mapping["Numeric"] = [*summarizer.mapping["Numeric"], plain, lazy]
    from polars_data_profiling.model.describe import describe

    d = describe(
        config, pl.DataFrame({"x": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]}), summarizer, typeset
    )
    assert d.variables["x"]["my_max"] == 5.5
    assert d.variables["x"]["n_big"] == 5
