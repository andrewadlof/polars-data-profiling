import datetime as dt

import numpy as np
import polars as pl
import pytest
from polars.io.plugins import register_io_source

from polars_data_profiling.config import Settings
from polars_data_profiling.model.describe import describe
from polars_data_profiling.model.summarizer import ProfilingSummarizer
from polars_data_profiling.model.typeset import ProfilingTypeSet


@pytest.fixture
def config() -> Settings:
    cfg = Settings()
    cfg.progress_bar = False
    return cfg


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """A frame with one column of every kind the profiler distinguishes."""
    rng = np.random.default_rng(0)
    n = 500
    return pl.DataFrame(
        {
            "int": rng.integers(0, 100, n),
            "float": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
            "few_ints": rng.integers(0, 3, n),
            "num_str": [f"{x:.3f}" for x in rng.normal(size=n)],
            "bool_str": rng.choice(["yes", "no"], n).tolist(),
            "bool": rng.random(n) < 0.7,
            "cat": pl.Series(rng.choice(["apple", "banana", "cherry"], n).tolist()),
            "text": [f"the quick brown fox number {i}" for i in range(n)],
            "date_str": [f"2020-01-{(i % 28) + 1:02d}" for i in range(n)],
            "date": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "list": [[1, 2]] * n,
            "const": ["x"] * n,
        }
    ).with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col("cat"))
        .alias("cat")
    )


def run_describe(df, config: Settings, **kwargs):
    typeset = ProfilingTypeSet(config, kwargs.pop("type_schema", None))
    return describe(config, df, ProfilingSummarizer(typeset), typeset, **kwargs)


@pytest.fixture
def counting_scan():
    """Build a LazyFrame over a DataFrame that counts how often it is read."""

    def build(data: pl.DataFrame):
        reads = []

        def source(with_columns, predicate, n_rows, batch_size):
            reads.append(with_columns)
            df = data if with_columns is None else data.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            if n_rows is not None:
                df = df.head(n_rows)
            yield df

        return register_io_source(io_source=source, schema=data.schema), reads

    return build
