import datetime as dt

import polars as pl
import pytest

from polars_data_profiling.config import Settings
from polars_data_profiling.model.probe import probe_frame
from polars_data_profiling.model.summary import infer_types
from polars_data_profiling.model.typeset import ProfilingTypeSet

N = 100


def infer(series: pl.Series, **settings) -> tuple[str, pl.DataType]:
    config = Settings()
    for path, value in settings.items():
        target = config
        *parents, leaf = path.split("__")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)
    typeset = ProfilingTypeSet(config)
    typed, _ = infer_types(config, series.to_frame().lazy(), typeset)
    return typed.types[series.name], typed.frame.collect_schema()[series.name]


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (list(range(N)), None, "Numeric"),
        ([i / 3 for i in range(N)], None, "Numeric"),
        ([i % 3 for i in range(N)], None, "Categorical"),
        ([str(i * 1.5) for i in range(N)], None, "Numeric"),
        ([f" {i} " for i in range(N)], None, "Numeric"),
        (["yes", "no"] * (N // 2), None, "Boolean"),
        (["True", "FALSE"] * (N // 2), None, "Boolean"),
        ([True, False] * (N // 2), None, "Boolean"),
        (["a", "b", "c", "d"] * (N // 4), None, "Categorical"),
        ([f"word {i}" for i in range(N)], None, "Text"),
        ([f"2020-01-{(i % 28) + 1:02d}" for i in range(N)], None, "DateTime"),
        (
            ["2020-01-01", "01/02/2020", "2020-03-04 10:11:12"] * 33 + [""],
            None,
            "DateTime",
        ),
        ([dt.date(2020, 1, 1)] * N, None, "DateTime"),
        ([dt.datetime(2020, 1, 1, 12)] * N, None, "DateTime"),
        ([str(2000 + i) for i in range(N)], None, "Numeric"),
        (["Jan", "Feb", "Mar", "Apr"] * (N // 4), None, "Categorical"),
        ([None] * N, pl.Null, "Unsupported"),
        ([None] * N, pl.String, "Unsupported"),
        ([[1, 2]] * N, None, "Unsupported"),
        ([{"a": 1}] * N, None, "Unsupported"),
        (["x", "y"] * (N // 2), pl.Categorical, "Categorical"),
        (["x", "y"] * (N // 2), pl.Enum(["x", "y"]), "Categorical"),
    ],
)
def test_inferred_type(values, dtype, expected):
    series = pl.Series("col", values, dtype=dtype)
    assert infer(series)[0] == expected


def test_conversions_change_the_dtype():
    assert infer(pl.Series("c", [str(i * 1.5) for i in range(N)]))[1] == pl.Float64
    assert infer(pl.Series("c", ["yes", "no"] * 50))[1] == pl.Boolean
    assert infer(pl.Series("c", [i % 3 for i in range(N)]))[1] == pl.String
    assert isinstance(
        infer(pl.Series("c", [f"2020-01-0{i % 9 + 1}" for i in range(N)]))[1],
        pl.Datetime,
    )
    assert infer(pl.Series("c", ["x", "y"] * 50, dtype=pl.Categorical))[1] == pl.String


def test_strings_that_are_mostly_numbers_stay_text():
    values = [str(i) for i in range(N - 1)] + ["not a number"]
    assert infer(pl.Series("c", values))[0] == "Text"


def test_url_and_path_types_need_activation():
    urls = pl.Series("c", [f"https://example.com/{i}" for i in range(N)])
    assert infer(urls)[0] == "Text"
    assert infer(urls, vars__url__active=True)[0] == "URL"


def test_timeseries_type():
    values = pl.Series("c", [float(i) for i in range(N)])
    assert infer(values)[0] == "Numeric"
    assert infer(values, vars__timeseries__active=True)[0] == "TimeSeries"


def test_infer_dtypes_off_keeps_the_storage_type():
    config = Settings()
    config.infer_dtypes = False
    typeset = ProfilingTypeSet(config)
    df = pl.LazyFrame({"num_str": [str(i) for i in range(N)]})
    typed, _ = infer_types(config, df, typeset)
    assert typed.types == {"num_str": "Text"}


def test_type_schema_overrides_inference():
    config = Settings()
    typeset = ProfilingTypeSet(config, {"a": "Categorical", "b": "numeric"})
    df = pl.LazyFrame({"a": list(range(N)), "b": [str(i) for i in range(N)]})
    typed, _ = infer_types(config, df, typeset)
    assert typed.types == {"a": "Categorical", "b": "Numeric"}
    assert typed.frame.collect_schema()["b"] == pl.Float64


def test_explain_type_reports_the_traversal():
    config = Settings()
    _, columns = probe_frame(config, pl.LazyFrame({"c": [str(i) for i in range(N)]}))
    trace = ProfilingTypeSet(config).explain_type(columns["c"])
    assert trace[0] == {"from": "Unsupported", "matched": ["Text"], "chosen": "Text"}
    assert trace[1]["chosen"] == "Numeric"


def test_probe_is_a_single_pass(counting_scan):
    data = pl.DataFrame(
        {f"c{i}": [str(j) for j in range(N)] for i in range(10)}
        | {f"n{i}": list(range(N)) for i in range(10)}
    )
    lf, reads = counting_scan(data)
    probe_frame(Settings(), lf)
    assert len(reads) == 1


def test_date_columns_get_a_second_probe_pass(counting_scan):
    data = pl.DataFrame(
        {"d": [f"2020-01-0{i % 9 + 1}" for i in range(N)], "x": list(range(N))}
    )
    lf, reads = counting_scan(data)
    _, columns = probe_frame(Settings(), lf)
    assert len(reads) == 2
    assert columns["d"].source_facts["dt_ok"] is True
