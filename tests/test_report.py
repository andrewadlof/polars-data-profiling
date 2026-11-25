import json

import numpy as np
import polars as pl
import pytest

import polars_data_profiling
from polars_data_profiling import ProfileReport, compare
from polars_data_profiling.controller.console import main as cli_main
from polars_data_profiling.report.presentation.frequency_table_utils import freq_table


def report(df, **kwargs) -> ProfileReport:
    return ProfileReport(df, progress_bar=False, **kwargs)


def test_html_and_json(mixed_df, tmp_path):
    profile = report(mixed_df, title="Mixed")
    html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
    profile.to_file(html_path)
    profile.to_file(json_path)
    html = html_path.read_text(encoding="utf-8")
    assert "<svg" in html and "Mixed" in html
    assert "<img" not in html  # charts are inline SVG
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data["variables"]) == set(mixed_df.columns)
    assert data["variables"]["int"]["type"] == "Numeric"
    assert isinstance(data["variables"]["int"]["histogram"]["counts"], list)
    assert data["table"]["n"] == mixed_df.height


@pytest.mark.parametrize("writer", ["parquet", "csv", "ipc", "ndjson"])
def test_scanned_files(mixed_df, tmp_path, writer):
    df = mixed_df.drop("list")
    path = tmp_path / f"data.{writer}"
    getattr(df, f"write_{writer}")(path)
    scan = {
        "parquet": pl.scan_parquet,
        "csv": pl.scan_csv,
        "ipc": pl.scan_ipc,
        "ndjson": pl.scan_ndjson,
    }
    profile = report(scan[writer](path))
    d = profile.get_description()
    assert d.table["n"] == df.height
    assert d.variables["date_str"]["type"] == "DateTime"
    assert d.variables["num_str"]["type"] == "Numeric"
    assert profile.to_html()


def test_minimal_and_explorative(mixed_df):
    minimal = report(mixed_df, minimal=True).get_description()
    assert minimal.correlations == {}
    assert minimal.missing == {}
    explorative = report(mixed_df, explorative=True).get_description()
    assert "word_counts" in explorative.variables["text"]


def test_sensitive_mode_hides_values(mixed_df):
    df = mixed_df.with_columns(pl.lit("SECRET-VALUE").alias("secret"))
    html = report(df, sensitive=True).to_html()
    assert "SECRET-VALUE" not in html
    data = json.loads(report(df, sensitive=True).to_json())
    assert "first_rows" not in data["variables"]["secret"]
    assert "SECRET-VALUE" not in json.dumps(
        data["variables"]["secret"]["value_counts_without_nan"]
    )


def test_namespace_on_dataframe_and_lazyframe(mixed_df):
    assert isinstance(mixed_df.profile.report(progress_bar=False), ProfileReport)
    lazy = mixed_df.lazy().profile.report(progress_bar=False, minimal=True)
    assert lazy.get_description().table["n"] == mixed_df.height


def test_samples_and_duplicates(config):
    df = pl.DataFrame({"a": list(range(30)) + [1, 1, 2], "b": ["x"] * 33})
    profile = report(df, samples={"head": 3, "tail": 2, "random": 4})
    samples = {s.id: s for s in profile.get_sample()}
    assert samples["head"].data.height == 3
    assert samples["tail"].index == [31, 32]
    assert samples["random"].data.height == 4
    duplicates = profile.get_duplicates()
    assert duplicates.row(0, named=True) == {"a": 1, "b": "x", "# duplicates": 3}
    assert profile.get_description().table["n_duplicates"] == 2


def test_custom_sample():
    df = pl.DataFrame({"a": list(range(10))})
    profile = report(df, sample={"name": "Mine", "data": df.lazy().head(2)})
    assert profile.get_sample()[0].data.height == 2
    assert "Mine" in profile.to_html()


def test_missing_diagrams(config):
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    a[::3] = np.nan
    b[::3] = np.nan  # missing together with a
    b[1::10] = np.nan
    d = report(
        pl.DataFrame({"a": a, "b": b, "c": rng.normal(size=300)})
    ).get_description()
    assert set(d.missing) == {"bar", "matrix", "heatmap"}
    assert "<svg" in d.missing["heatmap"]["matrix"]


def test_compare_reports(mixed_df):
    first = report(mixed_df.head(250), title="First")
    second = report(mixed_df.tail(250), title="Second")
    merged = compare([first, second])
    d = merged.get_description()
    assert d.table["n"] == [250, 250]
    assert isinstance(d.variables["cat"]["value_counts_without_nan"], list)
    html = merged.to_html()
    assert "First" in html and "Second" in html


def test_serialization_roundtrip(mixed_df, tmp_path):
    profile = report(mixed_df)
    profile.get_description()  # dump() stores what has been computed
    path = tmp_path / "profile.pp"
    profile.dump(path)
    loaded = report(None).load(path)
    assert loaded.get_description().table["n"] == mixed_df.height
    assert loaded.to_html()
    with pytest.raises(ValueError, match="no DataFrame"):
        report(None).get_description()


def test_type_schema_and_rejected_variables(mixed_df):
    profile = report(mixed_df, type_schema={"int": "categorical"})
    d = profile.get_description()
    assert d.variables["int"]["type"] == "Categorical"
    assert "list" in profile.get_rejected_variables()


def test_inputs():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"x": [1, 2, 3, 4, 5, 6], "y": ["a", "b", "c", "d", "e", "f"]})
    assert report(table).get_description().table["n"] == 6

    class FakePandas:
        pass

    FakePandas.__module__ = "pandas.core.frame"
    with pytest.raises(TypeError, match="pl.from_pandas"):
        report(FakePandas())
    with pytest.raises(ValueError, match="columns"):
        report(pl.DataFrame())


def test_object_columns_of_strings_are_profiled_as_strings():
    # polars builds an Object column from a numpy object array mixing str and None
    values = np.random.default_rng(0).choice(["a", "b", "c", None], 2000)
    df = pl.DataFrame({"o": values})
    if df.schema["o"] != pl.Object:
        pytest.skip("this polars version infers the column as strings itself")
    assert report(df).get_description().variables["o"]["type"] == "Categorical"
    assert report(df.lazy()).get_description().variables["o"]["type"] == "Categorical"


def test_zero_rows():
    profile = report(pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}))
    assert profile.get_description().table["n"] == 0
    assert profile.to_html()


def test_config_file_and_kwargs(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("title: From file\nsamples:\n  head: 1\n", encoding="utf-8")
    profile = report(pl.DataFrame({"a": [1, 2, 3]}), config_file=path, duplicates=None)
    assert profile.config.title == "From file"
    assert profile.config.duplicates.head == 0
    with pytest.raises(ValueError):
        ProfileReport(pl.DataFrame({"a": [1]}), minimal=True, config_file=path)


def test_frequency_table_of_truncated_counts():
    rows = freq_table(
        {"a": 50, "b": 30, "c": 5}, n=100, max_number_to_print=2, total=90, n_distinct=6
    )[0]
    labels = [r["label"] for r in rows]
    assert labels == ["a", "b", "Other values (4)", "(Missing)"]
    assert rows[2]["count"] == 10 and rows[3]["count"] == 10


def test_cli(tmp_path):
    data = tmp_path / "data.csv"
    pl.DataFrame({"a": list(range(20)), "b": ["x", "y"] * 10}).write_csv(data)
    out = tmp_path / "out.html"
    cli_main(["--silent", "--minimal", str(data), str(out)])
    assert out.exists() and "<svg" in out.read_text(encoding="utf-8")


def test_version():
    assert polars_data_profiling.__version__
