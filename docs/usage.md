# Usage

## A report from a frame

```python
import polars as pl
from polars_data_profiling import ProfileReport

df = pl.read_csv("titanic.csv")
report = ProfileReport(df, title="Titanic")
report.to_file("titanic.html")
```

`ProfileReport` takes a `pl.DataFrame` or a `pl.LazyFrame`. Nothing is computed until the report is first used (written,
rendered or described). Pass `lazy=False` to compute it immediately.

## Large data: pass a LazyFrame

```python
lf = pl.scan_parquet("events/*.parquet")
ProfileReport(lf, title="Events", minimal=True).to_file("events.html")
```

A LazyFrame is never collected whole. Every statistic is a lazy query, and the queries of all columns are batched, so the
source is read about a dozen times in total, whatever the number of columns. [How it works](architecture.md) explains the
batching and the `engine` settings that trade memory for reads.

## The `profile` namespace

Importing the package registers a `profile` namespace on Polars frames:

```python
import polars_data_profiling  # noqa: F401

df.profile.report(title="Quick look").to_file("quick.html")
pl.scan_csv("big.csv").profile.report(minimal=True).to_file("big.html")
```

`report(...)` accepts every argument of `ProfileReport`.

## Outputs

```python
report.to_file("report.html")  # a self-contained HTML file
report.to_file("report.json")  # the statistics as JSON
html = report.to_html()
data = report.to_json()
report.to_notebook_iframe()  # in Jupyter (also what displaying `report` does)
report.to_widgets()  # Jupyter widgets, with the `notebook` extra
```

The statistics are also available as Python objects:

```python
description = report.get_description()
description.table["n"]  # number of rows
description.variables["age"]["mean"]
description.variables["age"]["histogram"]  # (counts, bin_edges) numpy arrays
description.correlations["auto"]  # a square pl.DataFrame
report.get_duplicates()  # pl.DataFrame of the most frequent duplicate rows
report.get_sample()  # samples, each with a pl.DataFrame in .data
report.get_rejected_variables()
```

Value counts (`value_counts_without_nan`, `word_counts`, `character_counts`, ...) are dicts, most frequent first. To keep
summaries small on large data, `value_counts_without_nan` holds the most frequent values and `value_counts_index_sorted` the
smallest and largest ones. `n_distinct` and `count` are always exact.

## Presets

```python
ProfileReport(
    df, minimal=True
)  # no correlations, interactions or missing-value diagrams
ProfileReport(df, explorative=True)  # adds URL, path, file and image analysis
ProfileReport(
    df, sensitive=True
)  # hides values of categorical and text columns, no samples
```

## Time series

```python
ProfileReport(df, tsmode=True, sortby="timestamp")
```

Rows are sorted by `sortby`, which also becomes the time index. Numeric columns that correlate with their own past (see
`vars.timeseries` in [Configuration](configuration.md)) are profiled as time series: stationarity (augmented Dickey-Fuller),
seasonality, gaps, ACF and PACF.

## Declaring types

Type inference reads the values: strings of numbers become numbers, strings of dates become dates, numbers with few distinct
values become categories. To skip inference for some columns, name their type:

```python
ProfileReport(df, type_schema={"zip_code": "Categorical", "amount": "Numeric"})
```

## Comparing two datasets

```python
from polars_data_profiling import compare

train = ProfileReport(train_df, title="Train")
test = ProfileReport(test_df, title="Test")
compare([train, test]).to_file("train_vs_test.html")
# or
train.compare(test).to_file("train_vs_test.html")
```

## Saving and loading a computed report

```python
report.get_description()  # compute it first: dump() saves what has been computed
report.dump("report.pp")
loaded = ProfileReport(None).load("report.pp")
loaded.to_file("report.html")
```

## Command line

```bash
polars_data_profiling data.parquet report.html
polars_data_profiling --minimal --title "Events" events.csv
```

CSV, TSV, Parquet, Arrow IPC / Feather and NDJSON files are opened lazily, JSON eagerly.
