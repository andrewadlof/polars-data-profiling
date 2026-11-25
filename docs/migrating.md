# Coming from ydata-profiling

polars-data-profiling is a fork of
[fg-data-profiling](https://github.com/Data-Centric-AI-Community/fg-data-profiling) (published before as ydata-profiling and
pandas-profiling). The report, its configuration and most of the API are the same. The statistics engine was rewritten on
Polars LazyFrames, and everything that needed pandas was removed.

## Imports

```python
# before
from ydata_profiling import ProfileReport

df.profile_report()

# now
from polars_data_profiling import ProfileReport

df.profile.report()
```

## Inputs

`ProfileReport` takes a `pl.DataFrame` or `pl.LazyFrame` (and pyarrow tables, which Polars converts). A pandas DataFrame
raises a `TypeError` that points to `pl.from_pandas(df)`. Spark DataFrames are not supported.

Polars has no index, so:

- the index is never profiled or appended as a column.
- in time-series mode, `sortby` names the column that orders the rows and serves as the time index. Without it, the row
  number is the index.

## Results

| Then (pandas)                                       | Now                                                           |
| --------------------------------------------------- | ------------------------------------------------------------- |
| value counts as `pd.Series`                         | `dict`, most frequent first                                   |
| `value_counts_without_nan` holds every value        | holds the most frequent values (`n_distinct` stays exact)     |
| `value_counts_index_sorted` holds every value       | holds the smallest and largest values                         |
| correlation matrices as `pd.DataFrame`              | square `pl.DataFrame`, rows in the order of the columns       |
| samples and duplicates as `pd.DataFrame`            | `pl.DataFrame`                                                |
| `first_rows` as `pd.Series`                         | `pl.Series`                                                   |
| time series as `pd.Series` with an index            | `pl.DataFrame` with `index` and `value` columns               |
| `memory_size` from `DataFrame.memory_usage`         | estimated lazily from the data (bytes, offsets and validity)  |

## Removed

| Feature                          | Why                                                                      |
| -------------------------------- | ------------------------------------------------------------------------ |
| Spark backend                    | the single engine is Polars                                              |
| `phi_k` correlation              | the phik package requires pandas. `auto` and `cramers` cover categories  |
| matplotlib renderer (`plot.renderer`, `dpi`, PNG images) | charts are inline SVG only. seaborn required pandas |
| Great Expectations export (`to_expectation_suite`) | Great Expectations validates pandas frames                     |
| `vars.text.engine` (Arrow kernels) | the text statistics run in Polars                                      |
| `memory_deep`                    | kept as a setting but has no effect: sizes are estimated from the data   |
| usage analytics                  | the upstream package sent a request to packages.ydata.ai for every report. This fork sends nothing |

## Behaviour that changed

- **NaN is missing.** NaN in float columns is counted as missing, as pandas did. `inf` is a value (`n_infinite`).
- **Date parsing.** Strings are parsed against a fixed list of formats (ISO 8601 with or without time and offset, `m/d/Y`,
  `d/m/Y`, `d.m.Y`, month names) instead of `dateutil`. Month-first is tried before day-first, as in pandas.
- **ADF test, ACF and PACF** are computed with numpy. The results match statsmodels' `adfuller(regression="c",
  autolag="AIC")`, `acf(fft=True)` and `pacf(method="ywm")` to 1e-12 on the test series.
- **Histograms** use the same bins as before. With `plot.histogram.bins = 0` the bin count follows numpy's "auto" rule.
- **Object columns** of strings are re-read as strings, as `DataFrame.infer_objects()` did. Other `Object` columns are
  unsupported.
