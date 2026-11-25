# polars-data-profiling

Profiling reports for Polars DataFrames and LazyFrames, without pandas. For every column: its type, statistics, distribution,
missing values and alerts, plus correlations, interactions, samples and duplicate rows, in one HTML page or JSON document.

```python
import polars as pl
from polars_data_profiling import ProfileReport

lf = pl.scan_parquet("events/*.parquet")
ProfileReport(lf, title="Events").to_file("events.html")
```

Importing the package also registers a `profile` namespace on frames:

```python
import polars_data_profiling  # noqa: F401

df.profile.report(minimal=True).to_file("quick.html")
```

## Why this exists

[fg-data-profiling](https://github.com/Data-Centric-AI-Community/fg-data-profiling) (formerly ydata-profiling and
pandas-profiling) builds excellent reports, but on pandas: Polars data has to be converted first, every column is described
one at a time on materialised data, and pandas comes in through its own code and through phik, seaborn and statsmodels.

This fork keeps the report and replaces the engine with one built on Polars LazyFrames:

- **Lazy.** Every statistic is a lazy query. A LazyFrame is never collected whole, so `scan_parquet` and `scan_csv` inputs
  only materialise aggregates.
- **Batched.** Type inference is one `select` over all columns. Variable summaries are generators that yield their lazy
  requests, and the requests of every column are evaluated together, one `select` and one `collect_all` per round, sharing
  one read of the source. A profile reads its source about a dozen times, whatever the number of columns.
- **No pandas.** pandas is not a dependency, direct or transitive, with or without extras. CI fails if it appears in the
  resolved dependency tree, and checks that profiling never imports it.
- **Same report.** The HTML report, its configuration and most of the API are unchanged. See
  [Coming from ydata-profiling](https://andrewadlof.github.io/polars-data-profiling/migrating/).

## Install

```bash
uv add polars-data-profiling
```

Without uv, `pip install polars-data-profiling` works too. It needs Python 3.11+ and Polars 2.0+. Optional extras: `image`
(image columns), `wordcloud` (word clouds) and `notebook` (Jupyter widgets), or `all`.

## Speed

One 1,000,000-row, 15-column frame (6 float, 4 integer, 4 categorical and 1 unique-string column), default settings, on the
same Windows machine:

| Profiler                                                       | Time to HTML |
| -------------------------------------------------------------- | ------------ |
| fg-data-profiling 4.20.0 (pandas 2.3.3) on the frame converted | 83.3 s       |
| polars-data-profiling on the frame as a LazyFrame              | 8.9 s        |

The same profile reads its source 10 times. Without shared scans it would read it 255 times, once per query.

## Documentation

[andrewadlof.github.io/polars-data-profiling](https://andrewadlof.github.io/polars-data-profiling/): usage, configuration,
how the lazy engine works, and the differences from ydata-profiling.

## Development

```bash
uv sync --all-extras
uv run just check     # ruff, the pandas check and the tests
uv run just           # all recipes
```

See [Development](https://andrewadlof.github.io/polars-data-profiling/development/) for the recipes and the release process.

## License

MIT, like the project it is forked from. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Not affiliated with or endorsed by
YData Labs Inc.
