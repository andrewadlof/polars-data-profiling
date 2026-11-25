# polars-data-profiling

Profiling reports for Polars DataFrames and LazyFrames: types, statistics, distributions, correlations, missing values,
duplicates and alerts for every column, in one HTML page or JSON document.

```bash
uv add polars-data-profiling
```

```python
import polars as pl
from polars_data_profiling import ProfileReport

lf = pl.scan_parquet("events/*.parquet")
ProfileReport(lf, title="Events").to_file("events.html")
```

It is a fork of [fg-data-profiling](https://github.com/Data-Centric-AI-Community/fg-data-profiling) (formerly
ydata-profiling and pandas-profiling) with the pandas and Spark backends replaced by a single engine built on Polars
LazyFrames:

- **Lazy.** Every statistic is a lazy query. A LazyFrame is never collected whole.
- **Batched.** The queries of all columns are evaluated together, so a profile reads its source about a dozen times,
  whatever the number of columns.
- **No pandas.** pandas is not a dependency, direct or transitive. CI fails if it ever appears in the dependency tree.
- **Same report.** The HTML report, its configuration and most of the API are unchanged.

Next: [Install](install.md), [Usage](usage.md), [How it works](architecture.md), or
[Coming from ydata-profiling](migrating.md).
