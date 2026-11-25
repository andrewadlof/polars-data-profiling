# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-25

First release of the fork of [fg-data-profiling](https://github.com/Data-Centric-AI-Community/fg-data-profiling) (at
`c27f8dcf`), renamed `polars-data-profiling` (`import polars_data_profiling`).

### Added

- A Polars engine built on LazyFrames. Type inference runs in one lazy pass over all columns. Variable summaries are
  generators that yield their lazy requests, and the requests of all columns are evaluated together, one `select` and one
  `collect_all` per round. Queries of a round share one read of the source through a `cache()` node.
- `ProfileReport` accepts `pl.DataFrame`, `pl.LazyFrame` and pyarrow tables.
- Python 3.14 support (Python 3.11 to 3.14 are tested on Linux, macOS and Windows).
- The `profile` namespace on frames: `df.profile.report(...)` and `lf.profile.report(...)`.
- `engine.share_scans` and `engine.collect_engine` settings (shared scans, and the Polars engine used to collect, for
  example `streaming`).
- Memory sizes estimated lazily from the data, nested types included.
- The command line tool opens CSV, TSV, Parquet, IPC and NDJSON files lazily.
- A test suite that checks the statistics against numpy and scipy, the ADF test against statsmodels, and how often a
  profile reads its source.

### Changed

- Value counts are `dict`s, most frequent first. `value_counts_without_nan` keeps the most frequent values and
  `value_counts_index_sorted` the smallest and largest. Totals stay exact.
- Correlation matrices, samples and duplicates are `pl.DataFrame`s. Time series are frames with `index` and `value`
  columns.
- The augmented Dickey-Fuller test, ACF and PACF are computed with numpy (same results as statsmodels).
- Strings are parsed as dates against a fixed list of formats.
- In time-series mode, `sortby` is the time index (Polars frames have no index).
- The "constant value" alert no longer shows the value of a redacted (sensitive) variable.
- Describing a report that has no DataFrame raises a `ValueError` instead of profiling an empty frame.
- Settings use pydantic 2 and pydantic-settings instead of the `pydantic.v1` compatibility layer. Numbers are still
  accepted for string settings, and unknown nested keys are still ignored, so configurations written for ydata-profiling
  load.

### Fixed

- `config_default.yaml` and `config_minimal.yaml` quote the `yes`/`no`/`y`/`n`/`t`/`f` keys of `vars.bool.mappings`. YAML
  read them as booleans, so a report configured from these files did not recognise "yes"/"no" columns as booleans.
- The same files set `plot.cat_freq` (they named it `plot.cat_frequency`, which was silently ignored).

### Removed

- The pandas and Spark backends, and pandas, PySpark, phik, seaborn, statsmodels, matplotlib, multimethod, typeguard,
  dacite and requests as dependencies.
- The `phi_k` correlation (phik requires pandas).
- The matplotlib renderer and PNG charts: charts are inline SVG.
- The Great Expectations integration.
- Usage analytics: no request is sent to packages.ydata.ai.
- The `ydata_profiling` compatibility package.

[0.1.0]: https://github.com/andrewadlof/polars-data-profiling/releases/tag/v0.1.0
[unreleased]: https://github.com/andrewadlof/polars-data-profiling/compare/v0.1.0...HEAD
