# Configuration

Settings come from, in increasing order of precedence:

1. a preset: `minimal=True` or `config_file="my_config.yaml"`
2. a group: `explorative=True`, `sensitive=True`
3. a `Settings` object: `config=Settings(...)`
4. keyword arguments: `title="..."`, `samples={"head": 5}`, ...

```python
from polars_data_profiling import ProfileReport
from polars_data_profiling.config import Settings

ProfileReport(
    df,
    title="Orders",
    correlations={"pearson": {"calculate": True}},
    vars={"cat": {"words": False}},
    samples={"head": 5, "tail": 0},
)
```

Nested settings are merged, so `vars={"cat": {"words": False}}` changes one setting and keeps the others. Setting a section
to `None` turns it off: `samples=None`, `duplicates=None`, `correlations=None`, `interactions=None`, `missing_diagrams=None`.

The defaults are in
[`config_default.yaml`](https://github.com/andrewadlof/polars-data-profiling/blob/main/src/polars_data_profiling/config_default.yaml)
and the minimal preset in
[`config_minimal.yaml`](https://github.com/andrewadlof/polars-data-profiling/blob/main/src/polars_data_profiling/config_minimal.yaml).

## Frequently used settings

| Setting                                    | Default  | Meaning                                                              |
| ------------------------------------------ | -------- | -------------------------------------------------------------------- |
| `title`                                    | `Profiling Report` | Title of the report                                        |
| `infer_dtypes`                             | `True`   | Infer types from the values. `False` uses the dtypes as stored       |
| `vars.num.low_categorical_threshold`       | `5`      | Numeric columns with at most this many distinct values are categories |
| `vars.cat.cardinality_threshold`           | `50`     | String columns with more distinct values are text                    |
| `vars.cat.length` / `characters` / `words` | `True`   | Length, Unicode and word statistics of categories                    |
| `vars.text.length` / `characters` / `words` | `True`  | The same for text                                                    |
| `vars.bool.mappings`                       | yes/no, t/f, y/n, true/false | Strings read as booleans                         |
| `vars.url.active`, `vars.path.active`      | `False`  | Detect URL and path columns (on in `explorative`)                    |
| `vars.timeseries.active`                   | `False`  | Time-series mode (`tsmode=True`)                                     |
| `correlations.<name>.calculate`            | `auto` only | `auto`, `pearson`, `spearman`, `kendall`, `cramers`               |
| `interactions.continuous`                  | `True`   | Scatter plots between numeric columns                                |
| `missing_diagrams`                         | all on   | `bar`, `matrix`, `heatmap`                                           |
| `samples.head` / `tail` / `random`         | 10, 10, 0 | Rows shown in the report                                            |
| `duplicates.head`                          | `10`     | Most frequent duplicate rows shown                                   |
| `plot.histogram.bins`                      | `50`     | Histogram bins (`0`: numpy's "auto" rule, capped by `max_bins`)      |
| `html.style.theme`                         | light    | `light`, `dark` or `auto`                                            |
| `html.minify_html`                         | `True`   | Minify the HTML                                                      |

## Query execution

| Setting                 | Default | Meaning                                                                         |
| ----------------------- | ------- | ------------------------------------------------------------------------------- |
| `engine.share_scans`    | `True`  | Queries collected together read the source once (a shared `cache()` node)       |
| `engine.collect_engine` | `auto`  | Polars engine used to collect: `auto`, `in-memory`, `streaming` or `gpu`        |

See [How it works](architecture.md#shared-scans) for the trade-off.

## Environment variables

`Settings` reads environment variables with the `PROFILE_` prefix, for example `PROFILE_TITLE="Nightly"`.
