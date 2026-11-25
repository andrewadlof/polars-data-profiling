# How it works

A profile is a series of lazy Polars queries. The work is organised so that the number of times the source is read does not
grow with the number of columns.

## 1. Preprocessing

The input becomes a `LazyFrame`. NaN in float columns becomes null, so it counts as a missing value. In time-series mode the
rows are sorted by `sortby`.

## 2. Type inference: one pass

Type inference has to answer questions such as "do all non-null values of this string column parse as numbers?" or "how
many distinct values does it have?". `probe_frame` builds the expressions that answer every question the typeset may ask,
for every column, and evaluates them in a single `select`. The answers (`count`, `n_unique`, `num_ok`, `is_bool`, ...) are
attached to each column.

The typeset then walks its relations (a tree of identity relations with inference relations between types, after
[visions](https://github.com/dylan-profiler/visions)) using those facts only. A relation that converts a column returns a new
*expression*, for example the parse of a string column as numbers, rather than converted data.

Parsing dates is expensive (one attempt per supported format), so the first pass only parses the first 1,000 non-empty values
of each string column. Columns whose sample parses get their full date facts in a second pass, which most frames never need.

The inferred types turn the input into a typed `LazyFrame`: one `select` with the conversion expression of every column.

## 3. Variable summaries: batched plans

Each statistic of a variable type is a *plan*: a generator that yields what it needs and receives the results.

```python
def describe_numeric(config, column, summary):
    response = yield Request(
        aggs={"mean": column.expr.mean(), "min": column.expr.min()}
    )
    summary["mean"] = response.values["mean"]
    ...
    response = yield Request(frames={"hist": histogram_query(column, bins)})
```

`run_plans` advances the plans of all columns together. In every round it gathers the requests of every waiting plan,
evaluates all scalar aggregations in one `select` and all queries (value counts, histograms, character and word counts) in
one `collect_all`, then sends each plan its results. Histograms need the minimum and maximum first, so numeric columns take
two rounds. A whole frame is summarised in two or three rounds.

The summarizer maps each type to its plans, and a type runs the plans of its parent type first. A custom summarizer can add
plans (generators, batched like the built-in ones) or plain functions that materialise what they need with
`column.series()`.

## 4. Table-level statistics

Correlations, interactions, missing-value diagrams, samples and duplicates are each one or a few queries:

- Pearson and Spearman coefficients of every pair in one `select`, each on the rows where both values are present.
- Cramér's V from lazy cross-tabulations, collected together.
- The missing-values matrix from a group-by over at most 200 bands of rows, not from the full null mask.
- Duplicates from one group-by over the supported columns.
- Interaction scatter plots from every k-th row, so at most 10,000 points per pair are collected.

## Shared scans

Queries collected together share a single read of the source: the preprocessed and typed frames carry a `cache()` node, so
one `collect_all` reads the source once. Profiling a 1,000,000-row, 15-column LazyFrame reads its source 10 times. Without the
shared cache it reads it 255 times (once per query, each projected to the columns it needs).

The cache holds the columns of a round in memory. For very large Parquet sources, where projected reads are cheap and memory is
the constraint, turn it off and let Polars stream:

```python
ProfileReport(lf, engine={"share_scans": False, "collect_engine": "streaming"})
```

## What still materialises data

Some statistics need the values themselves, in order:

- time series (stationarity, seasonality, ACF/PACF) collect the series of each time-series column.
- file and image columns read the files they point to.
- Kendall's tau collects the numeric columns.

Everything else is aggregated by Polars before it reaches Python.

## Charts

Charts are drawn as inline SVG from small specs (histogram counts, value counts, matrices). There is no matplotlib backend
and no images are embedded.
