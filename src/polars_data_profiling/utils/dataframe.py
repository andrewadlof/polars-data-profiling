"""Utilities for Polars frames: reading files, hashing, input conversion."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl

COMPRESSIONS = (".bz2", ".gz", ".xz", ".zip", ".zst")


def uncompressed_extension(file_name: Path) -> str:
    """The extension of a file name, ignoring a compression suffix."""
    suffixes = [s.lower() for s in Path(file_name).suffixes]
    if len(suffixes) > 1 and suffixes[-1] in COMPRESSIONS:
        return suffixes[-2]
    return suffixes[-1] if suffixes else ""


def read_polars(file_name: Path | str) -> pl.LazyFrame:
    """Open a data file lazily, choosing the reader from the extension.

    CSV (and ``.tsv``, ``.txt``, ``.data``), Parquet, Arrow IPC / Feather,
    newline-delimited JSON and JSON are supported. Only JSON is read eagerly.
    Files with another extension are read as CSV.
    """
    path = Path(file_name)
    extension = uncompressed_extension(path)
    if extension == ".parquet":
        return pl.scan_parquet(path)
    if extension in (".arrow", ".feather", ".ipc"):
        return pl.scan_ipc(path)
    if extension in (".jsonl", ".ndjson"):
        return pl.scan_ndjson(path)
    if extension == ".json":
        return pl.read_json(path).lazy()
    separator = "\t" if extension == ".tsv" else ","
    return pl.scan_csv(path, separator=separator, infer_schema_length=10_000)


def infer_objects(df: pl.DataFrame) -> pl.DataFrame:
    """Give ``Object`` columns a real dtype when their values allow it.

    Polars builds ``Object`` columns from, e.g., numpy object arrays that mix
    strings and ``None``. Their values are re-read so that such a column
    becomes a ``String`` column (as ``pandas.DataFrame.infer_objects`` would
    do); columns of arbitrary Python objects stay ``Object``.
    """
    objects = [name for name, dtype in df.schema.items() if dtype == pl.Object]
    if not objects:
        return df
    converted = []
    for name in objects:
        try:
            series = pl.Series(name, df.get_column(name).to_list(), strict=False)
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            continue
        if (
            series.dtype not in (pl.Object, pl.Null)
            or df.get_column(name).null_count() == df.height
        ):
            converted.append(series)
    return df.with_columns(converted) if converted else df


def to_lazyframe(df: Any) -> pl.LazyFrame:
    """Accept a Polars frame or anything Polars can build a frame from.

    pyarrow tables and record batches, objects implementing the Arrow PyCapsule
    stream interface (DuckDB relations, ...) and dicts of columns are converted
    without going through pandas.
    """
    if df is None:
        raise TypeError("Expected a polars DataFrame or LazyFrame, got None")
    if isinstance(df, pl.LazyFrame):
        objects = [n for n, t in df.collect_schema().items() if t == pl.Object]
        if not objects:
            return df
        # Object columns only exist in memory (no scan produces them), so
        # collecting them alone does not read a file.
        inferred = infer_objects(df.select(objects).collect())
        return df.with_columns(pl.lit(s) for s in inferred.get_columns())
    if isinstance(df, pl.DataFrame):
        return infer_objects(df).lazy()
    module = type(df).__module__.split(".")[0]
    if module == "pandas":
        raise TypeError(
            "polars-data-profiling profiles Polars frames. Convert the pandas "
            "DataFrame first, e.g. `pl.from_pandas(df)`."
        )
    if module == "pyarrow":
        converted = pl.from_arrow(df)
        if isinstance(converted, pl.Series):
            converted = converted.to_frame()
        return converted.lazy()
    try:
        return pl.DataFrame(df).lazy()
    except Exception as e:
        raise TypeError(
            "Expected a polars DataFrame or LazyFrame (or an Arrow-compatible "
            f"object), got {type(df)!r}"
        ) from e


HASH_PREFIX = "pl1@"


def hash_dataframe(df: pl.LazyFrame | pl.DataFrame) -> str:
    """Hash a frame's schema and contents (computed lazily).

    Polars' row hashes are not stable across Polars versions, so neither is
    this hash.
    """
    lf = df.lazy()
    schema = lf.collect_schema()
    if len(schema) == 0:
        content = "0"
    else:
        value = (
            lf.select(pl.struct(pl.all()).hash(seed=0).sum().alias("h"))
            .collect()
            .item()
        )
        content = str(value)
    payload = f"{list(schema.items())}\n{content}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def sort_column_names(dct: dict, sort: str | None) -> dict:
    if sort is None:
        return dct

    sort = sort.lower()
    if sort.startswith("asc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold()))
    elif sort.startswith("desc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold(), reverse=True))
    else:
        raise ValueError('"sort" should be "ascending", "descending" or None.')
    return dct
