"""Reusable Polars expressions and dtype predicates."""

from __future__ import annotations

import os
import string

import polars as pl

# ---------------------------------------------------------------------------
# dtype classes
# ---------------------------------------------------------------------------


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Integers, floats and decimals (not booleans)."""
    return dtype.is_numeric()


def is_float_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_float()


def is_string_dtype(dtype: pl.DataType) -> bool:
    return dtype == pl.String


def is_categorical_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Categorical | pl.Enum)


def is_temporal_dtype(dtype: pl.DataType) -> bool:
    """Dates and datetimes. Times and durations are not profiled as dates."""
    return isinstance(dtype, pl.Datetime) or dtype == pl.Date


def is_hashable_dtype(dtype: pl.DataType) -> bool:
    """Whether values can be counted (and used as keys of the value counts).

    Nested values (lists, arrays, structs) come back to Python as lists and
    dicts, which cannot key a dict; like pandas, they are not counted.
    """
    return dtype != pl.Object and not dtype.is_nested()


def is_orderable_dtype(dtype: pl.DataType) -> bool:
    """Whether distinct values can be sorted (for the extreme values tables)."""
    return (
        dtype.is_numeric()
        or dtype in (pl.String, pl.Boolean, pl.Date, pl.Time)
        or isinstance(dtype, pl.Datetime | pl.Duration | pl.Categorical | pl.Enum)
    )


# ---------------------------------------------------------------------------
# Parsing strings
# ---------------------------------------------------------------------------


def parse_numeric(expr: pl.Expr) -> pl.Expr:
    """Strings as Float64; values that are not numbers become null.

    Like ``pandas.to_numeric(errors="coerce")``: surrounding whitespace is
    ignored, ``"nan"`` is a missing value and ``"inf"`` is infinity.
    """
    return expr.str.strip_chars().cast(pl.Float64, strict=False).fill_nan(None)


# Formats tried, in order, by :func:`parse_datetime`. Polars infers a single
# format per column from its first value; trying explicit formats one after the
# other parses columns that mix formats, as pandas' ``format="mixed"`` does.
# Month-first comes before day-first, as in pandas.
NAIVE_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%b %d %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
)
TZ_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%d %H:%M:%S%.f%z",
)


def parse_datetime(expr: pl.Expr) -> pl.Expr:
    """Strings as naive ``Datetime("us")``; values that are not dates become null.

    Offsets (``+02:00``, ``Z``) are converted to UTC.
    """
    stripped = expr.str.strip_chars()
    zulu = stripped.str.replace(r"Z$", "+00:00")
    attempts = [
        stripped.str.to_datetime(fmt, strict=False, time_unit="us")
        for fmt in NAIVE_DATETIME_FORMATS
    ]
    attempts += [
        zulu.str.to_datetime(
            fmt, strict=False, time_unit="us", time_zone="UTC"
        ).dt.replace_time_zone(None)
        for fmt in TZ_DATETIME_FORMATS
    ]
    return pl.coalesce(attempts)


def parse_bool(expr: pl.Expr, mapping: dict[str, bool]) -> pl.Expr:
    """Strings as booleans through a case-insensitive mapping."""
    return expr.str.to_lowercase().replace_strict(
        mapping, default=None, return_dtype=pl.Boolean
    )


# RFC 3986, appendix B: scheme, netloc, path, query and fragment.
URL_PATTERN = r"^(?:([^:/?#]+):)?(?://([^/?#]*))?([^?#]*)(?:\?([^#]*))?(?:#(.*))?$"
# What ``urllib.parse.urlparse`` needs to report both a scheme and a netloc.
URL_LIKE_PATTERN = r"^[A-Za-z][A-Za-z0-9+.\-]*://[^/?#\s]+"

if os.name == "nt":
    ABSOLUTE_PATH_PATTERN = r"^(?:[A-Za-z]:[\\/]|[\\/]{2}[^\\/]+[\\/][^\\/]+)"
else:
    ABSOLUTE_PATH_PATTERN = r"^/"

WORD_STRIP_CHARACTERS = string.punctuation + string.whitespace


# ---------------------------------------------------------------------------
# Memory size
# ---------------------------------------------------------------------------


def estimated_size(
    expr: pl.Expr, dtype: pl.DataType, n: pl.Expr | None = None
) -> pl.Expr | None:
    """An expression for the number of bytes a column takes in memory.

    Computed from the data, so it works on a LazyFrame that is never
    materialised. Fixed-width types count their width, strings and binaries
    their bytes plus 8-byte offsets, lists their offsets plus their values,
    structs the sum of their fields, and every column one validity bit per
    row. Returns ``None`` for object columns.

    Args:
        expr: the column.
        dtype: its dtype.
        n: its number of values (default: the frame's length).
    """
    n = pl.len() if n is None else n
    validity = (n + 7) // 8
    if dtype == pl.String:
        return expr.str.len_bytes().sum().fill_null(0) + n * 8 + validity
    if dtype == pl.Binary:
        return expr.bin.size().sum().fill_null(0) + n * 8 + validity
    if isinstance(dtype, pl.Categorical | pl.Enum):
        # Physical codes; the (small) dictionary is not counted.
        return n * 4 + validity
    if dtype == pl.Boolean:
        return validity * 2
    if dtype == pl.Null:
        return pl.lit(0, dtype=pl.Int64)
    if isinstance(dtype, pl.List | pl.Array):
        values = expr.explode()
        inner = estimated_size(values, dtype.inner, values.len())  # type: ignore[union-attr]
        if inner is None:
            return None
        offsets = n * 8 if isinstance(dtype, pl.List) else pl.lit(0)
        return inner + offsets + validity
    if isinstance(dtype, pl.Struct):
        sizes = [
            estimated_size(expr.struct.field(f.name), f.dtype, n) for f in dtype.fields
        ]
        if any(size is None for size in sizes):
            return None
        return pl.sum_horizontal(sizes) + validity  # type: ignore[arg-type]
    width = _byte_width(dtype)
    if width is not None:
        return n * width + validity
    return None


_BYTE_WIDTHS: dict[type, int] = {
    pl.Int8: 1,
    pl.UInt8: 1,
    pl.Int16: 2,
    pl.UInt16: 2,
    pl.Int32: 4,
    pl.UInt32: 4,
    pl.Float32: 4,
    pl.Date: 4,
    pl.Int64: 8,
    pl.UInt64: 8,
    pl.Float64: 8,
    pl.Datetime: 8,
    pl.Duration: 8,
    pl.Time: 8,
    pl.Decimal: 16,
    pl.Int128: 16,
}


def _byte_width(dtype: pl.DataType) -> int | None:
    base = dtype if isinstance(dtype, type) else type(dtype)
    return _BYTE_WIDTHS.get(base)
