"""HTML tables of Polars frames (samples, duplicates, correlation tables).

Produces the same markup as ``pandas.DataFrame.to_html`` (a ``dataframe``
table with a header row and an index column), which the report's stylesheet
targets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

import polars as pl


def _cell(value: Any, float_format: Callable[[float], str] | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if float_format is not None:
            return float_format(value)
    return escape(str(value))


def frame_to_html(
    df: pl.DataFrame,
    classes: str = "",
    index: Sequence[Any] | None = None,
    float_format: Callable[[float], str] | None = None,
    empty_message: str | None = None,
) -> str:
    """Render a frame as an HTML table.

    Args:
        df: the frame.
        classes: CSS classes added to the table.
        index: labels of the rows (default: 0, 1, ...).
        float_format: formats float cells.
        empty_message: a row shown when the frame has no rows.
    """
    labels = list(range(df.height)) if index is None else list(index)
    class_attr = " ".join(c for c in ("dataframe", classes) if c)
    parts = [
        f'<table border="1" class="{escape(class_attr)}">',
        '<thead><tr style="text-align: right;"><th></th>',
    ]
    parts += [f"<th>{escape(str(name))}</th>" for name in df.columns]
    parts.append("</tr></thead><tbody>")
    if df.height == 0 and empty_message:
        parts.append(
            f'<tr><td colspan="{df.width + 1}">{escape(empty_message)}</td></tr>'
        )
    for label, row in zip(labels, df.iter_rows()):
        parts.append(f"<tr><th>{escape(str(label))}</th>")
        parts += [f"<td>{_cell(v, float_format)}</td>" for v in row]
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)
