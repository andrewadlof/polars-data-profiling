"""Rows of the frequency and extreme-value tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _frequency_table(
    freqtable: Mapping[Any, int],
    n: int,
    max_number_to_print: int,
    total: int | None = None,
    n_distinct: int | None = None,
) -> list[dict[str, Any]]:
    """Rows of one frequency table.

    Args:
        freqtable: value -> count, most frequent first. May hold only the most
            frequent values; ``total`` and ``n_distinct`` then give the exact
            totals.
        n: number of rows (values plus missing values).
        max_number_to_print: number of values to show.
        total: number of non-missing values (default: the sum of the counts).
        n_distinct: number of distinct values (default: the number of entries).
    """
    if len(freqtable) == 0:
        return []
    counts = list(freqtable.items())
    total = sum(freqtable.values()) if total is None else total
    n_distinct = len(counts) if n_distinct is None else n_distinct

    max_number_to_print = min(max_number_to_print, n)
    shown = counts[:max_number_to_print]
    freq_other = total - sum(count for _, count in shown)
    if max_number_to_print < n_distinct:
        min_freq = (
            counts[max_number_to_print][1] if len(counts) > max_number_to_print else 0
        )
    else:
        freq_other = 0
        min_freq = 0

    freq_missing = n - total
    max_freq = max(counts[0][1], freq_other, freq_missing)
    if max_freq == 0:
        return []

    rows = [
        {
            "label": label,
            "width": freq / max_freq,
            "count": freq,
            "percentage": float(freq) / n,
            "n": n,
            "extra_class": "",
        }
        for label, freq in shown
    ]

    if freq_other > min_freq:
        other_count = str(n_distinct - len(shown))
        rows.append(
            {
                "label": f"Other values ({other_count})",
                "width": freq_other / max_freq,
                "count": freq_other,
                "percentage": min(float(freq_other) / n, 1.0),
                "n": n,
                "extra_class": "other",
            }
        )

    if freq_missing > min_freq:
        rows.append(
            {
                "label": "(Missing)",
                "width": freq_missing / max_freq,
                "count": freq_missing,
                "percentage": float(freq_missing) / n,
                "n": n,
                "extra_class": "missing",
            }
        )

    return rows


def _listify(value: Any, length: int) -> list:
    return value if isinstance(value, list) else [value] * length


def freq_table(
    freqtable: Mapping | list[Mapping],
    n: int | list[int],
    max_number_to_print: int,
    total: int | list[int] | None = None,
    n_distinct: int | list[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """Render the rows for a frequency table (value, count).

    Args:
        freqtable: value -> count, most frequent first (a list when comparing).
        n: The total number of rows.
        max_number_to_print: The maximum number of observations to print.
        total: exact number of non-missing values, when ``freqtable`` is
            truncated.
        n_distinct: exact number of distinct values, when ``freqtable`` is
            truncated.

    Returns:
        The rows of each frequency table.
    """
    if isinstance(freqtable, list) and isinstance(n, list):
        k = len(freqtable)
        return [
            _frequency_table(v, n2, max_number_to_print, t, d)
            for v, n2, t, d in zip(
                freqtable, n, _listify(total, k), _listify(n_distinct, k)
            )
        ]
    return [_frequency_table(freqtable, n, max_number_to_print, total, n_distinct)]  # type: ignore[arg-type]


def _extreme_obs_table(
    freqtable: list[tuple[Any, int]], number_to_print: int, n: int
) -> list[dict[str, Any]]:
    obs_to_print = freqtable[:number_to_print]
    max_freq = max((count for _, count in obs_to_print), default=0)
    return [
        {
            "label": label,
            "width": freq / max_freq if max_freq != 0 else 0,
            "count": freq,
            "percentage": float(freq) / n,
            "extra_class": "",
            "n": n,
        }
        for label, freq in obs_to_print
    ]


def _items(freqtable: Mapping | list, reverse: bool) -> list[tuple[Any, int]]:
    items = (
        list(freqtable.items()) if isinstance(freqtable, Mapping) else list(freqtable)
    )
    return items[::-1] if reverse else items


def extreme_obs_table(
    freqtable: Mapping | list[Mapping],
    number_to_print: int,
    n: int | list[int],
    reverse: bool = False,
) -> list[list[dict[str, Any]]]:
    """Similar to the frequency table, for extreme observations.

    Args:
        freqtable: value -> count, sorted by value (a list when comparing).
        number_to_print: The number of observations to print.
        n: The total number of observations.
        reverse: show the largest values first.
    """
    if isinstance(freqtable, list) and isinstance(n, list):
        return [
            _extreme_obs_table(_items(v, reverse), number_to_print, n1)
            for v, n1 in zip(freqtable, n)
        ]
    return [_extreme_obs_table(_items(freqtable, reverse), number_to_print, n)]  # type: ignore[arg-type]
