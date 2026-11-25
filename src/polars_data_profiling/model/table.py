"""Statistics of the whole table."""

from __future__ import annotations

from collections import Counter

from polars_data_profiling.config import Settings


def get_table_stats(config: Settings, n_rows: int, variable_stats: dict) -> dict:
    """General statistics of the frame, from the column descriptions.

    Args:
        config: report settings.
        n_rows: number of rows.
        variable_stats: the description of every column.

    Returns:
        The table statistics.
    """
    n_var = len(variable_stats)
    sizes = [v.get("memory_size") for v in variable_stats.values()]
    memory_size = sum(s for s in sizes if s is not None)
    table_stats = {
        "n": n_rows,
        "n_var": n_var,
        "memory_size": memory_size,
        "record_size": float(memory_size) / n_rows if n_rows > 0 else 0,
        "n_cells_missing": 0,
        "n_vars_with_missing": 0,
        "n_vars_all_missing": 0,
    }
    for summary in variable_stats.values():
        n_missing = summary.get("n_missing", 0)
        if n_missing > 0:
            table_stats["n_vars_with_missing"] += 1
            table_stats["n_cells_missing"] += n_missing
            if n_missing == n_rows:
                table_stats["n_vars_all_missing"] += 1

    table_stats["p_cells_missing"] = (
        table_stats["n_cells_missing"] / (n_rows * n_var)
        if n_rows > 0 and n_var > 0
        else 0
    )
    table_stats["types"] = dict(Counter(v["type"] for v in variable_stats.values()))
    return table_stats
