from polars_data_profiling.config import Settings
from polars_data_profiling.report.presentation.frequency_table_utils import (
    extreme_obs_table,
    freq_table,
)


def render_common(config: Settings, summary: dict) -> dict:
    n_extreme_obs = config.n_extreme_obs
    n_freq_table_max = config.n_freq_table_max

    sorted_counts = summary.get("value_counts_index_sorted", {})
    template_variables = {
        "freq_table_rows": freq_table(
            freqtable=summary["value_counts_without_nan"],
            n=summary["n"],
            max_number_to_print=n_freq_table_max,
            total=summary.get("count"),
            n_distinct=summary.get("n_distinct"),
        ),
        "firstn_expanded": extreme_obs_table(
            freqtable=sorted_counts,
            number_to_print=n_extreme_obs,
            n=summary["n"],
        ),
        "lastn_expanded": extreme_obs_table(
            freqtable=sorted_counts,
            number_to_print=n_extreme_obs,
            n=summary["n"],
            reverse=True,
        ),
    }

    return template_variables
