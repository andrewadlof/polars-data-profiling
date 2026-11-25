from typing import Any

import polars as pl

from polars_data_profiling.report.presentation.core.item_renderer import ItemRenderer


class CorrelationTable(ItemRenderer):
    def __init__(self, name: str, correlation_matrix: pl.DataFrame, **kwargs):
        super().__init__(
            "correlation_table",
            {"correlation_matrix": correlation_matrix},
            name=name,
            **kwargs,
        )

    def __repr__(self) -> str:
        return "CorrelationTable"

    def render(self) -> Any:
        raise NotImplementedError()
