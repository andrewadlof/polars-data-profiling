from typing import Any

import polars as pl

from polars_data_profiling.report.presentation.core.item_renderer import ItemRenderer


class Duplicate(ItemRenderer):
    def __init__(self, name: str, duplicate: pl.DataFrame, **kwargs):
        super().__init__("duplicate", {"duplicate": duplicate}, name=name, **kwargs)

    def __repr__(self) -> str:
        return "Duplicate"

    def render(self) -> Any:
        raise NotImplementedError()
