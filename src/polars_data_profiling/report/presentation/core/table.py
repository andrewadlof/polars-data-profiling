from collections.abc import Sequence
from typing import Any

from polars_data_profiling.config import Style
from polars_data_profiling.report.presentation.core.item_renderer import ItemRenderer


class Table(ItemRenderer):
    def __init__(
        self,
        rows: Sequence,
        style: Style,
        name: str | None = None,
        caption: str | None = None,
        **kwargs,
    ):
        super().__init__(
            "table",
            {"rows": rows, "name": name, "caption": caption, "style": style},
            **kwargs,
        )

    def __repr__(self) -> str:
        return "Table"

    def render(self) -> Any:
        raise NotImplementedError()
