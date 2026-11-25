from collections.abc import Sequence
from typing import Any

import polars as pl

from polars_data_profiling.report.presentation.core.item_renderer import ItemRenderer


class Sample(ItemRenderer):
    def __init__(
        self,
        name: str,
        sample: pl.DataFrame,
        caption: str | None = None,
        index: Sequence[Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            "sample",
            {"sample": sample, "caption": caption, "index": index},
            name=name,
            **kwargs,
        )

    def __repr__(self) -> str:
        return "Sample"

    def render(self) -> Any:
        raise NotImplementedError()
