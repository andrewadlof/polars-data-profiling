from abc import ABC

from polars_data_profiling.report.presentation.core.renderable import Renderable


class ItemRenderer(Renderable, ABC):
    def __init__(
        self,
        item_type: str,
        content: dict,
        name: str | None = None,
        anchor_id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(content, name, anchor_id, classes)
        self.item_type = item_type
