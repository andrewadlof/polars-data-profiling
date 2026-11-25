from typing import Any

from polars_data_profiling.config import Style
from polars_data_profiling.model.alerts import Alert
from polars_data_profiling.report.presentation.core.item_renderer import ItemRenderer


class Alerts(ItemRenderer):
    def __init__(
        self, alerts: list[Alert] | dict[str, list[Alert]], style: Style, **kwargs
    ):
        super().__init__("alerts", {"alerts": alerts, "style": style}, **kwargs)

    def __repr__(self):
        return "Alerts"

    def render(self) -> Any:
        raise NotImplementedError()
