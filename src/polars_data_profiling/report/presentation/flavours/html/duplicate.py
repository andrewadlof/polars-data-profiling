import polars as pl

from polars_data_profiling.report.presentation.core.duplicate import Duplicate
from polars_data_profiling.report.presentation.flavours.html import templates
from polars_data_profiling.report.presentation.flavours.html.frame_table import (
    frame_to_html,
)


def to_html(df: pl.DataFrame) -> str:
    return frame_to_html(
        df,
        classes="duplicate table table-striped",
        empty_message="Dataset does not contain duplicate rows.",
    )


class HTMLDuplicate(Duplicate):
    def render(self) -> str:
        duplicate_html = to_html(self.content["duplicate"])
        return templates.template("duplicate.html").render(
            **self.content, duplicate_html=duplicate_html
        )
