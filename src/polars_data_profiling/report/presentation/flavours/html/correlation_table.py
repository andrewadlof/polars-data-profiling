from polars_data_profiling.report.presentation.core.correlation_table import (
    CorrelationTable,
)
from polars_data_profiling.report.presentation.flavours.html import templates
from polars_data_profiling.report.presentation.flavours.html.frame_table import (
    frame_to_html,
)


class HTMLCorrelationTable(CorrelationTable):
    def render(self) -> str:
        matrix = self.content["correlation_matrix"]
        correlation_matrix_html = frame_to_html(
            matrix,
            classes="correlation-table table table-striped",
            index=matrix.columns,
            float_format="{:.3f}".format,
        )
        return templates.template("correlation_table.html").render(
            **self.content, correlation_matrix_html=correlation_matrix_html
        )
