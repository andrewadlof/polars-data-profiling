"""Native chart rendering: data-driven specs rendered as inline SVG.

Every chart of the report is a spec (see ``specs``) built from the summary by
``builders`` and rendered by ``svg.render`` as inline ``<svg>`` markup: no
images, no matplotlib.
"""

from polars_data_profiling.visualisation.charts.svg import render


def is_inline_svg(image: str) -> bool:
    """True for markup produced by the native renderer."""
    return isinstance(image, str) and image.lstrip().startswith("<svg")


__all__ = ["render", "is_inline_svg"]
