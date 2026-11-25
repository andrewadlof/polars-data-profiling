from polars_data_profiling.config import ImageType
from polars_data_profiling.report.presentation.core import Container, Image


def image_or_empty(
    image: str | None,
    *,
    alt: str,
    image_format: ImageType,
    caption: str | None = None,
    name: str | None = None,
    anchor_id: str | None = None,
) -> Container | Image:
    """
    Add helper to render an image or an empty container when necessary.
    """
    if image is None:
        return Container(
            items=[],
            name=name or f"{alt}_empty",
            anchor_id=anchor_id,
            sequence_type="grid",
        )

    return Image(
        image=image,
        image_format=image_format,
        alt=alt,
        caption=caption,
        name=name,
        anchor_id=anchor_id,
    )
