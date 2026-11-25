"""Image statistics. Needs the optional ``image`` extra (Pillow, imagehash)."""

from collections import Counter
from pathlib import Path

try:
    import imagehash
    from PIL import ExifTags, Image
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "Image variables need Pillow and imagehash: "
        "pip install 'polars-data-profiling[image]'"
    ) from e

from polars_data_profiling.model.summary_algorithms import (
    ValueCounts,
    named_aggregate_summary,
)


def open_image(path: Path) -> Image.Image | None:
    """

    Args:
        path:

    Returns:

    """
    try:
        return Image.open(path)
    except (OSError, AttributeError):
        return None


def is_image_truncated(image: Image) -> bool:
    """Returns True if the path refers to a truncated image

    Args:
        image:

    Returns:
        True if the image is truncated
    """
    try:
        image.load()
    except (OSError, AttributeError):
        return True
    else:
        return False


def get_image_shape(image: Image) -> tuple[int, int] | None:
    """

    Args:
        image:

    Returns:

    """
    try:
        return image.size
    except (OSError, AttributeError):
        return None


def hash_image(image: Image) -> str | None:
    """

    Args:
        image:

    Returns:

    """
    try:
        return str(imagehash.phash(image))
    except (OSError, AttributeError):
        return None


def decode_byte_exif(exif_val: str | bytes) -> str:
    """Decode byte encodings

    Args:
        exif_val:

    Returns:

    """
    if isinstance(exif_val, str):
        return exif_val
    else:
        return exif_val.decode()


def extract_exif(image: Image) -> dict:
    """

    Args:
        image:

    Returns:

    """
    try:
        exif_data = image._getexif()
        if exif_data is not None:
            exif = {
                ExifTags.TAGS[k]: decode_byte_exif(v)
                for k, v in exif_data.items()
                if k in ExifTags.TAGS
            }
        else:
            exif = {}
    except (AttributeError, OSError):
        # Not all file types (e.g. .gif) have exif information.
        exif = {}

    return exif


def count_duplicate_hashes(image_descriptions: list) -> int:
    counts = Counter(x["hash"] for x in image_descriptions if "hash" in x)
    return sum(counts.values()) - len(counts)


def extract_exif_series(image_exifs: list) -> dict:
    keys: Counter = Counter()
    values: dict = {}
    for image_exif in image_exifs:
        keys.update(image_exif.keys())
        for key, value in image_exif.items():
            values.setdefault(key, Counter())[value] += 1
    series: dict = {"exif_keys": ValueCounts(keys.most_common())}
    for key, counter in values.items():
        series[key] = ValueCounts(counter.most_common())
    return series


def extract_image_information(
    path: Path, exif: bool = False, hash: bool = False
) -> dict:
    """Everything needed from one image, opened once."""
    information: dict = {}
    image = open_image(path)
    information["opened"] = image is not None
    if image is not None:
        information["truncated"] = is_image_truncated(image)
        if not information["truncated"]:
            information["size"] = image.size
            if exif:
                information["exif"] = extract_exif(image)
            if hash:
                information["hash"] = hash_image(image)
    return information


def image_summary(paths: list, exif: bool = False, hash: bool = False) -> dict:
    """Dimensions, truncation, duplicate hashes and EXIF data of images."""
    info = [extract_image_information(Path(p), exif=exif, hash=hash) for p in paths]
    dimensions = [x["size"] for x in info if "size" in x]
    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    summary = {
        "n_truncated": sum(1 for x in info if x.get("truncated")),
        "image_dimensions": dimensions,
    }
    summary.update(named_aggregate_summary(widths, "width"))
    summary.update(named_aggregate_summary(heights, "height"))
    summary.update(named_aggregate_summary([w * h for w, h in dimensions], "area"))
    if hash:
        summary["n_duplicate_hash"] = count_duplicate_hashes(info)
    if exif:
        exif_series = extract_exif_series([x["exif"] for x in info if "exif" in x])
        summary["exif_keys_counts"] = exif_series["exif_keys"]
        summary["exif_data"] = exif_series
    return summary
