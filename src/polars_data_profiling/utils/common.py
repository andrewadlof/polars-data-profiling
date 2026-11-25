"""Common util functions (e.g. missing in Python)."""

import collections.abc
from collections.abc import Mapping
from datetime import datetime, timedelta


def update(d: dict, u: Mapping) -> dict:
    """Recursively update a dict.

    Args:
        d: Dictionary to update.
        u: Dictionary with values to use.

    Returns:
        The merged dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def convert_timestamp_to_datetime(timestamp: int) -> datetime:
    if timestamp >= 0:
        return datetime.fromtimestamp(timestamp)
    return datetime(1970, 1, 1) + timedelta(seconds=int(timestamp))
