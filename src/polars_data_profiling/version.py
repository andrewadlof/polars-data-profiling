"""The installed version of polars-data-profiling."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polars-data-profiling")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.dev0"
