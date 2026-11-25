"""Polars-native data profiling.

Profile a ``pl.DataFrame`` or ``pl.LazyFrame`` into an HTML or JSON report::

    import polars as pl
    from polars_data_profiling import ProfileReport

    ProfileReport(pl.scan_parquet("data.parquet"), title="Data").to_file("report.html")

Importing the package also registers the ``profile`` namespace on Polars
frames: ``df.profile.report(...)``.
"""

from polars_data_profiling.profile_report import ProfileReport  # isort: skip
from polars_data_profiling.compare_reports import compare
from polars_data_profiling.controller import namespace  # noqa: F401  (registers df.profile)
from polars_data_profiling.version import __version__

__all__ = ["ProfileReport", "__version__", "compare"]
