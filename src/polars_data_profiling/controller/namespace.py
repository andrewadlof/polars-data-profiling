"""The ``profile`` namespace on Polars frames.

Importing ``polars_data_profiling`` registers it, so that::

    df.profile.report(title="My data")          # pl.DataFrame
    pl.scan_parquet("big.parquet").profile.report(minimal=True)  # pl.LazyFrame

return a :class:`~polars_data_profiling.ProfileReport`.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_data_profiling.profile_report import ProfileReport


class _ProfileNamespace:
    def __init__(self, frame: pl.DataFrame | pl.LazyFrame) -> None:
        self._frame = frame

    def report(self, **kwargs: Any) -> ProfileReport:
        """Profile this frame.

        Args:
            **kwargs: any argument of :class:`~polars_data_profiling.ProfileReport`
                (``minimal``, ``title``, ``tsmode``, ...).

        Returns:
            The (lazily computed) report.
        """
        return ProfileReport(self._frame, **kwargs)


pl.api.register_dataframe_namespace("profile")(_ProfileNamespace)
pl.api.register_lazyframe_namespace("profile")(_ProfileNamespace)
