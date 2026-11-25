"""The package must neither depend on nor import pandas."""

import subprocess
import sys
from importlib.metadata import requires

import pytest


def test_no_pandas_in_requirements():
    for requirement in requires("polars-data-profiling") or []:
        name = requirement.split(";")[0].split("[")[0]
        for sep in "<>=!~ ":
            name = name.split(sep)[0]
        assert name.strip().lower() not in {
            "pandas",
            "pyspark",
            "phik",
            "seaborn",
            "statsmodels",
        }


def test_pandas_not_installed_in_dev_environment():
    """The dev environment resolves the full dependency tree; pandas must not
    be in it (no transitive dependency pulls it in)."""
    with pytest.raises(ImportError):
        import pandas  # noqa: F401


SCRIPT = """
import sys
import numpy as np
import polars as pl
import polars_data_profiling
from polars_data_profiling import ProfileReport

df = pl.DataFrame({
    "a": np.arange(200),
    "b": ["x", "y"] * 100,
    "c": [f"text {i}" for i in range(200)],
    "d": np.where(np.arange(200) % 5 == 0, None, np.arange(200) * 0.5),
})
ProfileReport(df, progress_bar=False, explorative=True).to_html()
ProfileReport(df.lazy(), progress_bar=False, tsmode=True).to_json()
# matplotlib may load through the optional wordcloud extra; it does not need pandas
loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"pandas", "pyspark", "seaborn", "statsmodels", "phik"})
print(",".join(loaded))
"""


def test_profiling_imports_no_pandas():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
