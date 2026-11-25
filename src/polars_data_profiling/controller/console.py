"""The ``polars_data_profiling`` command line interface."""

import argparse
from pathlib import Path
from typing import Any

from polars_data_profiling.profile_report import ProfileReport
from polars_data_profiling.utils.dataframe import read_polars
from polars_data_profiling.version import __version__


def parse_args(args: list[Any] | None = None) -> argparse.Namespace:
    """Parse the command line arguments for the `polars_data_profiling` binary.

    Args:
      args: List of input arguments. (Default value=None).

    Returns:
      Namespace with parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Profile a CSV, Parquet, IPC or JSON file with Polars and write an HTML (or JSON) report."
    )

    # Version
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Console specific
    parser.add_argument(
        "-s",
        "--silent",
        help="Only generate but do not open report",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--minimal",
        help="Minimal configuration for big data sets",
        action="store_true",
    )

    parser.add_argument(
        "-e",
        "--explorative",
        help="Explorative configuration featuring unicode, file and image analysis",
        action="store_true",
    )

    # Config
    parser.add_argument(
        "--pool_size", type=int, default=0, help="Number of CPU cores to use"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Profiling Report",
        help="Title for the report",
    )

    parser.add_argument(
        "--infer_dtypes",
        default=True,
        action="store_true",
        help="Infer variable types from the values (default)",
    )

    parser.add_argument(
        "--no-infer_dtypes",
        dest="infer_dtypes",
        action="store_false",
        help="Use the dtypes the file was read with",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Specify a yaml config file. Have a look at the 'config_default.yaml' as a starting point.",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="CSV, TSV, Parquet, Arrow IPC/Feather, NDJSON or JSON file to profile (read lazily)",
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        help="Output report file. If empty, replaces the input_file's extension with .html and uses that.",
        default=None,
    )

    return parser.parse_args(args)


def main(args: list[Any] | None = None) -> None:
    """Run the `polars_data_profiling` package.

    Args:
      args: Arguments for the programme (Default value=None).
    """

    # Parse the arguments
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)

    input_file = Path(kwargs.pop("input_file"))
    output_file = kwargs.pop("output_file")
    if output_file is None:
        output_file = str(input_file.with_suffix(".html"))

    silent = kwargs.pop("silent")

    # open the file lazily
    df = read_polars(input_file)

    # Generate the profiling report
    p = ProfileReport(
        df,
        **kwargs,
    )
    p.to_file(Path(output_file), silent=silent)
