import json
import warnings
from dataclasses import asdict
from typing import Any

import polars as pl

from polars_data_profiling.config import Correlation, Settings
from polars_data_profiling.model import BaseDescription
from polars_data_profiling.model.alerts import Alert
from polars_data_profiling.model.description import BaseAnalysis, TimeIndexAnalysis
from polars_data_profiling.model.summary_algorithms import ValueCounts
from polars_data_profiling.profile_report import ProfileReport


def _should_wrap(v1: Any, v2: Any) -> bool:
    # Value counts are leaves: they are paired, not merged key by key
    if isinstance(v1, ValueCounts) or isinstance(v2, ValueCounts):
        return True
    if isinstance(v1, (list, dict)):
        return False

    if isinstance(v1, pl.DataFrame) and isinstance(v2, pl.DataFrame):
        return v1.equals(v2)
    if isinstance(v1, pl.Series) and isinstance(v2, pl.Series):
        return v1.equals(v2)

    try:
        return v1 == v2
    except ValueError:
        return False


def _update_merge_dict(d1: Any, d2: Any) -> dict:
    # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
    # Next unwrap a dict that treats shared keys
    # If two keys have an equal value, we take that value as new value
    # If the values are not equal, we recursively merge them

    return {
        **d1,
        **d2,
        **{
            k: [d1[k], d2[k]]
            if _should_wrap(d1[k], d2[k])
            else _update_merge_mixed(d1[k], d2[k])
            for k in {*d1} & {*d2}
        },
    }


def _update_merge_seq(d1: Any, d2: Any) -> list | tuple:
    # This case happens when values are merged
    # It bundle values in a list, making sure
    # to flatten them if they are already lists

    if isinstance(d1, list) and isinstance(d2, list):
        # This is the tuple for alerts
        return d1, d2
    elif isinstance(d1, tuple) and isinstance(d2, list):
        return (*d1, d2)
    else:
        return [
            *(d1 if isinstance(d1, list) else [d1]),
            *(d2 if isinstance(d2, list) else [d2]),
        ]


def _update_merge_mixed(d1: Any, d2: Any) -> dict | list | tuple:
    if (
        isinstance(d1, dict)
        and isinstance(d2, dict)
        and not isinstance(d1, ValueCounts)
        and not isinstance(d2, ValueCounts)
    ):
        return _update_merge_dict(d1, d2)
    else:
        return _update_merge_seq(d1, d2)


def _update_merge(d1: dict | None, d2: dict) -> dict:
    # For convenience in the loop, allow d1 to be empty initially
    if d1 is None:
        return d2

    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError(
            "Both arguments need to be of type dictionary (ProfileReport.description_set)"
        )

    return _update_merge_dict(d1, d2)


def _placeholders(reports: list[BaseDescription]) -> None:
    """Generates placeholders in the dataset descriptions where needed"""

    keys = {v for r in reports for v in r.scatter}
    type_keys = {v for r in reports for v in r.table["types"]}
    for report in reports:
        # Interactions
        for k1 in keys:
            for k2 in keys:
                if k1 not in report.scatter:
                    report.scatter[k1] = {}
                if k2 not in report.scatter[k1]:
                    report.scatter[k1][k2] = ""

        # Types
        for type_key in type_keys:
            if type_key not in report.table["types"]:
                report.table["types"][type_key] = 0


def _update_titles(reports: list[ProfileReport]) -> None:
    """Redefine title of reports with the default one."""
    for idx, report in enumerate(reports):
        if report.config.title == Settings().title:
            report.config.title = f"Dataset {chr(65 + idx)}"


def _compare_title(titles: list[str]) -> str:
    if all(titles[0] == title for title in titles[1:]):
        return titles[0]
    else:
        title = ", ".join(titles[:-1])
        return f"<em>Comparing</em> {title} <em>and</em> {titles[-1]}"


def _compare_profile_report_preprocess(
    reports: list[ProfileReport],
    config: Settings | None = None,
) -> tuple[list[str], list[BaseDescription]]:
    # Use titles as labels
    labels = [report.config.title for report in reports]

    # Use color per report if not custom set
    if config is None:
        if len(reports[0].config.html.style.primary_colors) > 1:
            for idx, report in enumerate(reports):
                report.config.html.style.primary_colors = [
                    report.config.html.style.primary_colors[idx]
                ]
    else:
        if len(config.html.style.primary_colors) > 1:
            for report in reports:
                report.config.html.style.primary_colors = (
                    config.html.style.primary_colors
                )

    # Obtain description sets
    descriptions = [report.get_description() for report in reports]
    for label, description in zip(labels, descriptions):
        description.analysis.title = label

    return labels, descriptions


def _compare_dataset_description_preprocess(
    reports: list[BaseDescription],
) -> tuple[list[str], list[BaseDescription]]:
    labels = [report.analysis.title for report in reports]
    return labels, reports


def validate_reports(
    reports: list[ProfileReport] | list[BaseDescription], configs: list[dict]
) -> None:
    """Validate if the reports are comparable.

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
    """
    if len(reports) < 2:
        raise ValueError("At least two reports are required for this comparison")

    if len(reports) > 2:
        warnings.warn(
            "Comparison of more than two reports is not supported. "
            "Reports may be produced, but may yield unexpected formatting.",
            stacklevel=2,
        )

    report_types = [c.vars.timeseries.active for c in configs]  # type: ignore
    if all(report_types) != any(report_types):
        raise ValueError(
            "Comparison between timeseries and tabular reports is not supported."
        )

    if isinstance(reports[0], ProfileReport):
        is_df_available = [r.df is not None for r in reports]  # type: ignore
        if not all(is_df_available):
            raise ValueError("Reports where not initialized with a DataFrame.")

    if isinstance(reports[0], ProfileReport):
        features = [set(r.df.collect_schema().names()) for r in reports]  # type: ignore
    else:
        features = [set(r.variables.keys()) for r in reports]  # type: ignore

    if not all(features[0] == x for x in features):
        warnings.warn(
            "The datasets being profiled have a different set of columns. "
            "Only the left side profile will be calculated.",
            stacklevel=2,
        )


def _apply_config(description: BaseDescription, config: Settings) -> BaseDescription:
    """Apply the configuration for visualilzation purposes.

    This handles the cases in which the report description
    was computed prior to comparison with a different config

    Args:
        description: report summary
        config: the settings object for the ProfileReport

    Returns:
        the updated description
    """
    description.missing = {
        k: v for k, v in description.missing.items() if config.missing_diagrams[k]
    }

    description.correlations = {
        k: v
        for k, v in description.correlations.items()
        if config.correlations.get(k, Correlation(calculate=False).calculate)
    }

    samples = [config.samples.head, config.samples.tail, config.samples.random]
    samples = [s > 0 for s in samples]
    description.sample = description.sample if any(samples) else []
    description.duplicates = (
        description.duplicates if config.duplicates.head > 0 else None
    )
    description.scatter = description.scatter if config.interactions.continuous else {}

    return description


def _is_alert_present(alert: Alert, alert_list: list) -> bool:
    return any(
        a.column_name == alert.column_name and a.alert_type == alert.alert_type
        for a in alert_list
    )


def _create_placehoder_alerts(report_alerts: tuple) -> tuple:
    from copy import copy

    fixed: list = [[] for _ in report_alerts]
    for idx, alerts in enumerate(report_alerts):
        for alert in alerts:
            fixed[idx].append(alert)
            for i, fix in enumerate(fixed):
                if i == idx:
                    continue
                if not _is_alert_present(alert, report_alerts[i]):
                    empty_alert = copy(alert)
                    empty_alert._is_empty = True
                    fix.append(empty_alert)
    return tuple(fixed)


def compare(
    reports: list[ProfileReport] | list[BaseDescription],
    config: Settings | None = None,
    compute: bool = False,
) -> ProfileReport:
    """
    Compare Profile reports

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
        config: the settings object for the merged ProfileReport
        compute: recompute the profile report using config or the left report config
                 recommended in cases where the reports were created using different settings

    """
    if len(reports) == 0:
        raise ValueError("No reports available for comparison.")

    report_dtypes = [type(r) for r in reports]
    if len(set(report_dtypes)) > 1:
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )

    if isinstance(reports[0], ProfileReport):
        all_configs = [r.config for r in reports]  # type: ignore
    else:
        configs_str = [
            json.loads(r.package["polars_data_profiling_config"])
            for r in reports  # type: ignore
        ]
        all_configs = []
        for c_str in configs_str:
            c_setting = Settings()
            c_setting = c_setting.update(c_str)
            all_configs.append(c_setting)

    validate_reports(reports=reports, configs=all_configs)

    if isinstance(reports[0], ProfileReport):
        base_features = reports[0].df.collect_schema().names()  # type: ignore
        for report in reports[1:]:
            columns = set(report.df.collect_schema().names())  # type: ignore
            cols_2_compare = [col for col in base_features if col in columns]
            report.df = report.df.select(cols_2_compare)  # type: ignore
        reports = [r for r in reports if len(r.df.collect_schema()) > 0]  # type: ignore
        if len(reports) == 1:
            return reports[0]  # type: ignore
    else:
        base_features = list(reports[0].variables.keys())
        non_empty_reports = 0
        for report in reports[1:]:
            cols_2_compare = [
                col
                for col in base_features
                if col in list(report.variables.keys())  # type: ignore
            ]
            if len(cols_2_compare) > 0:
                non_empty_reports += 1
        if non_empty_reports == 0:
            profile = ProfileReport(None, config=all_configs[0])
            profile._description_set = reports[0]
            return profile

    _config = None
    if config is None:
        _config = all_configs[0].model_copy(deep=True)
    else:
        _config = config.model_copy(deep=True)
        if isinstance(reports[0], ProfileReport):
            for report in reports:
                tsmode = report.config.vars.timeseries.active  # type: ignore
                title = report.config.title  # type: ignore
                report.config = config.model_copy(deep=True)  # type: ignore
                report.config.title = title  # type: ignore
                report.config.vars.timeseries.active = tsmode  # type: ignore
                if compute:
                    report._description_set = None  # type: ignore

    if all(isinstance(report, ProfileReport) for report in reports):
        # Type ignore is needed as mypy does not pick up on the type narrowing
        # Consider using TypeGuard (3.10): https://docs.python.org/3/library/typing.html#typing.TypeGuard
        _update_titles(reports)  # type: ignore
        labels, descriptions = _compare_profile_report_preprocess(reports, _config)  # type: ignore
    elif all(isinstance(report, BaseDescription) for report in reports):
        labels, descriptions = _compare_dataset_description_preprocess(reports)  # type: ignore
    else:
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )

    _config.html.style._labels = labels

    _placeholders(descriptions)

    descriptions_dict = [asdict(_apply_config(d, _config)) for d in descriptions]

    res: dict = _update_merge(None, descriptions_dict[0])
    for r in descriptions_dict[1:]:
        res = _update_merge(res, r)

    res["analysis"]["title"] = _compare_title(res["analysis"]["title"])
    res["alerts"] = _create_placehoder_alerts(res["alerts"])
    if not any(res["time_index_analysis"]):
        res["time_index_analysis"] = None
    profile = ProfileReport(None, config=_config)
    profile._description_set = _description_from_dict(res)
    return profile


def _description_from_dict(data: dict) -> BaseDescription:
    """Rebuild a (merged) description from its dict form."""
    data = dict(data)
    analysis = data["analysis"]
    data["analysis"] = BaseAnalysis(
        analysis["title"], analysis["date_start"], analysis["date_end"]
    )
    if data.get("time_index_analysis"):
        data["time_index_analysis"] = TimeIndexAnalysis(**data["time_index_analysis"])
    return BaseDescription(**data)
