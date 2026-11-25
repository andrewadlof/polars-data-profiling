"""Settings (pydantic 2 and pydantic-settings)."""

import pydantic
import pytest

from polars_data_profiling.config import Settings


def test_defaults_are_not_shared_between_instances():
    a, b = Settings(), Settings()
    a.vars.url.active = True
    a.missing_diagrams["bar"] = False
    a.html.style._labels = ["x", "y"]
    assert b.vars.url.active is False
    assert b.missing_diagrams["bar"] is True
    assert b.html.style._labels == ["_"]


def test_update_merges_nested_settings():
    base = Settings()
    base.vars.url.active = True
    updated = base.update({"vars": {"cat": {"words": False}}})
    assert updated.vars.url.active is True
    assert updated.vars.cat.words is False
    assert updated.vars.cat.characters is True


def test_numbers_are_accepted_for_string_settings():
    assert (
        Settings().update({"dataset": {"copyright_year": 2024}}).dataset.copyright_year
        == "2024"
    )


def test_unknown_settings_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        Settings(not_a_setting=True)


def test_environment_variables(monkeypatch):
    monkeypatch.setenv("PROFILE_TITLE", "From the environment")
    assert Settings().title == "From the environment"


def test_json_roundtrip():
    settings = Settings(title="Round trip")
    settings.correlations["pearson"].calculate = True
    loaded = Settings().update(
        Settings.model_validate_json(settings.model_dump_json()).model_dump()
    )
    assert loaded.title == "Round trip"
    assert loaded.correlations["pearson"].calculate is True


def test_legacy_theme_warns():
    with pytest.warns(FutureWarning, match="deprecated"):
        Settings().update({"html": {"style": {"theme": "flatly"}}})


def _paths(data: dict, prefix: str = "") -> set[str]:
    paths = set()
    for key, value in data.items():
        path = f"{prefix}{key}"
        paths.add(path)
        if isinstance(value, dict) and path not in {
            "vars.bool.mappings",
            "variables.descriptions",
            "correlations",
        }:
            paths |= _paths(value, f"{path}.")
    return paths


@pytest.mark.parametrize("name", ["config_default.yaml", "config_minimal.yaml"])
def test_bundled_config_files_only_hold_real_settings(name):
    import yaml

    from polars_data_profiling.utils.paths import get_config

    data = yaml.safe_load(get_config(name).read_text(encoding="utf-8"))
    known = _paths(Settings().model_dump())
    assert _paths(data) - known == set()
    settings = Settings.from_file(get_config(name))
    assert set(settings.vars.bool.mappings) == {
        "t",
        "f",
        "yes",
        "no",
        "y",
        "n",
        "true",
        "false",
    }


def test_default_config_file_recognises_yes_no_booleans():
    """YAML reads unquoted yes/no keys as booleans: the bundled file quotes them."""
    import polars as pl

    from polars_data_profiling import ProfileReport
    from polars_data_profiling.utils.paths import get_config

    df = pl.DataFrame({"flag": ["yes", "no"] * 50})
    profile = ProfileReport(
        df, config_file=get_config("config_default.yaml"), progress_bar=False
    )
    assert profile.get_description().variables["flag"]["type"] == "Boolean"


def test_settings_from_ydata_profiling_still_load():
    settings = Settings().update({"plot": {"dpi": 800, "image_format": "svg"}})
    assert not hasattr(settings.plot, "dpi")
