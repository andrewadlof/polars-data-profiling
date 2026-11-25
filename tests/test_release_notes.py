import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts" / "release_notes.py"
)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0] - 2027-01-01

Faster profiles.

### Added

- A thing.

### Fixed

- A bug.

## [0.1.0] - 2026-11-25

### Added

- The first thing.

[unreleased]: https://example.com/compare/v0.2.0...HEAD
[0.2.0]: https://example.com/v0.2.0
"""


def test_notes_have_intro_install_and_sections():
    notes = release_notes.release_notes(CHANGELOG, "0.2.0")
    assert notes.startswith("Faster profiles.\n\n## Install")
    assert "## Added\n\n- A thing." in notes
    assert "## Fixed\n\n- A bug." in notes
    assert "0.1.0" not in notes and "###" not in notes


def test_section_without_intro_and_last_before_links():
    notes = release_notes.release_notes(CHANGELOG, "0.1.0")
    assert notes.startswith("## Install")
    assert notes.rstrip().endswith("- The first thing.")
    assert "example.com" not in notes


def test_missing_version_fails():
    with pytest.raises(SystemExit, match="no section for version 9.9.9"):
        release_notes.release_notes(CHANGELOG, "9.9.9")


def test_every_released_version_has_notes():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = release_notes.release_notes(changelog, "0.1.0")
    assert "## Added" in notes and "## Removed" in notes
