"""Print the GitHub release notes of a version, from CHANGELOG.md.

    python scripts/release_notes.py 0.2.0 > notes.md

The notes are the text under the version's heading (before its first
subsection), an Install section, then the version's subsections (Added,
Changed, Fixed, ...) as level-2 headings. Exits with an error if the version
has no section in the changelog.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INSTALL = """## Install

```bash
uv add polars-data-profiling
```

Documentation: https://andrewadlof.github.io/polars-data-profiling/
"""


def changelog_section(changelog: str, version: str) -> str:
    """The body of ``## [version] - date`` up to the next version or the links."""
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)"
    match = re.search(pattern, changelog, re.S | re.M)
    if match is None or not match.group(1).strip():
        raise SystemExit(f"CHANGELOG.md has no section for version {version}")
    return match.group(1).strip()


def release_notes(changelog: str, version: str) -> str:
    section = changelog_section(changelog, version)
    intro, _, changes = section.partition("\n### ")
    if section.startswith("### "):
        intro, changes = "", section[len("### ") :]
    parts = [intro.strip()] if intro.strip() else []
    parts.append(INSTALL.strip())
    if changes:
        parts.append(("### " + changes).replace("### ", "## ").strip())
    return "\n\n".join(parts) + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: release_notes.py VERSION")
    changelog = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    sys.stdout.write(release_notes(changelog.read_text(encoding="utf-8"), sys.argv[1]))


if __name__ == "__main__":
    main()
