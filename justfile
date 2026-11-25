# Define project root
root := justfile_directory()

# List available recipes
default:
    @just --list

# ----------------------------------------------------------------------------
# dev
# ----------------------------------------------------------------------------

# Install the package and the dev tools into .venv
[group('dev')]
sync:
    uv sync --all-extras

# Run the test suite
[group('dev')]
test *args:
    uv run pytest {{args}}

# Run the test suite with coverage (branch + term-missing report)
[group('dev')]
test_cov *args:
    uv run pytest --cov=polars_data_profiling --cov-report=term-missing --cov-branch {{args}}

# Lint with ruff and auto-fix what it safely can
[group('dev')]
lint *args:
    uv run ruff check --fix {{args}}

# Format the codebase with ruff
[group('dev')]
format:
    uv run ruff format

# Lint (no fixes) + format check + pandas check + tests
[group('dev')]
check: no-pandas
    uv run ruff check
    uv run ruff format --check
    uv run pytest

# Fail if pandas is anywhere in the resolved dependency tree
[group('dev')]
no-pandas:
    #!/usr/bin/env bash
    set -euo pipefail
    if uv tree --all-groups --no-dedupe | grep -Ei '(^|[ ─])pandas( |$)'; then
        echo "error: pandas is in the dependency tree" >&2
        exit 1
    fi
    echo "ok: no pandas in the dependency tree"

# Run all pre-commit hooks via prek (ruff, taplo, mdformat and file checks)
[group('dev')]
precommit *args:
    uv run prek run --all-files --show-diff-on-failure {{args}}

# Profile a data file from the command line: `just profile data.parquet`
[group('dev')]
profile input output="":
    uv run polars_data_profiling --silent {{input}} {{output}}

# ----------------------------------------------------------------------------
# docs
# ----------------------------------------------------------------------------

# Live-preview the docs at http://localhost:8000
[group('docs')]
docs-serve:
    uv run --only-group docs zensical serve

# Build the static docs site into ./site
[group('docs')]
docs-build:
    uv run --only-group docs zensical build --clean

# ----------------------------------------------------------------------------
# release
#
# Releases are cut by CI: push a v<version> tag and .github/workflows/release.yml
# builds the sdist and wheel and publishes them through PyPI Trusted Publishing.
# Published versions are immutable.
# ----------------------------------------------------------------------------

# Build the sdist and wheel into ./dist
[group('release')]
build:
    uv build

# Print the current package version
[group('release')]
version:
    uv version --short

# Bump the version: `just bump patch` | minor | major
[group('release')]
bump level="patch":
    uv version --bump {{level}}

# Guard: refuse to tag a release from anywhere but `main`
[private]
_require-main:
    #!/usr/bin/env bash
    set -euo pipefail
    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$branch" != "main" ]; then
        echo "error: release tags are cut from 'main', but you are on '$branch'." >&2
        exit 1
    fi

# Tag the current version and push it, which triggers the release workflow
[group('release')]
tag: _require-main
    #!/usr/bin/env bash
    set -euo pipefail
    version="$(uv version --short)"
    echo "Tagging v$version"
    git tag "v$version"
    git push origin "v$version"
