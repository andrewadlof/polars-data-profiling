# Development

You need [uv](https://docs.astral.sh/uv/). The `just` task runner is installed into the project's environment, so prefix
recipes with `uv run` (or install `just` globally).

```bash
uv sync --all-extras     # creates .venv with the package, all extras and the dev tools
uv run just              # lists the recipes
uv run just test         # runs the test suite
uv run just check        # lint, format check, the pandas check and the tests
```

## Recipes

| Recipe                 | What it does                                                        |
| ---------------------- | ------------------------------------------------------------------- |
| `just test [args]`     | `pytest`, with any pytest arguments                                 |
| `just test_cov`        | tests with a branch coverage report                                 |
| `just lint`            | `ruff check --fix`                                                  |
| `just format`          | `ruff format`                                                       |
| `just check`           | everything CI checks                                                |
| `just no-pandas`       | fails if pandas is anywhere in the resolved dependency tree         |
| `just precommit`       | all pre-commit hooks through prek                                   |
| `just profile FILE`    | profiles a data file with the command line tool                     |
| `just docs-serve`      | live preview of these docs on http://localhost:8000                 |
| `just build`           | builds the sdist and wheel into `dist/`                             |
| `just bump LEVEL`      | bumps the version (`patch`, `minor`, `major`)                       |
| `just tag`             | tags the current version on `main` and pushes it, which releases    |

## Tests

The suite checks the statistics against numpy and scipy, the ADF test against hard-coded statsmodels results, type
inference on every dtype, how often a profile reads its source (with a counting IO source), that pandas is never
imported, and the HTML, JSON, comparison and command-line outputs.

## Releasing

Releases are cut by CI. Bump the version, add the changes to `CHANGELOG.md` under the new version, merge to `main`, then:

```bash
uv run just tag
```

`.github/workflows/release.yml` builds the sdist and wheel, checks that the tag matches the package version and that the
wheel profiles a frame in a clean environment, then publishes to PyPI through Trusted Publishing. The `pypi` environment
must first be registered as a trusted publisher of the project on PyPI (Publishing, then "Add a new pending publisher" for
the first release), with the repository `andrewadlof/polars-data-profiling` and the workflow `release.yml`.

Run the workflow by hand (`workflow_dispatch`) to rehearse the build without publishing.
