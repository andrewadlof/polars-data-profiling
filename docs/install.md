# Install

Add it to a project with [uv](https://docs.astral.sh/uv/):

```bash
uv add polars-data-profiling
```

Or with pip:

```bash
pip install polars-data-profiling
```

It needs Python 3.11 or later and Polars 2.0 or later. The package is pure Python, so one wheel works on every platform.

## Optional extras

The core install profiles every variable type except images. Extras add:

| Extra       | Adds                                                        | Packages               |
| ----------- | ----------------------------------------------------------- | ---------------------- |
| `image`     | dimensions, EXIF data and duplicate hashes of image columns | Pillow, imagehash      |
| `wordcloud` | word clouds for text columns (a word-frequency chart without) | wordcloud            |
| `notebook`  | the Jupyter widgets view, `report.to_widgets()`             | ipywidgets             |
| `all`       | all of the above                                            |                        |

```bash
uv add "polars-data-profiling[all]"
```

## No pandas

pandas is not a dependency, direct or transitive, with or without extras. CI checks the resolved dependency tree on every pull
request and fails if pandas appears in it, and the wheel smoke test asserts that profiling a frame never imports it.
