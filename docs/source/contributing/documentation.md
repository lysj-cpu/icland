# Writing Documentation

The documentation is found in `docs/source` and can be written in either Markdown ([MyST](https://mystmd.org/)) or [reStructuredText](https://www.writethedocs.org/guide/writing/reStructuredText/).

## Building Locally

Install the docs dependencies:

```shell
$ uv sync --extra docs
```

Run the following from the root directory to build the docs:

```shell
$ uv run sphinx-build -v docs/source docs/_build/html
```

Then, use a browser to open the html files in `docs/_build/html/`.

## Adding a New Page

[`docs/source/index.md`](https://github.com/lysj-cpu/icland/blob/main/docs/source/index.md) contains a "table of contents tree" ([toctree](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-toctree)) that defines which pages are included in the top level of the sidebar.

When writing a new page, be sure to include it in a toctree!

## API Reference

[API documentation](../reference/library) is built based on [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) found within the code base. This is done using [sphinx.ext.autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html).

:::{tip}
See the [icland.init docs](../reference/_autosummary/icland.init) and the source code buttons on its page for good examples on how to write the docstrings effectively.
:::
