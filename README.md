# ICLand

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/lysj-cpu/icland/tests.yml?style=flat-square)](https://github.com/lysj-cpu/icland/actions/workflows/tests.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lysj-cpu/icland?style=flat-square)](https://app.codecov.io/github/lysj-cpu/icland)

## Development Instructions

Clone the project locally:

```shell
$ git clone git@github.com:lysj-cpu/icland.git
$ cd icland
```

### uv (recommended)

The project can be run via [uv](https://docs.astral.sh/uv/):

```shell
$ uv run python -c "import icland; print(icland.__doc__)"
Recreating Google DeepMind's XLand RL environment in JAX.
```

[Ruff](https://docs.astral.sh/ruff/) is used for formatting.

```shell
$ uv run ruff check   # Linting
$ uv run ruff format  # Formatting
```

You can also install [pre-commit](https://pre-commit.com/) hooks to automatically run validation checks when making a commit:

```shell
$ uv run pre-commit install
```

### pip

`uv` is [preferred](https://docs.astral.sh/uv/#highlights) over pip and also installs developer dependencies. However, pip might still be useful for small modifications.

First, we create and activate a [virtual environment](https://docs.python.org/3/library/venv.html):

```shell
$ python -m venv env
$ source env/bin/activate
```


Then install in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs):

```shell
$ pip install -e .
$ python -c "import icland; print(icland.__doc__)"
Recreating Google DeepMind's XLand RL environment in JAX.
```
