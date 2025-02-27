# Development Setup


Fork the [GitHub project](https://github.com/lysj-cpu/icland) to your account. Then, run the following with your GitHub handle in place of GITHUB_NAME:

```shell
$ git clone git@github.com:GITHUB_NAME/icland.git
$ cd icland
```

The project can be run via [uv](https://docs.astral.sh/uv/):

<!-- Adding shell formatting means sphinx struggles with the single quote -->
```
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


:::{admonition} Using pip
:class: dropdown

`uv` is [preferred](https://docs.astral.sh/uv/#highlights) over pip and also installs developer dependencies. However, pip might still be useful for small modifications.

First, we create and activate a [virtual environment](https://docs.python.org/3/library/venv.html):

```shell
$ python -m venv env
$ source env/bin/activate
```

Then install in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs):

```
$ pip install -e .
$ python -c "import icland; print(icland.__doc__)"
Recreating Google DeepMind's XLand RL environment in JAX.
```

:::
