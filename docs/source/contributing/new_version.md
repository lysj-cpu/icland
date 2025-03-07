# Publishing a New Version

Congratulations on adding new features and fixing bugs! Here's how to automatically deploy a new version to [PyPi](https://pypi.org/) and [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

1) Update the version number in [`pyproject.toml`](https://github.com/lysj-cpu/icland/blob/main/pyproject.toml), which should be near the top of the file:

```toml
[project]
name = "icland"
version = "0.1.0a2"  # <-- This line here
```

2) Push the commit you want to publish (including the updated `pyproject.toml`)

3) Create a new [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging)

```shell
$ git tag -a v1.2.3 -m "v1.2.3"
$ git push origin v1.2.3
```

4) Celebrate the new release! 🎉
