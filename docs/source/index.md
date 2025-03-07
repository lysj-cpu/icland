---
og:title: icland
---

::::{grid} 1 2 2 2
:padding: 0

:::{grid-item}
:child-align: center
<div align="center">
    <h1>ICLand</h1>
    <a href="https://pypi.org/project/icland/">
        <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/icland?style=flat-square&logo=pypi&logoColor=white&color=blue">
    </a>
    <a href="https://github.com/lysj-cpu/icland/actions/workflows/tests.yml">
        <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/lysj-cpu/icland/tests.yml?style=flat-square&logo=github">
    </a>
    <a href="https://app.codecov.io/github/lysj-cpu/icland">
        <img alt="Codecov" src="https://img.shields.io/codecov/c/github/lysj-cpu/icland?style=flat-square&logo=codecov">
    </a>
</div>
:::

:::{grid-item}
\
\
Recreating Google DeepMind's [XLand RL environment](https://deepmind.google/discover/blog/generally-capable-agents-emerge-from-open-ended-play/) in JAX

```{button-ref} quickstart
:ref-type: doc
:color: primary
:expand:

Get Started
```
:::

::::

::::{grid} 1 3 3 3

:::{grid-item-card} {octicon}`rocket;1.5em` Real-Time GPU Rendering & Editing
:link: benchmarking
:link-type: doc
Harnesses [ray marching](https://en.wikipedia.org/wiki/Ray_marching) with [Signed Distance Functions (SDFs)](https://en.wikipedia.org/wiki/Signed_distance_function) for ultra-smooth rendering and GPU-accelerated model editing.
:::

:::{grid-item-card} {octicon}`globe;1.5em` Procedural World Generation
:link: system_architecture
:link-type: doc
Utilises [Wave Function Collapse (WFC)](https://en.wikipedia.org/wiki/Wave_function_collapse) and randomised sampling to create diverse, dynamic environments for RL agents.
:::

:::{grid-item-card} {octicon}`gear;1.5em` Powerful API
:link: reference/library
:link-type: doc
A fully type-annotated, extensively documented Python RL library, compatible with [Brax](https://github.com/google/brax), and designed for [JAX](http://jax.readthedocs.io/)-first workflows.
:::

::::

![Simulation Visualisation](_static/simulation.jpeg)

```{toctree}
:hidden:

🔎 Overview <self>
background
<!-- quickstart -->
system_architecture
benchmarking
```
```{toctree}
:caption: 🔨 Contributing
:hidden:

contributing/installation
contributing/documentation
contributing/new_version
```
```{toctree}
:caption: 📖 Reference
:hidden:
:glob:

reference/*
```
