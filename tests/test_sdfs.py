"""Test suite for signed distance functions."""

import jax.numpy as jnp
import numpy as np

from icland.renderer.sdfs import *

NUM_ITERS = 10


def test_ramp_sdf() -> None:
    """Test the ramp signed distance function."""
    dists = []
    for i in range(NUM_ITERS):
        dists.append(
            float(ramp_sdf(jnp.full((3,), i * 1.0), jnp.array(1), jnp.array(1)))
        )

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    0.03983153775334358,
                    1.5537738800048828,
                    2.6912150382995605,
                    3.36519718170166,
                    3.805694818496704,
                    4.181540489196777,
                    4.5145134925842285,
                    4.816433906555176,
                    5.0945515632629395,
                    5.3536810874938965,
                ]
            ),
            atol=1e-05,
        )
    )


def test_box_sdf() -> None:
    """Test the box signed distance function."""
    dists = []
    for i in range(NUM_ITERS):
        dists.append(
            float(box_sdf(jnp.full((3,), i * 1.0), jnp.array(1), jnp.array(1)))
        )

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    -0.49949952960014343,
                    0.7071071267127991,
                    2.345208168029785,
                    4.062018871307373,
                    5.7879180908203125,
                    7.516647815704346,
                    9.246621131896973,
                    10.977249145507812,
                    12.70826530456543,
                    14.439528465270996,
                ]
            ),
            atol=1e-05,
        )
    )


def test_capsule_sdf() -> None:
    """Test the capsule signed distance function."""
    dists = []
    for i in range(NUM_ITERS):
        dists.append(float(capsule_sdf(jnp.full((3,), i * 1.0), 1, 1)))

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    -0.9995002746582031,
                    0.41421353816986084,
                    2.0,
                    3.690415382385254,
                    5.40312385559082,
                    7.124037742614746,
                    8.848857879638672,
                    10.575836181640625,
                    12.304134368896484,
                    14.033295631408691,
                ]
            ),
            atol=1e-05,
        )
    )


def test_sphere_sdf() -> None:
    """Test the sphere signed distance function."""
    dists = []
    for i in range(NUM_ITERS):
        dists.append(float(sphere_sdf(jnp.full((3,), i * 1.0), 1)))

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    -1.0,
                    0.7320507764816284,
                    2.464101552963257,
                    4.196152210235596,
                    5.928203105926514,
                    7.660253524780273,
                    9.392304420471191,
                    11.12435531616211,
                    12.856406211853027,
                    14.588457107543945,
                ]
            ),
            atol=1e-05,
        )
    )


def test_cube_sdf() -> None:
    """Test the cube signed distance function."""
    dists = []
    for i in range(NUM_ITERS):
        dists.append(float(cube_sdf(jnp.full((3,), i * 1.0), 1.0)))

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    -0.49919065833091736,
                    0.8660256862640381,
                    2.598076105117798,
                    4.330126762390137,
                    6.062177658081055,
                    7.794228553771973,
                    9.52627944946289,
                    11.258330345153809,
                    12.990381240844727,
                    14.722432136535645,
                ]
            ),
            atol=1e-05,
        )
    )


def test_beam_sdf() -> None:
    """Test the beam signed distance function."""
    dists = []
    view_dir = jnp.array([1, 1, 1]) / jnp.linalg.norm(jnp.array([1, 1, 1]))
    for i in range(NUM_ITERS):
        dists.append(float(beam_sdf(jnp.full((3,), i * 1.0), view_dir, 1)))

    assert np.all(
        np.isclose(
            dists,
            np.array(
                [
                    0.0,
                    0.7320507764816284,
                    2.464101791381836,
                    4.196152210235596,
                    5.928203105926514,
                    7.660254001617432,
                    9.392305374145508,
                    11.124356269836426,
                    12.856407165527344,
                    14.588457107543945,
                ]
            ),
            atol=1e-05,
        )
    )
