"""Test script for SDFs."""

import jax
import jax.numpy as jnp

from icland.world_gen.sdfs import box_sdf, capsule_sdf, ramp_sdf


def test_capsule_sdf() -> None:
    """Test the capsule_sdf function."""
    # JIT compile the function
    jit_func = jax.jit(capsule_sdf)
    csdf = []
    for i in range(10):
        d = i / 2
        csdf.append(float(capsule_sdf(jnp.full((3,), d), 2, 1)))
    assert csdf == [
        -1.0,
        -0.1339746117591858,
        0.7320507764816284,
        1.5980761051177979,
        2.464101552963257,
        3.330127239227295,
        4.196152210235596,
        5.062177658081055,
        5.928203105926514,
        6.794228553771973,
    ]


def test_box_sdf() -> None:
    """Test the box_sdf function."""
    # JIT compile the function
    jit_func = jax.jit(box_sdf)
    bsdf = []
    for i in range(10):
        d = i / 2
        bsdf.append(float(box_sdf(jnp.full((3,), d), 2, 1)))

    assert bsdf == [
        -0.9991908669471741,
        -0.49919065833091736,
        0.0006273954641073942,
        0.8660256862640381,
        1.7320510149002075,
        2.598076343536377,
        3.4641013145446777,
        4.330127239227295,
        5.196152210235596,
        6.062177658081055,
    ]


def test_ramp_sdf() -> None:
    """Test the ramp_sdf function."""
    # JIT compile the function
    jit_func = jax.jit(ramp_sdf)
    rsdf = []
    for i in range(10):
        d = i / 2
        rsdf.append(float(ramp_sdf(jnp.full((3,), d), 1, 2)))

    assert rsdf == [
        4.76837158203125e-07,
        5.514593794941902e-07,
        0.7071069478988647,
        1.5,
        2.345207691192627,
        3.20156192779541,
        4.062018871307373,
        4.924428462982178,
        5.7879180908203125,
        6.652067184448242,
    ]
