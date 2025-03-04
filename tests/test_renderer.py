"""Test utility functions in the renderer file."""

from functools import partial

import jax
import jax.numpy as jnp

import icland.renderer.sdfs as Sdf
from icland.presets import (
    TEST_FRAME_WITH_PROPS,
    TEST_TILEMAP_BUMP,
    TEST_TILEMAP_FLAT,
)
from icland.renderer.renderer import *


def test_can_see_object() -> None:
    """Test if the can_see_object func returns true in unoccluded case."""
    # Player                       Sphere
    #  [] ----------------------->   ()
    # ===================================
    agent_pos = jnp.array([0.5, 3.4, 0])
    agent_dir = jnp.array([0, 0, 1])

    prop_pos = jnp.array([0.5, 3.5, 10])
    prop_sdf = partial(Sdf.sphere_sdf, r=0.5)

    terrain_sdf = lambda x: scene_sdf_from_tilemap(TEST_TILEMAP_FLAT, x)[0]
    visible = can_see_object(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        obj_pos=prop_pos,
        obj_sdf=prop_sdf,
        terrain_sdf=terrain_sdf,
    )
    assert visible

    terrain_sdf_2 = lambda x: scene_sdf_from_tilemap(TEST_TILEMAP_BUMP, x)[0]
    visible = can_see_object(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        obj_pos=prop_pos,
        obj_sdf=prop_sdf,
        terrain_sdf=terrain_sdf_2,
    )
    assert not visible


def test_generate_colormap() -> None:
    """Test the dummy generate_colormap function."""
    w, h = 10, 10
    cmap = generate_colormap(jax.random.PRNGKey(42), w, h)
    assert cmap.shape == (w, h, 3)
    res = jnp.logical_and(cmap >= 0.0, cmap <= 1.0)
    assert jnp.all(res, axis=None)


def test_render_frame_with_objects() -> None:
    """Test if the render_frame_with_objects can correctly render one frame with props."""
    players = RenderAgentInfo(
        jnp.array([[8.5, 3, 1]]),
        jnp.array([[0, -0.5, 1.0]]),
        jnp.array([[1.0, 0.0, 1.0]]),
    )
    props = RenderPropInfo(
        jnp.array([1]),
        jnp.array([[4, 3, 1]]),
        jnp.array([[1, 0, 0, 0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )
    frame = render_frame_with_objects(
        jnp.array([0, 5.0, -10]),
        jnp.array([0, -0.5, 1.0]),
        TEST_TILEMAP_BUMP,
        jnp.array([[[0, 1, 0] for _ in range(10)] for _ in range(10)]),
        players,
        props,
        jnp.zeros((1, 6)),
        view_width=10,
        view_height=10,
    )
    assert (
        jnp.linalg.norm(
            frame[1:6, :5].flatten() - TEST_FRAME_WITH_PROPS[1:6, :5].flatten(),
            ord=jnp.inf,
        )
        < 0.1
    )
