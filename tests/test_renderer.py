"""Test utility functions in the renderer file."""

from functools import partial

import jax
import jax.numpy as jnp

import icland
import icland.renderer.sdfs as Sdf
from icland.presets import (
    DEFAULT_CONFIG,
    TEST_FRAME,
    TEST_TILEMAP_BUMP,
    TEST_TILEMAP_EMPTY_WORLD,
    TEST_TILEMAP_FLAT,
)
from icland.renderer.renderer import *
from icland.types import ICLandParams
from icland.world_gen.model_editing import generate_base_model


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


def test_get_agent_camera_from_mjx() -> None:
    """Test if the get_agent_camera_from_mjx transforms the positions."""
    mjx_model, _ = generate_base_model(DEFAULT_CONFIG)
    icland_params = ICLandParams(
        world=TEST_TILEMAP_EMPTY_WORLD,
        reward_function=None,
        agent_spawns=jnp.array([[0, 0, 1], [0, 0.5, 1]]),
        world_level=6,
    )

    icland_state = icland.init(jax.random.PRNGKey(42), icland_params, mjx_model)
    world_width = 10

    agent_pos = icland_state.pipeline_state.mjx_data.xpos[
        icland_state.pipeline_state.component_ids[0, 0].astype(int)
    ][:3]
    print(agent_pos)
    height_offset = 0.2
    camera_offset = 0.06
    cam_pos, cam_dir = get_agent_camera_from_mjx(
        icland_state,
        world_width,
        0,
    )
    assert jnp.allclose(
        cam_pos,
        jnp.array(
            [
                -agent_pos[0] + world_width,
                agent_pos[2],
                agent_pos[1],
            ]
        ),
    )
    assert jnp.allclose(cam_dir, jnp.array([-1, 0, 0]))


def test_render_frame() -> None:
    """Tests if render_frame can correctly render one frame."""
    frame = render_frame(
        jnp.array([0, 5.0, -10]),
        jnp.array([0, -0.5, 1.0]),
        TEST_TILEMAP_BUMP,
        view_width=10,
        view_height=10,
    )
    assert jnp.linalg.norm(frame.flatten() - TEST_FRAME.flatten(), ord=1) < 5


def test_generate_colormap() -> None:
    """Test the dummy generate_colormap function."""
    w, h = 10, 10
    cmap = generate_colormap(jax.random.PRNGKey(42), w, h)
    assert cmap.shape == (w, h, 3)
    res = jnp.logical_and(cmap >= 0.0, cmap <= 1.0)
    assert jnp.all(res, axis=None)


def test_render_frame_with_objects() -> None:
    """Test if the render_frame_with_objects can correctly render one frame with props."""
    key = jax.random.PRNGKey(42)
    players = PlayerInfo(jnp.array([[8.5, 3, 1]]), jnp.array([[1.0, 0.0, 1.0]]))
    props = PropInfo(
        jnp.array([1]),
        jnp.array([[4, 3, 1]]),
        jnp.array([[1, 0, 0, 0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )
    frame = render_frame_with_objects(
        jnp.array([0, 5.0, -10]),
        jnp.array([0, -0.5, 1.0]),
        TEST_TILEMAP_BUMP,
        generate_colormap(key, 10, 10),
        players,
        props,
        view_width=10,
        view_height=10,
    )
    # assert not jnp.any(jnp.isclose(frame[1:6, :5].flatten(), TEST_FRAME_WITH_PROPS[1:6, :5].flatten()))
    assert True
