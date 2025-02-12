"""Test utility functions in the renderer file."""

from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from assets.frames import *
from assets.worlds import EMPTY_WORLD

import icland
import icland.renderer.sdfs as Sdf
from icland.renderer.renderer import (
    _scene_sdf_from_tilemap,
    can_see_object,
    generate_colormap,
    get_agent_camera_from_mjx,
    render_frame,
    render_frame_with_objects,
)
from icland.types import ICLandParams


def test_can_see_object_free() -> None:
    """Test if the can_see_object func returns true in unoccluded case."""
    # Player                       Sphere
    #  [] ----------------------->   ()
    # ===================================
    player_pos = jnp.array([0.5, 3.4, 0])
    player_dir = jnp.array([0, 0, 1])

    prop_pos = jnp.array([0.5, 3.5, 10])
    prop_sdf = partial(Sdf.sphere_sdf, r=0.5)

    terrain_sdf = lambda x: _scene_sdf_from_tilemap(TEST_TILEMAP_FLAT, x)[0]
    assert can_see_object(
        player_pos=player_pos,
        player_dir=player_dir,
        obj_pos=prop_pos,
        obj_sdf=prop_sdf,
        terrain_sdf=terrain_sdf,
    )


def test_can_see_object_occluded() -> None:
    """Test if the can_see_object func returns false in occluded case."""
    # Player                       Sphere
    #  [] ---------> ||              ()
    # ===============||==================
    player_pos = jnp.array([0.5, 3.4, 0])
    player_dir = jnp.array([0, 0, 1])

    prop_pos = jnp.array([0.5, 3.5, 10])
    prop_sdf = partial(Sdf.sphere_sdf, r=0.5)

    terrain_sdf = lambda x: _scene_sdf_from_tilemap(TEST_TILEMAP_BUMP, x)[0]
    assert not can_see_object(
        player_pos=player_pos,
        player_dir=player_dir,
        obj_pos=prop_pos,
        obj_sdf=prop_sdf,
        terrain_sdf=terrain_sdf,
    )


def test_get_agent_camera_from_mjx() -> None:
    """Test if the get_agent_camera_from_mjx transforms the positions."""
    icland_params = ICLandParams(mujoco.MjModel.from_xml_string(EMPTY_WORLD), None, 1)
    icland_state = icland.init(jax.random.PRNGKey(0), icland_params)
    world_width = 10

    agent_pos = icland_state.pipeline_state.mjx_data.xpos[
        icland_state.pipeline_state.component_ids[0, 0]
    ][:3]
    print(agent_pos)
    height_offset = 0.2
    camera_offset = 0.06
    cam_pos, cam_dir = get_agent_camera_from_mjx(
        icland_state,
        world_width,
        0,
        camera_height=height_offset,
        camera_offset=camera_offset,
    )
    assert jnp.allclose(
        cam_pos,
        jnp.array(
            [
                -agent_pos[0] + world_width - camera_offset,
                agent_pos[2] + height_offset,
                agent_pos[1],
            ]
        ),
    )
    assert jnp.allclose(cam_dir, jnp.array([-1, 0, 0]))


def test_render_frame() -> None:
    """Tests if the renderer can correctly render one frame."""
    frame = render_frame(
        jnp.array([0, 5.0, -10]),
        jnp.array([0, -0.5, 1.0]),
        TEST_TILEMAP_BUMP,
        view_width=10,
        view_height=10,
    )
    assert jnp.linalg.norm(frame.flatten() - TEST_FRAME.flatten(), ord=jnp.inf) < 0.05


def test_generate_colormap() -> None:
    """Test the dummy colormap generation function."""
    w, h = 10, 10
    cmap = generate_colormap(jax.random.PRNGKey(42), w, h)
    assert cmap.shape == (w, h, 3)
    res = jnp.logical_and(cmap >= 0.0, cmap <= 1.0)
    assert jnp.all(res, axis=None)


def test_render_frame_with_objs() -> None:
    """Test if the renderer can correctly render one frame with props."""
    key = jax.random.PRNGKey(42)
    frame = render_frame_with_objects(
        jnp.array([0, 5.0, -10]),
        jnp.array([0, -0.5, 1.0]),
        TEST_TILEMAP_BUMP,
        generate_colormap(key, 10, 10),
        view_width=10,
        view_height=10,
    )
    assert (
        jnp.linalg.norm(frame.flatten() - TEST_FRAME_WITH_PROPS.flatten(), ord=jnp.inf)
        < 0.05
    )
