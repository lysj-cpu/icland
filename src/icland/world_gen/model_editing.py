from typing import Any  # noqa: D100

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from icland.agent import create_agent
from icland.constants import (
    AGENT_COMPONENT_IDS_DIM,
    BODY_OFFSET,
    WALL_OFFSET,
    WORLD_LEVEL,
)
from icland.types import ICLandConfig, MjxModelType


def generate_base_model(
    config: ICLandConfig,
    # prop_spawns: jax.Array,
) -> MjxModelType:  # pragma: no cover
    """Generates base MJX model from column meshes that form the world."""
    # This code is run entirely on CPU
    width = config.world_width
    height = config.world_height
    max_agent_count = config.max_agent_count
    max_world_level = config.max_world_level

    spec = mujoco.MjSpec()

    spec.compiler.degree = 1

    # Add assets
    # Ramp
    spec.add_mesh(
        name="ramp",
        uservert=[
            -0.5,
            -0.5,
            0,
            0.5,
            -0.5,
            0,
            0.5,
            0.5,
            0,
            -0.5,
            0.5,
            0,
            0.5,
            -0.5,
            1,
            0.5,
            0.5,
            1,
        ],
    )

    # Add the ground
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], rgba=[1, 1, 1, 1]
    )

    # Add the walls
    if width > 0 and height > 0:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[height / 2, 10, 0.01],
            quat=[0.5, -0.5, -0.5, 0.5],
            pos=[width, height / 2, 10],
            rgba=[0, 0, 0, 0],
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[height / 2, 10, 0.01],
            quat=[0.5, 0.5, 0.5, 0.5],
            pos=[0, height / 2, 10],
            rgba=[0, 0, 0, 0],
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, width / 2, 0.01],
            quat=[0.5, -0.5, 0.5, 0.5],
            pos=[width / 2, 0, 10],
            rgba=[0, 0, 0, 0],
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, width / 2, 0.01],
            quat=[0.5, 0.5, -0.5, 0.5],
            pos=[width / 2, height, 10],
            rgba=[0, 0, 0, 0],
        )

    # Default constants
    COLUMN_HEIGHT = 3
    RAMP_HEIGHT = 2

    # Add tiles, all at max height to create correct BVH interactions
    # Tile indices: 5 to w * h + 4 inclusive
    for i in range(width):
        for j in range(height):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[i + 0.5, j + 0.5, max_world_level - COLUMN_HEIGHT],
                size=[0.5, 0.5, 3],
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="ramp",
                pos=[i + 0.5, j + 0.5, max_world_level - RAMP_HEIGHT],
            )

    # Add agents
    # Agent indices: w * h + 5 to num_agents + w * h + 4 inclusive
    for i in range(max_agent_count):
        create_agent(i, jnp.zeros((AGENT_COMPONENT_IDS_DIM,)), spec)  # default position

    # TODO: Add props
    mj_model = spec.compile()
    mjx_model = mujoco.mjx.put_model(mj_model)

    return mjx_model, mj_model


@jax.jit
def edit_model_data(
    tilemap: jax.Array,
    base_model: MjxModelType,
    agent_spawns: jax.Array,  # shape (num_agents, 3)
    prop_spawns: jax.Array = jnp.array([]),  # shape (num_props, 3)
    max_world_level: int = WORLD_LEVEL,
) -> MjxModelType:
    """Edit the base model data such that the terrain matches that of the tilemap."""
    # Pre: the width and height of the tilemap MUST MATCH that of the base_model

    RAMP_OFFSET = 13 / 3
    COL_OFFSET = 2
    agent_count = agent_spawns.shape[0]
    prop_count = prop_spawns.shape[0]
    b_geom_xpos = base_model.geom_pos
    b_geom_xquat = base_model.geom_quat
    b_pos = base_model.body_pos
    # b_agent_pos, b_prop_pos = (
    #     b_pos[1:agent_count + 1],
    #     b_pos[agent_count + 1 : agent_count + prop_count + 1],
    # )
    w, h = tilemap.shape[0], tilemap.shape[1]

    def rot_offset(i: jax.Array) -> jax.Array:
        return 0.5 + jnp.cos(jnp.pi * i / 2) / 6

    def process_tile(
        i: jax.Array, tile: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        t_type, rot, _, to_h = tile
        is_ramp = t_type % 2
        offset = to_h - is_ramp - max_world_level + 1

        x, y = i // w, i % w
        quat_consts = jnp.array(
            [
                [0, 0.382683, 0, 0.92388],
                [-0.653282, 0.270598, 0.270598, 0.653282],
                [-0.92388, 0, 0.382683, 0],
                [-0.653282, -0.270598, 0.270598, -0.653282],
            ]
        )

        return (
            jnp.array([x + 0.5, y + 0.5, offset + COL_OFFSET]),
            jnp.array(
                [
                    x + rot_offset(rot),
                    y + rot_offset(rot - 1),
                    RAMP_OFFSET + offset + is_ramp,
                ]
            ),
            jnp.array([1, 0, 0, 0]),
            quat_consts[rot],
        )

    t_cpos, t_rpos, t_cquat, t_rquat = jax.vmap(process_tile, in_axes=(0, 0))(
        jnp.arange(w * h, dtype=int), jnp.reshape(tilemap, (w * h, -1))
    )
    tile_offsets_aligned = jnp.stack([t_cpos, t_rpos], axis=1).reshape(
        -1, t_cpos.shape[1]
    )
    tile_quats_aligned = jnp.stack([t_cquat, t_rquat], axis=1).reshape(
        -1, t_cquat.shape[1]
    )

    b_geom_xpos = jax.lax.dynamic_update_slice_in_dim(
        b_geom_xpos, tile_offsets_aligned, WALL_OFFSET, axis=0
    )

    b_geom_xquat = jax.lax.dynamic_update_slice_in_dim(
        b_geom_xquat, tile_quats_aligned, WALL_OFFSET, axis=0
    )
    b_pos = jax.lax.dynamic_update_slice_in_dim(
        b_pos, agent_spawns.astype("float32"), BODY_OFFSET, axis=0
    )
    b_pos = jax.lax.dynamic_update_slice_in_dim(
        b_pos,
        prop_spawns.astype("float32").reshape((-1, 3)),
        BODY_OFFSET + agent_count,
        axis=0,
    )

    return base_model.replace(
        geom_pos=b_geom_xpos, geom_quat=b_geom_xquat, body_pos=b_pos
    )


def _edit_mj_model_data(
    tilemap: jax.Array, base_model: mujoco.MjModel, max_world_level: int = WORLD_LEVEL
) -> None:  # pragma: no cover
    b_geom_xpos = base_model.geom_pos
    b_geom_xquat = base_model.geom_quat
    w, h = tilemap.shape[0], tilemap.shape[1]
    RAMP_OFFSET = 13 / 3
    COL_OFFSET = 2
    WALL_OFFSET = 5

    def rot_offset(i: jax.Array) -> jax.Array:
        return 0.5 + jnp.cos(jnp.pi * i / 2) / 6

    def process_tile(
        i: int, tile: jax.Array
    ) -> tuple[
        np.ndarray[tuple[int, ...], Any],
        np.ndarray[tuple[int, ...], Any],
        jax.Array,
    ]:
        t_type, rot, _, to_h = tile
        is_ramp = t_type % 2
        offset = to_h - is_ramp - max_world_level - 1

        x, y = i // w, i % w
        quat_consts = jnp.array(
            [
                [0, 0.382683, 0, 0.92388],
                [-0.653282, 0.270598, 0.270598, 0.653282],
                [-0.92388, 0, 0.382683, 0],
                [-0.653282, -0.270598, 0.270598, -0.653282],
            ]
        )
        return (
            np.array([x + 0.5, y + 0.5, offset + COL_OFFSET]),
            np.array(
                [
                    x + rot_offset(rot),
                    y + rot_offset(rot - 1),
                    RAMP_OFFSET + offset + is_ramp,
                ]
            ).astype("float32"),
            quat_consts[rot],
        )

    for i in range(w * h):
        c, r, rq = process_tile(i, tilemap[i // w, i % w])
        b_geom_xpos[2 * i + WALL_OFFSET] = c
        b_geom_xpos[2 * i + WALL_OFFSET + 1] = r
        b_geom_xquat[2 * i + WALL_OFFSET + 1] = rq
