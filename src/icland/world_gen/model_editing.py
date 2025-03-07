"""Code to parallelize model editing in JAX given a base model."""

import jax
import jax.numpy as jnp
import mujoco

from icland.agent import create_agent
from icland.constants import (
    AGENT_DOF_OFFSET,
    BODY_OFFSET,
    TIMESTEP,
    WALL_OFFSET,
    WORLD_LEVEL,
)
from icland.prop import PropType, create_prop
from icland.types import ICLandAgentInfo, ICLandPropInfo, MjxModelType


def generate_base_model(
    max_world_width: int,
    max_world_depth: int,
    max_world_height: int,
    max_agent_count: int,
    max_sphere_count: int,
    max_cube_count: int,
) -> MjxModelType:  # pragma: no cover
    """Generates base MJX model from column meshes that form the world.

    This code is run entirely on CPU and should only be used in the
    smart constructor of the `ICLandConfig` object.

    Args:
        max_world_width: The maximum width of the world.
        max_world_depth: The maximum depth of the world.
        max_world_height: The maximum height of the world.
        max_agent_count: The maximum number of agents.
        max_sphere_count: The maximum number of spheres.
        max_cube_count: The maximum number of cubes.

    Returns:
        A tuple containing the MJX model and the MuJoCo model.
    """
    spec = mujoco.MjSpec()

    spec.compiler.degree = 1
    spec.add_material(
        name="default",
        rgba=[0.8, 0.8, 0.8, 1],
    )

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
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.01],
        rgba=[1, 1, 1, 1],
        material="default",
    )

    # Add the walls
    if max_world_width > 0 and max_world_depth > 0:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[max_world_depth / 2, 10, 0.01],
            quat=[0.5, -0.5, -0.5, 0.5],
            pos=[max_world_width, max_world_depth / 2, 10],
            rgba=[0, 0, 0, 0],
            material="default",
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[max_world_depth / 2, 10, 0.01],
            quat=[0.5, 0.5, 0.5, 0.5],
            pos=[0, max_world_depth / 2, 10],
            rgba=[0, 0, 0, 0],
            material="default",
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, max_world_width / 2, 0.01],
            quat=[0.5, -0.5, 0.5, 0.5],
            pos=[max_world_width / 2, 0, 10],
            rgba=[0, 0, 0, 0],
            material="default",
        )

        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, max_world_width / 2, 0.01],
            quat=[0.5, 0.5, -0.5, 0.5],
            pos=[max_world_width / 2, max_world_depth, 10],
            rgba=[0, 0, 0, 0],
            material="default",
        )

    # Default constants
    COLUMN_HEIGHT = 3
    RAMP_HEIGHT = 2

    # Add tiles, all at max height to create correct BVH interactions
    # Tile indices: 5 to w * h + 4 inclusive
    for i in range(max_world_width):
        for j in range(max_world_depth):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[i + 0.5, j + 0.5, max_world_height - COLUMN_HEIGHT],
                size=[0.5, 0.5, 3],
                material="default",
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="ramp",
                pos=[i + 0.5, j + 0.5, max_world_height - RAMP_HEIGHT],
                material="default",
            )

    # Add agents
    # Agent indices: w * h + 5 to num_agents + w * h + 4 inclusive
    for i in range(max_agent_count):
        create_agent(i, jnp.zeros((3,)), spec)  # default position

    max_prop_count = max_cube_count + max_sphere_count
    if max_prop_count > 0:
        curr_ind = 0
        for _ in range(max_cube_count):
            create_prop(curr_ind, jnp.zeros((3,)), spec, PropType.CUBE)
            curr_ind += 1

        for _ in range(max_sphere_count):
            create_prop(curr_ind, jnp.zeros((3,)), spec, PropType.SPHERE)
            curr_ind += 1
    else:
        # Create empty placeholder prop.
        # This ensures the renderer and model editing functions can run
        # as expected, as JAX dislikes empty arrays.
        # See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html
        create_prop(0, jnp.zeros((3,)), spec, PropType.NONE)

    mj_model = spec.compile()
    mj_model.opt.timestep = TIMESTEP
    mjx_model = mujoco.mjx.put_model(mj_model)

    return mjx_model, mj_model


@jax.jit
def edit_model_data(
    tilemap: jax.Array,
    base_model: MjxModelType,
    agent_info: ICLandAgentInfo,
    prop_info: ICLandPropInfo,
    max_world_height: int = WORLD_LEVEL,
) -> MjxModelType:
    """Edit the base model data such that the terrain matches that of the tilemap.

    **NOTE**: the width and deoth of the tilemap MUST MATCH that of the base_model.

    Args:
        tilemap: A JAX array representing the tilemap of the world.
        base_model: The base MJX model to be edited.
        agent_info: Information about the agents in the environment.
        prop_info: Information about the props in the environment.
        max_world_height: The maximum height of the world.

    Returns:
        The edited MJX model.
    """
    agent_spawns = agent_info.spawn_points
    prop_spawns = prop_info.spawn_points

    RAMP_OFFSET = 13 / 3
    COL_OFFSET = 2
    agent_count = agent_spawns.shape[0]
    prop_count = prop_spawns.shape[0]
    b_geom_xpos = base_model.geom_pos
    b_geom_xquat = base_model.geom_quat
    b_pos = base_model.body_pos
    b_q_pos0 = base_model.qpos0
    b_q_pos_spring = base_model.qpos_spring
    w, h = tilemap.shape[0], tilemap.shape[1]

    def rot_offset(i: jax.Array) -> jax.Array:
        return 0.5 + jnp.cos(jnp.pi * i / 2) / 6

    def process_tile(
        i: jax.Array, tile: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        t_type, rot, _, to_h = tile
        is_ramp = t_type % 2
        offset = to_h - is_ramp - max_world_height + 1

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

    prop_spawns = prop_spawns.astype("float32").reshape((-1, 3))

    # Update props qsprings
    prop_qpos = jax.vmap(
        lambda s: jnp.concatenate([s, jnp.array([1, 0, 0, 0])], axis=0)
    )(prop_spawns)

    b_q_pos0 = jax.lax.dynamic_update_slice_in_dim(
        b_q_pos0,
        prop_qpos.astype("float32").flatten(),
        WALL_OFFSET - 1 + AGENT_DOF_OFFSET * agent_count,
        axis=0,
    )

    b_q_pos_spring = jax.lax.dynamic_update_slice_in_dim(
        b_q_pos_spring,
        prop_qpos.astype("float32").flatten(),
        WALL_OFFSET - 1 + AGENT_DOF_OFFSET * agent_count,
        axis=0,
    )

    return base_model.replace(
        geom_pos=b_geom_xpos,
        geom_quat=b_geom_xquat,
        body_pos=b_pos,
        qpos0=b_q_pos0,
        qpos_spring=b_q_pos_spring,
    )
