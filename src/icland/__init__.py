"""Recreating Google DeepMind's XLand RL environment in JAX."""

from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx

from icland.constants import (
    AGENT_DOF_OFFSET,
    AGENT_VARIABLES_DIM,
    PROP_DOF_OFFSET,
    PROP_DOF_MULTIPLIER,
    WALL_OFFSET,
)
from icland.types import *
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.converter import sample_spawn_points
from icland.world_gen.model_editing import edit_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES


def config(*args: Tuple[int, ...]) -> ICLandConfig:
    """Smart constructor for ICLand config, initialising the base model on the fly."""

    # Unpack the arguments
    world_width, world_depth, world_height, agent_count = args[0:4]

    return ICLandConfig(
        *args,
        model=generate_base_model(world_width, world_depth, world_height, agent_count)[0],
    )


# Default configuration
DEFAULT_CONFIG = config(
    2,
    2,
    6,
    1,
    0,
    0,
)

@jax.jit
def sample(key: jax.Array, config: ICLandConfig = DEFAULT_CONFIG) -> ICLandParams:
    """Sample the world and generate the initial parameters for the ICLand environment.

    Args:
        key: The random key for sampling.
        config: The configuration for the ICLand environment.

    Returns:
        The initial parameters for the ICLand environment.

    Example:
        >>> import jax
        >>> import icland
        >>> key = jax.random.PRNGKey(42)
        >>> icland.sample(key)
        ICLandParams(world=ICLandWorld(...), agent_info=ICLandAgentInfo(...), prop_info=ICLandPropInfo(...), reward_function=None)
    """
    
    # Unpack config
    (
        max_world_width,
        max_world_depth,
        max_world_height,
        max_agent_count,
        max_sphere_count,
        max_cube_count,
        model
    ) = vars(config).values()

    # Define constants
    USE_PERIOD = True
    HEURISTIC = 1

    # Sample the world via wave function collapse
    wfc_model = sample_world(
        width=max_world_width,
        height=max_world_depth,
        key=key,
        periodic=USE_PERIOD,
        heuristic=HEURISTIC,
    )

    # Export the world tilemap
    world_tilemap = export(
        model=wfc_model,
        tilemap=TILECODES,
        width=max_world_width,
        height=max_world_depth,
    )

    # Sample number of props and agents
    max_object_count = max_agent_count + max_sphere_count + max_cube_count
    max_prop_count = max_sphere_count + max_cube_count

    # Sample spawn points for objects
    spawnpoints = sample_spawn_points(
        key=key, tilemap=world_tilemap, num_objects=max_object_count
    )

    # Update with randomised number of agents
    num_agents = jax.random.randint(key, (), 1, max_agent_count)

    # Update with randomised number of props
    num_props = jax.random.randint(key, (), 0, max_prop_count)
    spawnpoints = jax.lax.dynamic_update_slice_in_dim(
        spawnpoints,
        jax.vmap(lambda i: (i < num_agents) * spawnpoints[i])(
            jnp.arange(max_agent_count)
        ),
        0,
        axis=0,
    )
    spawnpoints = jax.lax.dynamic_update_slice_in_dim(
        spawnpoints,
        jax.vmap(lambda i: (i < num_props) * spawnpoints[i])(
            jnp.arange(max_prop_count)
        ),
        max_agent_count,
        axis=0,
    )

    # Generate the agent and prop information
    agent_info = ICLandAgentInfo(
        agent_count=num_agents,
        spawn_points=spawnpoints[:max_agent_count],
        spawn_orientations=jnp.zeros((max_agent_count,), dtype="float32"),
        body_ids=jnp.arange(1, max_agent_count + 1, dtype="int32"),
        geom_ids=(jnp.arange(max_agent_count) + max_world_width * max_world_depth) * 2
        + WALL_OFFSET,
        dof_addresses=jnp.arange(max_agent_count) * AGENT_DOF_OFFSET,
        colour=jnp.zeros((max_agent_count,), dtype="int"),
    )

    # Generate the prop information
    prop_info = ICLandPropInfo(
        prop_count=num_props,
        prop_types=jnp.zeros((max_prop_count, ), dtype="int32"),
        spawn_points=spawnpoints[max_agent_count:max_object_count],
        spawn_rotations=jnp.zeros((max_prop_count,), dtype="float32"),
        body_ids=jnp.arange(1 + max_agent_count, max_agent_count + max_prop_count + 1, dtype="int32"),
        geom_ids=max_agent_count * 2 + jnp.arange(max_prop_count) + max_agent_count + max_world_width * max_world_depth
        + WALL_OFFSET,
        dof_addresses=jnp.arange(max_prop_count) * PROP_DOF_MULTIPLIER + PROP_DOF_OFFSET,
        colour=jnp.zeros((max_prop_count,), dtype="float32"),
    )

    # Edit the model data for the specification
    model = edit_model_data(world_tilemap, model, agent_info, prop_info)
    icland_world = ICLandWorld(
        tilemap=world_tilemap,
        max_world_width=max_world_width,
        max_world_depth=max_world_depth,
        max_world_height=max_world_height,
    )

    # Return the parameters
    return ICLandParams(
        world=icland_world,
        agent_info=agent_info,
        prop_info=prop_info,
        reward_function=None,
        mjx_model=model,
    )

@jax.jit
def init(icland_params: ICLandParams) -> ICLandState:
    """Initialise the ICLand environment.
    
    Args:
        icland_params: The parameters for the ICLand environment.

    Returns:
        The initial state of the ICLand environment.

    Example:
        >>> import icland
        >>> icland_params = icland.sample(jax.random.PRNGKey(42))
        >>> icland.init(icland_params)
        ICLandState(mjx_data=DeviceArray(...), agent_variables=ICLandAgentVariables(...), prop_variables=ICLandPropVariables(...))
    """

    max_agent_count = icland_params.agent_info.spawn_points.shape[0]
    max_prop_count = icland_params.prop_info.spawn_points.shape[0]

    agent_variables = ICLandAgentVariables(
        pitch=jnp.zeros((max_agent_count,), dtype="float32"),
        is_tagged=jnp.zeros((max_agent_count,), dtype="int"),
    )

    prop_variables = ICLandPropVariables(
        prop_owner=jnp.zeros((max_prop_count,), dtype="int"),
    )

    return ICLandState(
        mjx_data=icland_params.model,
        agent_variables=agent_variables,
        prop_variables=prop_variables
    )


def step(
    state: ICLandState, params: ICLandParams, action_batch: Tuple[ICLandAction, ...]
) -> Tuple[ICLandState, ICLandObservation, jax.Array]:
    pass
