"""Recreating Google DeepMind's XLand RL environment in JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from icland.agent import step_agents
from icland.constants import *
from icland.prop import step_props
from icland.renderer.renderer import generate_colormap, render, select_random_color
from icland.types import *
from icland.world_gen.converter import sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.model_editing import edit_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES


def config(*args: int) -> ICLandConfig:
    """Smart constructor for ICLand config, initialising the base model on the fly.

    This function is the recommended way to create ICLandConfig objects.
    Direct instantiation of ICLandConfig should be avoided.

    Currently, due to the constraints imposed by the MuJoCo-XLA (MJX) engine,
    only spheres and cubes are supported.

    Args:
        *args: A tuple of integers representing the configuration parameters:
            - max_world_width: The maximum width of the world.
            - max_world_depth: The maximum depth of the world.
            - max_world_height: The maximum height of the world.
            - max_agent_count: The maximum number of agents.
            - max_sphere_count: The maximum number of spheres.
            - max_cube_count: The maximum number of cubes.

    Returns:
        An ICLandConfig object.

    Example:
        >>> config(5, 5, 6, 1, 0, 0)
        ICLandConfig(max_world_width=5, max_world_depth=5, max_world_height=6, max_agent_count=1, max_sphere_count=0, max_cube_count=0, no_props=True)
    """
    # Unpack the arguments
    model, _ = generate_base_model(*args)
    SPHERE_INDEX = 4
    CUBE_INDEX = 5
    no_props = args[SPHERE_INDEX] + args[CUBE_INDEX] == 0

    return ICLandConfig(*args, no_props=no_props, model=model)


# Default global configuration
DEFAULT_CONFIG = config(
    5,
    5,
    6,
    2,
    2,
    2,
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
        no_props,
        model,
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
    max_prop_count = max_sphere_count + max_cube_count + no_props
    max_object_count = max_agent_count + max_prop_count

    # Sample spawn points for objects
    spawnpoints = sample_spawn_points(
        key=key, tilemap=world_tilemap, num_objects=max_object_count
    )
    key, _ = jax.random.split(key)

    # Update with randomised number of agents
    num_agents = jax.random.randint(key, (), 1, max_agent_count + 1)

    # Update with randomised number of props
    key, s = jax.random.split(key)
    num_props = jax.random.randint(s, (), 0, max_prop_count + 1)
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
        jax.vmap(lambda i: (i < num_props) * spawnpoints[max_agent_count + i])(
            jnp.arange(max_prop_count)
        ),
        max_agent_count,
        axis=0,
    )

    colors = jnp.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    )
    key, s = jax.random.split(key)
    agent_colors = select_random_color(key, colors=colors, num_colors=max_agent_count)

    # Generate the agent and prop information
    agent_info = ICLandAgentInfo(
        agent_count=num_agents,
        spawn_points=spawnpoints[:max_agent_count],
        spawn_orientations=jnp.zeros((max_agent_count,), dtype="float32"),
        body_ids=jnp.arange(BODY_OFFSET, max_agent_count + BODY_OFFSET, dtype="int32"),
        geom_ids=(jnp.arange(max_agent_count) + max_world_width * max_world_depth) * 2
        + WALL_OFFSET,
        dof_addresses=jnp.arange(max_agent_count) * AGENT_DOF_OFFSET,
        colour=agent_colors,
    )

    # For prop types, from 0 -> max_cube_count - 1, set to 1
    # from max_cube_count -> max_cube_count + max_sphere_count - 1, set to 2
    prop_types = jax.vmap(lambda x: (x >= max_cube_count) + 1)(
        jnp.arange(max_prop_count)
    ) * (1 - no_props)  # If no props = 1, zero the array.

    prop_colors = select_random_color(s, colors=colors, num_colors=max_prop_count)

    # Generate the prop information
    prop_info = ICLandPropInfo(
        prop_count=num_props,
        prop_types=prop_types,
        spawn_points=spawnpoints[max_agent_count : max_agent_count + max_prop_count],
        spawn_rotations=jnp.zeros((max_prop_count, 4), dtype="float32"),
        body_ids=BODY_OFFSET
        + max_agent_count
        + jnp.arange(max_prop_count, dtype="int32"),
        geom_ids=(max_agent_count + max_world_width * max_world_depth) * 2
        + jnp.arange(max_prop_count, dtype="int32")
        + WALL_OFFSET,
        dof_addresses=jnp.arange(max_prop_count, dtype="int32") * PROP_DOF_MULTIPLIER
        + PROP_DOF_OFFSET,
        colour=prop_colors,
    )

    # Edit the model data for the specification
    model = edit_model_data(world_tilemap, model, agent_info, prop_info)
    cmap = generate_colormap(key, max_world_width, max_world_depth)

    icland_world = ICLandWorld(
        tilemap=world_tilemap,
        max_world_width=max_world_width,
        max_world_depth=max_world_depth,
        max_world_height=max_world_height,
        cmap=cmap,
    )

    # Return the parameters
    return ICLandParams(
        world=icland_world,
        agent_info=agent_info,
        prop_info=prop_info,
        reward_function=None,
        mjx_model=model,
    )


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
        ICLandState(mjx_data=Data(...), agent_variables=ICLandAgentVariables(...), prop_variables=ICLandPropVariables(...))
    """
    max_agent_count = icland_params.agent_info.spawn_points.shape[0]
    max_prop_count = icland_params.prop_info.spawn_points.shape[0]

    agent_variables = ICLandAgentVariables(
        pitch=jnp.zeros((max_agent_count,), dtype="float32"),
        time_of_tag=jnp.full((max_agent_count,), -AGENT_TAG_SECS_OUT, dtype="float32"),
    )

    prop_variables = ICLandPropVariables(
        prop_owner=-jnp.ones((max_prop_count,), dtype="int32"),
        time_of_grab=jnp.full((max_prop_count,), -AGENT_GRAB_DURATION, dtype="float32"),
    )

    return ICLandState(
        mjx_data=mjx.make_data(icland_params.mjx_model),
        agent_variables=agent_variables,
        prop_variables=prop_variables,
    )


@jax.jit
def step(
    state: ICLandState, params: ICLandParams, action_batch: jax.Array
) -> tuple[ICLandState, ICLandObservation, jax.Array]:
    """Step the ICLand environment forward in time.

    Args:
        state: The current state of the ICLand environment.
        params: The parameters of the ICLand environment.
        action_batch: The batch of actions to apply to the agents.
                      Should be of shape (num_agents, ACTION_SPACE_DIM).

    Returns:
        A tuple containing the new state (with observation), and reward.

    Example:
        >>> import icland
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(42)
        >>> params = icland.sample(key)
        >>> state = icland.init(params)
        >>> actions = jnp.zeros((2, icland.constants.ACTION_SPACE_DIM))
        >>> new_state, obs, rew = icland.step(state, params, actions)
        >>> new_state
        ICLandState(mjx_data=Data(...), agent_variables=ICLandAgentVariables(...), prop_variables=ICLandPropVariables(...))
    """
    # Unpack state
    mjx_data = state.mjx_data
    agent_variables = state.agent_variables
    prop_variables = state.prop_variables

    # Unpack params
    mjx_model = params.mjx_model
    agent_info = params.agent_info
    prop_info = params.prop_info
    world = params.world

    # Ensure parameters are in correct shape
    action_batch = action_batch.reshape(-1, ACTION_SPACE_DIM)

    def physics_step(
        carry: tuple[MjxStateType, ICLandAgentVariables, ICLandPropVariables], _: Any
    ) -> tuple[tuple[MjxStateType, ICLandAgentVariables, ICLandPropVariables], None]:
        mjx_data, agent_variables, prop_variables = carry
        # Step through each agent
        mjx_data, agent_variables, prop_variables = step_agents(
            mjx_data,
            mjx_model,
            action_batch,
            agent_info,
            agent_variables,
            prop_info,
            prop_variables,
        )
        # Steo through props
        mjx_data = step_props(
            mjx_data, mjx_model, agent_info, agent_variables, prop_info, prop_variables
        )
        # Update state
        mjx_data = mjx.step(params.mjx_model, mjx_data)
        return (mjx_data, agent_variables, prop_variables), None

    # Scan over PHYS_PER_CTRL_STEP iterations; we ignore the output per iteration (_)
    (mjx_data, agent_variables, prop_variables), _ = jax.lax.scan(
        physics_step,
        (mjx_data, agent_variables, prop_variables),
        None,
        length=PHYS_PER_CTRL_STEP,
    )

    # Evaluate reward function
    reward = jnp.zeros(
        (agent_info.spawn_points.shape[0],)
    )  # params.reward_function(state)

    # Update observation and render frames
    frames = render(
        agent_info,
        agent_variables,
        prop_info,
        prop_variables,
        action_batch,
        world,
        mjx_data,
    )

    observation = ICLandObservation(
        frames,
        is_grabbing=jnp.isin(
            jnp.arange(agent_variables.time_of_tag.shape[0]), prop_variables.prop_owner
        ).astype(jnp.int32),
        acceleration=jax.vmap(
            lambda dof: jax.lax.dynamic_slice(mjx_data.qacc, (dof,), (3,))
        )(agent_info.dof_addresses),
    )

    new_icland_state = ICLandState(
        mjx_data=mjx_data,
        agent_variables=agent_variables,
        prop_variables=prop_variables,
    )

    return new_icland_state, observation, reward
