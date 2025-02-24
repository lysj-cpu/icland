"""Recreating Google DeepMind's XLand RL environment in JAX."""

# Enforce runtime type-checking.
# See: https://beartype.readthedocs.io/en/latest/api_claw/
# Allow lossy conversion of integers to floating-point numbers
# https://beartype.readthedocs.io/en/latest/api_decor/#beartype.BeartypeConf.is_pep484_tower
# beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
import jax
import jax.numpy as jnp
import mujoco

# from beartype import BeartypeConf
# from beartype.claw import beartype_this_package
from mujoco import mjx

from .agent import collect_body_scene_info, step_agents
from .constants import *

# from .game import generate_game
from .types import *
from .world_gen.converter import sample_spawn_points
from .world_gen.JITModel import export, sample_world
from .world_gen.model_editing import edit_model_data
from .world_gen.tile_data import TILECODES


def sample(key: jax.Array, config: ICLandConfig) -> ICLandParams:  # pragma: no cover
    """Sample a new set of environment parameters using 'key'.

    Returns:
        ICLandParams: Parameters for the ICLand environment.

        - mj_model: Mujoco model of the environment.
        - reward_function: Reward function for the environment.
        - agent_count: Number of agents in the environment.

    Examples:
        >>> from icland import sample
        >>> from icland.world_gen.model_editing import generate_base_model
        >>> import jax
        >>> key = jax.random.key(42)
        >>> config = ICLandConfig(1, 1, 1, {}, 6)
        >>> base_model, _ = generate_base_model(config)
        >>> sample(key, config)
        ICLandParams(world=ArrayImpl, reward_function=lambda function, agent_spawns=[[0.5 0.5 4. ]], world_level=6)
    """
    # Sample the number of agents in the environment
    # TODO(Iajedi): communicate with harens, add global config (max agent count, world width/height, etc.)
    agent_count = config.max_agent_count

    # Sample the world based on config, using the WFC model (for now)
    MAX_STEPS = 1000
    wfc_model = sample_world(
        config.world_width, config.world_height, MAX_STEPS, key, True, 1
    )
    world_tilemap = export(
        wfc_model, TILECODES, config.world_width, config.world_height
    )

    # Generate the reward function
    # TODO: Adapt for JIT-ted manner
    # reward_function = generate_game(key, agent_count)
    reward_function = None

    # Generate the spawn points for each object
    # TODO: Add prop spawns as well
    key, subkey = jax.random.split(key)
    num_objects = config.max_agent_count
    spawnpoints = sample_spawn_points(subkey, world_tilemap, num_objects)

    return ICLandParams(
        world_tilemap, reward_function, spawnpoints, config.max_world_level
    )


def init(key: jax.Array, params: ICLandParams, base_model: MjxModelType) -> ICLandState:
    """Initialize the environment state from params.

    Returns:
        ICLandState: State of the ICLand environment.

        - mjx_model: JAX-compatible Mujoco model.
        - mjx_data: JAX-compatible Mujoco data.
        - agent_data: Body and geometry IDs for agents.

    Examples:
        >>> from icland import sample, init
        >>> from icland.presets import DEFAULT_CONFIG
        >>> from icland.world_gen.model_editing import generate_base_model
        >>> import jax
        >>> base_model, _ = generate_base_model(DEFAULT_CONFIG)
        >>> key = jax.random.key(42)
        >>> params = sample(key, DEFAULT_CONFIG)
        >>> init(key, params, base_model) # doctest:+ELLIPSIS
        ICLandState(pipeline_state=PipelineState(mjx_model=Model, mjx_data=Data, component_ids=[[  1 205   0   0]], world=World(width: 10, height: 10)), observation=[0. 0. 0. 0.], data=ICLandInfo(agents=[Agent(...)], props=[]))
    """
    # Unpack params
    world_tilemap = params.world
    agent_count = params.agent_spawns.shape[0]

    # Put Mujoco model and data into JAX-compatible format
    mjx_model = edit_model_data(
        world_tilemap,
        base_model,
        params.agent_spawns,
        jnp.array([]),
        params.world_level,
    )
    mjx_data = mjx.make_data(mjx_model)

    agent_components = collect_agent_components_mjx(
        mjx_model, world_tilemap.shape[0], world_tilemap.shape[1], agent_count
    )
    pipeline_state = PipelineState(mjx_model, mjx_data, agent_components, world_tilemap)

    return ICLandState(
        pipeline_state,
        jnp.zeros(AGENT_OBSERVATION_DIM),
        collect_body_scene_info(agent_components, mjx_data),
    )


def __collect_agent_components(
    mj_model: mujoco.MjModel, agent_count: int
) -> jax.Array:  # pragma: no cover
    agent_components = jnp.empty(
        (agent_count, AGENT_COMPONENT_IDS_DIM), dtype=jnp.float16
    )

    for agent_id in range(agent_count):
        body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, f"agent{agent_id}"
        )

        geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"agent{agent_id}_geom"
        )

        dof_address = mj_model.body_dofadr[body_id]

        agent_components = agent_components.at[agent_id].set(
            [body_id, geom_id, dof_address, 0]
        )

    return agent_components


def __collect_prop_components(
    mj_model: mujoco.MjModel, prop_count: int
) -> jax.Array:  # pragma: no cover
    agent_components = jnp.empty((prop_count, AGENT_COMPONENT_IDS_DIM), dtype=jnp.int32)

    for prop_id in range(prop_count):
        print(f"prop{prop_id}", f"prop{prop_id}_geom")
        body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, f"prop{prop_id}"
        )

        geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"prop{prop_id}_geom"
        )

        dof_address = mj_model.body_dofadr[body_id]

        agent_components = agent_components.at[prop_id].set(
            [body_id, geom_id, dof_address]
        )

    return agent_components


def collect_agent_components_mjx(
    mjx_model: mjx.Model, width: int, height: int, agent_count: int
) -> jax.Array:
    """Collect object IDs for all agents in a GPU-optimized way using mjx.Model."""

    def get_components_aux(agent_id: jax.Array) -> jax.Array:
        body_id = agent_id + BODY_OFFSET
        geom_id = (agent_id + width * height) * 2 + WALL_OFFSET
        dof_address = agent_id * 4
        return jnp.array([body_id, geom_id, dof_address, 0])

    agent_components = jax.vmap(get_components_aux)(jnp.arange(agent_count)).astype(
        "int32"
    )

    return agent_components


def collect_prop_components_mjx(
    mjx_model: mjx.Model, width: int, height: int, prop_count: int, agent_count: int
) -> jax.Array:  # pragma: no cover
    """Collect prop components in a GPU-optimized way using mjx.Model."""

    def get_components_aux(prop_id: jax.Array) -> jax.Array:
        body_id = prop_id + BODY_OFFSET + agent_count
        geom_id = (
            agent_count * 2 + (prop_id + agent_count + width * height) + WALL_OFFSET
        )
        dof_address = PROP_DOF_OFFSET + prop_id * PROP_DOF_MULTIPLIER
        return jnp.array([body_id, geom_id, dof_address])

    prop_components = jax.vmap(get_components_aux)(jnp.arange(prop_count)).astype(
        "int32"
    )

    return prop_components


@jax.jit
def step(
    key: jax.Array,
    state: ICLandState,
    params: ICLandParams,
    actions: jax.Array,
) -> ICLandState:
    """Advance environment one step for all agents.

    Returns:
        ICLandState: State of the ICLand environment.

        - mjx_model: JAX-compatible Mujoco model.
        - mjx_data: JAX-compatible Mujoco data.
        - agent_data: Body and geometry IDs for agents.

    Examples:
        >>> from icland import sample, init, step
        >>> import jax
        >>> import jax.numpy as jnp
        >>> forward_policy = jnp.array([1, 0, 0, 0])
        >>> from icland.presets import DEFAULT_CONFIG
        >>> from icland.world_gen.model_editing import generate_base_model
        >>> base_model, _ = generate_base_model(DEFAULT_CONFIG)
        >>> key = jax.random.key(42)
        >>> params = sample(key, DEFAULT_CONFIG)
        >>> state = init(key, params, base_model)
        >>> step(key, state, params, forward_policy) # doctest:+ELLIPSIS
        ICLandState(pipeline_state=PipelineState(mjx_model=Model, mjx_data=Data, component_ids=[[  1 205   0   0]], world=World(width: 10, height: 10)), observation=[...], data=ICLandInfo(agents=[Agent(...)], props=[]))
    """
    # Unpack state
    pipeline_state = state.pipeline_state
    mjx_model = pipeline_state.mjx_model
    mjx_data = pipeline_state.mjx_data
    agent_components = pipeline_state.component_ids

    # Ensure actions are in the correct shape
    actions = actions.reshape(-1, AGENT_ACTION_SPACE_DIM)

    # Use vmap to step through each agent.
    updated_data, updated_agent_components = step_agents(
        mjx_data, actions, agent_components
    )

    # Step the environment after applying all agent actions
    updated_data = mjx.step(mjx_model, updated_data)
    new_pipeline_state = PipelineState(
        mjx_model, updated_data, agent_components, pipeline_state.world
    )
    data: ICLandInfo = collect_body_scene_info(agent_components, mjx_data)
    observation = updated_data.qpos

    return ICLandState(new_pipeline_state, observation, data)
