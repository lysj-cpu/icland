"""Recreating Google DeepMind's XLand RL environment in JAX."""

import os
import shutil
import warnings

if shutil.which("nvidia-smi") is None:
    warnings.warn("Cannot communicate with GPU")
else:
    # N.B. These need to be before the mujoco imports
    # Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
    os.environ["MUJOCO_GL"] = "egl"

    # Tell XLA to use Triton GEMM, this can improve steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    # See: https://github.com/jax-ml/jax/issues/8916#issuecomment-1101113497
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .agent import collect_body_scene_info, create_agent, step_agent
from .constants import *
from .game_generator import generate_game
from .types import *


def sample(key: jax.Array) -> ICLandParams:
    """Sample a new set of environment parameters using 'key'.

    Returns:
        ICLandParams: Parameters for the ICLand environment.

        - mj_model: Mujoco model of the environment.
        - reward_function: Reward function for the environment.
        - agent_count: Number of agents in the environment.

    Examples:
        >>> from icland import sample
        >>> import jax
        >>> key = jax.random.key(42)
        >>> sample(key)
        ICLandParams(model=MjModel, reward_function=<function sample.<locals>.<lambda> at 0x...>, agent_count=1)
    """
    # Sample the number of agents in the environment
    agent_count: int = jax.random.randint(
        key, (), WORL_MIN_AGENT_COUNT, WORLD_MAX_AGENT_COUNT
    )

    # Create the Mujoco model
    specification = mujoco.MjSpec()

    # Add the ground plane
    specification.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.01],
        rgba=[1, 1, 1, 1],
    )

    # Add the agents
    for agent_id in range(agent_count):
        specification = create_agent(agent_id, [agent_id, 0, 0.5], specification)

    # Compile the Mujoco model
    mj_model: mujoco.MjModel = specification.compile()

    # Generate the reward function
    reward_function = generate_game(key, agent_count)

    return ICLandParams(mj_model, reward_function, agent_count)


def init(key: jax.Array, params: ICLandParams) -> ICLandState:
    """Initialize the environment state from params.

    Returns:
        ICLandState: State of the ICLand environment.

        - mjx_model: JAX-compatible Mujoco model.
        - mjx_data: JAX-compatible Mujoco data.
        - agent_data: Body and geometry IDs for agents.

    Examples:
        >>> from icland import sample, init
        >>> import jax
        >>> key = jax.random.key(42)
        >>> params = sample(key)
        >>> init(key, params)
        ICLandState(pipeline_state=PipelineState(...), observation=Array(...), reward=Array(...), done=Array(...), metrics={...}, info=ICLandInfo(...))
    """
    # Unpack params
    mj_model = params.model
    agent_count = params.agent_count
    mj_data: mujoco.MjData = mujoco.MjData(mj_model)

    # Put Mujoco model and data into JAX-compatible format
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    agent_components = collect_agent_components(mj_model, agent_count)

    pipeline_state = PipelineState(mjx_model, mjx_data, agent_components)

    return ICLandState(
        pipeline_state,
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        # jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        # jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        {},
        collect_body_scene_info(agent_components, mjx_data),
    )


# TODO: Original implementation had both mj_model and mjx_model - can we only use one?
def collect_agent_components(mj_model: mujoco.MjModel, agent_count: int) -> jnp.ndarray:
    """Collect object IDs for all agents."""
    agent_components = jnp.empty(
        (agent_count, AGENT_COMPONENT_IDS_DIM), dtype=jnp.int32
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
            [body_id, geom_id, dof_address]
        )

    return agent_components


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
        >>> forward_policy = jnp.array([1, 0, 0])
        >>> key = jax.random.key(42)
        >>> params = sample(key)
        >>> state = init(key, params)
        >>> step(key, state, params, forward_policy)
        ICLandState(pipeline_state=PipelineState(...), observation=Array(...), reward=Array(...), done=Array(...), metrics={...}, info=ICLandInfo(...))
    """
    # Unpack state
    pipeline_state = state.pipeline_state

    mjx_model = pipeline_state.mjx_model
    mjx_data = pipeline_state.mjx_data
    agent_components = pipeline_state.component_ids

    # Ensure actions are in the correct shape
    actions = actions.reshape(-1, AGENT_ACTION_SPACE_DIM)

    # Define a function to step a single agent
    def step_single_agent(
        carry: Tuple[MjxStateType, jax.Array],
        agent_components: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Tuple[MjxStateType, jax.Array], None]:
        mjx_data, action = carry

        agent_component, agent_index = agent_components

        #  Reshape agent_component to (1, 2) to match the expected shape of step_agent
        agent_component = agent_component.reshape(-1, AGENT_COMPONENT_IDS_DIM)

        # Step the agent
        mjx_data = step_agent(
            mjx_data, action[agent_index], agent_component[agent_index]
        )
        return (mjx_data, action), None

    # Use `jax.lax.scan` to iterate through agents and step each one
    (updated_data, _), _ = jax.lax.scan(
        step_single_agent,
        (mjx_data, actions),
        (agent_components, jnp.arange(agent_components.shape[0], dtype=jnp.int32)),
    )

    # Step the environment after applying all agent actions
    updated_data = mjx.step(mjx_model, updated_data)
    new_pipeline_state = PipelineState(mjx_model, updated_data, agent_components)

    # TO BE ADDED
    observations = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    rewards = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    done = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    # metrics = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    # infos = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    metrics: dict[str, jax.Array] = {}
    infos: dict[str, jax.Array] = collect_body_scene_info(
        agent_components, updated_data
    )

    return ICLandState(new_pipeline_state, observations, rewards, done, metrics, infos)
