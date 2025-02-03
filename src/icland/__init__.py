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


from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .agent import step_agent
from .constants import *
from .types import *

TEST_XML_STRING: str = """
<mujoco>
  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="agent0" pos="0 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent0_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.001"
      />
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom type="box" size="0.5 1 1" pos="0.45 2 -0.2" euler="0 -5 0"
          rgba="1 0.8 0.8 1"
          />
    <geom type="box" size="0.5 1 1" rgba="1 0.8 0.8 1"
          pos="1.5 0 0.1" euler="0 45 90"
          />
  </worldbody>
</mujoco>
"""


def sample(key: jax.Array) -> ICLandParams:
    """Sample a new set of environment parameters using 'key'.

    Returns:
        ICLandParams: Parameters for the ICLand environment.

        - mj_model: Mujoco model of the environment.
        - game: Game string (placeholder, currently None).
        - agent_count: Number of agents in the environment.

    Examples:
        >>> from icland import sample
        >>> import jax
        >>> key = jax.random.key(42)
        >>> sample(key)
        ICLandParams(model=MjModel, game=None, agent_count=1)
    """
    mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_string(TEST_XML_STRING)
    return ICLandParams(mj_model, None, 1)


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
        ICLandState(pipeline_state=PipelineState(...), observation=Array(...), reward=Array(...), done=Array(...), metrics=Array(...), info=Array(...))
    """
    # Unpack params
    mj_model = params.model
    agent_count = params.agent_count
    mj_data: mujoco.MjData = mujoco.MjData(mj_model)

    # Put Mujoco model and data into JAX-compatible format
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Collect object IDs for all agents
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

        dof_address = mjx_model.body_dofadr[body_id]

        agent_components = agent_components.at[agent_id].set(
            [body_id, geom_id, dof_address]
        )

    pipeline_state = PipelineState(mjx_model, mjx_data, agent_components)

    return ICLandState(
        pipeline_state,
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
        jnp.zeros((agent_count, AGENT_OBSERVATION_DIM)),
    )


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
        ICLandState(pipeline_state=PipelineState(...), observation=Array(...), reward=Array(...), done=Array(...), metrics=Array(...), info=Array(...))
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
    metrics = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))
    infos = jnp.zeros((agent_components.shape[0], AGENT_OBSERVATION_DIM))

    return ICLandState(new_pipeline_state, observations, rewards, done, metrics, infos)
