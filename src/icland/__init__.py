"""Recreating Google DeepMind's XLand RL environment in JAX."""

import os
import warnings
import shutil

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


import jax
import mujoco
from mujoco import mjx
from .agent import step_agent
import jax.numpy as jnp
from typing import Tuple, Optional, Any, TypeVar

MjxStateType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Data)
MjxModelType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Model)

TEST_XML_STRING: str = """
<mujoco>
  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="agent1" pos="0 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent1_geom"
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


def sample(key: jax.Array) -> Tuple[mujoco.MjModel, Optional[str], int]:
    """Sample a new set of environment parameters using 'key'.

    Returns a tuple containing:
    - mj_model: Mujoco model of the environment.
    - game: Game string (placeholder, currently None).
    - agent_count: Number of agents in the environment.
    """
    mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_string(TEST_XML_STRING)
    return (mj_model, None, 1)


def init(
    key: jax.Array, params: Tuple[mujoco.MjModel, Optional[str], int]
) -> Tuple[Any, Any, jnp.ndarray]:
    """Initialize the environment state from params.

    Returns a tuple containing:
    - mjx_model: JAX-compatible Mujoco model.
    - mjx_data: JAX-compatible Mujoco data.
    - object_ids: Array of body and geometry IDs for agents.
    """
    mj_model, game, agent_count = params
    mj_data: mujoco.MjData = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Collect object IDs for all agents
    object_ids = []
    for agent_id in range(agent_count):
        body_name = f"agent{agent_id + 1}"
        geom_name = f"agent{agent_id + 1}_geom"

        body_id: int = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id: int = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        object_ids.append([body_id, geom_id])

    return (mjx_model, mjx_data, jnp.array(object_ids))


@jax.jit
def step(
    key: jax.Array,
    state: Tuple[Any, Any, jnp.ndarray],
    params: Tuple[mujoco.MjModel, Optional[str], int],
    actions: jnp.ndarray,
) -> Tuple[Any, Any, jnp.ndarray]:
    """Advance environment one step for all agents.

    Returns the updated state containing:
    - mjx_model: Updated Mujoco model.
    - mjx_data: Updated Mujoco data.
    - object_ids: Array of body and geometry IDs for agents.
    """
    mjx_model, mjx_data, object_ids = state
    updated_data = mjx_data

    def step_single_agent(
        carry: Tuple[Any, Any], inputs: jnp.ndarray
    ) -> Tuple[Tuple[Any, Any], None]:
        mjx_data, action = carry
        body_id, geom_id = inputs
        mjx_data = step_agent(mjx_data, action, inputs)
        return (mjx_data, action), None

    # Use `jax.lax.scan` to iterate through agents and step each one
    (updated_data, _), _ = jax.lax.scan(
        step_single_agent, (updated_data, actions), object_ids
    )

    # Step the environment after applying all agent actions
    updated_data = mjx.step(mjx_model, updated_data)

    return (mjx_model, updated_data, object_ids)
