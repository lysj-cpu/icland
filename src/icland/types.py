"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

import inspect
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode

"""Type variables from external modules."""

MjxStateType: TypeAlias = mujoco.mjx._src.types.Data
MjxModelType: TypeAlias = mujoco.mjx._src.types.Model

"""Type aliases for ICLand project."""


class PipelineState(PyTreeNode):  # type: ignore[misc]
    """State of the ICLand environment.

    Attributes:
        mjx_model: JAX-compatible Mujoco model.
        mjx_data: JAX-compatible Mujoco data.
        component_ids: Array of body and geometry IDs for agents. (agent_count, [body_ids, geom_ids])
    """

    mjx_model: MjxModelType
    mjx_data: MjxStateType
    component_ids: jnp.ndarray

    def __repr__(self) -> str:
        """Return a string representation of the PipelineState object."""
        return f"PipelineState(mjx_model={type(self.mjx_model).__name__}, mjx_data={type(self.mjx_data).__name__}, component_ids={self.component_ids})"


class ICLandInfo(PyTreeNode):  # type: ignore[misc]
    """Information about the ICLand environment.

    Attributes:
        agent_positions: [[x, y, z]] of agent positions, indexed by agent's body ID.
        agent_velocities: [[x, y, z]] of agent velocities, indexed by agent's body ID.
        agent_rotations: Quat of agent rotations, indexed by agent's body ID.
    """

    agent_positions: jax.Array
    agent_velocities: jax.Array
    agent_rotations: jax.Array


class ICLandState(PyTreeNode):  # type: ignore[misc]
    """Information regarding the current step.

    Attributes:
        pipeline_state: State of the ICLand environment.
        observation: Observation of the environment.
        reward: Reward of the environment.
        done: Flag indicating if the episode is done.
        metrics: Dictionary of metrics for the environment.
        info: Dictionary of additional information.
    """

    pipeline_state: PipelineState
    obs: jax.Array
    data: ICLandInfo

    def __repr__(self) -> str:
        """Return a string representation of the ICLandState object."""
        return f"ICLandState(pipeline_state={self.pipeline_state}, observation={self.obs}, data={self.data})"


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        model: Mujoco model of the environment.
        reward_function: Reward function for the environment
        agent_count: Number of agents in the environment.
    """

    model: mujoco.MjModel
    reward_function: Callable[[ICLandInfo], jax.Array] | None
    agent_count: int

    # Without this, model is model=<mujoco._structs.MjModel object at 0x7b61fb18dc70>
    # For some arbitrary memory address. __repr__ provides cleaner output
    # for users and for testing.
    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object.

        Examples:
            >>> from icland.types import ICLandParams, ICLandState
            >>> import mujoco
            >>> import jax
            >>> mj_model = mujoco.MjModel.from_xml_string("<mujoco/>")
            >>> def example_reward_function(state: ICLandState) -> jax.Array:
            ...     return jax.numpy.array(0)
            >>> ICLandParams(mj_model, example_reward_function, 1)
            ICLandParams(model=MjModel, reward_function=example_reward_function(state: icland.types.ICLandState) -> jax.Array, agent_count=1)
            >>> ICLandParams(mj_model, lambda state: jax.numpy.array(0), 1)
            ICLandParams(model=MjModel, reward_function=lambda function(state), agent_count=1)
        """
        if (
            self.reward_function
            and hasattr(self.reward_function, "__name__")
            and self.reward_function.__name__ != "<lambda>"
        ):
            reward_function_name = self.reward_function.__name__
        else:
            reward_function_name = "lambda function"

        reward_function_signature = ""
        if self.reward_function is not None:
            reward_function_signature = str(inspect.signature(self.reward_function))

        return f"ICLandParams(model={type(self.model).__name__}, reward_function={reward_function_name}{reward_function_signature}, agent_count={self.agent_count})"
