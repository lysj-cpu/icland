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
    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    # metrics: jax.Array  # These were Dict[str, jax.Array] = struct.field(default_factory=dict) but we need to install flax
    # info: jax.Array  # These were Dict[str, Any] = struct.field(default_factory=dict) but we need to install flax
    metrics: dict[str, jax.Array]
    info: dict[str, jax.Array]


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        model: Mujoco model of the environment.
        reward_function: Reward function for the environment
        agent_count: Number of agents in the environment.
    """

    model: mujoco.MjModel
    reward_function: Callable[[ICLandState], jax.Array]
    agent_count: int

    # Without this, model is model=<mujoco._structs.MjModel object at 0x7b61fb18dc70>
    # For some arbitrary memory address. __repr__ provides cleaner output
    # for users and for testing.
    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object."""
        if (
            hasattr(self.reward_function, "__name__")
            and self.reward_function.__name__ != "<lambda>"
        ):
            reward_function_name = self.reward_function.__name__
        else:
            reward_function_name = "lambda function"

        reward_function_signature = str(inspect.signature(self.reward_function))

        return f"ICLandParams(model={type(self.model).__name__}, reward_function=<function sample.<locals>.<lambda> at 0x...>, agent_count={self.agent_count})"
