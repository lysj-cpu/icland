"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

from typing import TypeAlias, Optional, Tuple, TypeVar
import mujoco
import jax.numpy as jnp

"""Type variables from external modules."""

MjxStateType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Data)
MjxModelType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Model)

"""Type aliases for ICLand project."""

ICLandParams: TypeAlias = Tuple[mujoco.MjModel, Optional[str], int]
# ICLandParams is a tuple containing:
#   - mujoco.MjModel: The Mujoco model.
#   - Optional[str]: An optional string parameter.
#   - int: An integer parameter.

ICLandState: TypeAlias = Tuple[MjxModelType, MjxStateType, jnp.ndarray]
# ICLandState is a tuple containing:
#   - MjxModelType: JAX-compatible Mujoco model.
#   - MjxStateType: JAX-compatible Mujoco data.
#   - jnp.ndarray: Array of body and geometry IDs for agents.
#     The first dimension represents the ID of the agent, and the next dimension
#     is used to index the body and geometry IDs for the agent.

ICLandActionSet: TypeAlias = jnp.ndarray
# ICLandActionSet is a JAX array representing the actions taken by each agent.
# The first dimension represents the ID of the agent, and the next 3 dimensions
# specify the action.
#
# Example for 2 agents:
#   agent ID | forwards/backwards | left/right | rotate left/right
#    0       | -1                 | 0          | 1
#    1       | 1                  | 1          | -1
