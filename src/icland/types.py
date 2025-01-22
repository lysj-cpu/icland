"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

from typing import Optional, TypeAlias

import jax
import mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode

"""Type variables from external modules."""

MjxStateType: TypeAlias = mujoco.mjx._src.types.Data
MjxModelType: TypeAlias = mujoco.mjx._src.types.Model

"""Type aliases for ICLand project."""


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    r"""\Parameters for the ICLand environment.

    Attributes:
        model: Mujoco model of the environment.
        game: Game string (placeholder, currently None).
        agent_count: Number of agents in the environment.
    """

    model: mujoco.MjModel
    game: Optional[str]
    agent_count: int


class AgentData(PyTreeNode):  # type: ignore[misc]
    r"""\Agent in the ICLand environment.

    Attributes:
        body_id: Body IDs of the agents.
        geom_id: Geometry IDs of the agents.
    """

    body_id: jax.Array
    geom_id: jax.Array


class ICLandState(PyTreeNode):  # type: ignore[misc]
    r"""\State of the ICLand environment.

    Attributes:
        mjx_model: JAX-compatible Mujoco model.
        mjx_data: JAX-compatible Mujoco data.
        object_ids: Array of body and geometry IDs for agents.
    """

    mjx_model: MjxModelType
    mjx_data: MjxStateType
    agent_data: AgentData


class ICLandActionSet(PyTreeNode):  # type: ignore[misc]
    r"""\Actions taken by agents in the ICLand environment.

    Attributes:
        actions: Array representing the actions taken by each agent.

    Example for 2 agents:
     agent ID | forwards/backwards | left/right | rotate left/right
      0       | -1                 | 0          | 1
      1       | 1                  | 1          | -1
    """

    actions: jax.Array
