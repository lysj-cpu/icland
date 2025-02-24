"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, TypeAlias

import jax
import mujoco
from jaxtyping import Array, Float
from mujoco.mjx._src.dataclasses import PyTreeNode

"""Type variables from external modules."""

# Replacing with `type` keyword breaks tests
# https://docs.astral.sh/ruff/rules/non-pep695-type-alias/
MjxStateType: TypeAlias = mujoco.mjx._src.types.Data  # noqa: UP040
MjxModelType: TypeAlias = mujoco.mjx._src.types.Model  # noqa: UP040

"""Type aliases for ICLand project."""


class PipelineState(PyTreeNode):  # type: ignore[misc]
    """State of the ICLand environment.

    Attributes:
        mjx_model: JAX-compatible Mujoco model.
        mjx_data: JAX-compatible Mujoco data.
        component_ids: Array of body and geometry IDs for agents. (agent_count, [body_ids, geom_ids])
        world: The world tilemap.
    """

    mjx_model: MjxModelType
    mjx_data: MjxStateType
    component_ids: jax.Array
    world: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the PipelineState object."""
        return f"PipelineState(mjx_model={type(self.mjx_model).__name__}, mjx_data={type(self.mjx_data).__name__}, component_ids={self.component_ids}, world=World(width: {self.world.shape[0]}, height: {self.world.shape[1]}))"


class Agent(PyTreeNode):  # type: ignore[misc]
    """Information about an agent in the ICLand environment."""

    position: Float[Array, "3"]
    velocity: Float[Array, "4"]
    rotation: Float[Array, "1"] | Float[Array, ""]


class Prop(PyTreeNode):  # type: ignore[misc]
    """Information about a prop in the ICLand environment."""

    centre_of_mass: Float[Array, "3"]


class ICLandInfo(PyTreeNode):  # type: ignore[misc]
    """Information about the ICLand environment.

    Attributes:
        agent_positions: [[x, y, z]] of agent positions, indexed by agent's body ID.
        agent_velocities: [[x, y, z]] of agent velocities, indexed by agent's body ID.
        agent_rotations: Quat of agent rotations, indexed by agent's body ID.
    """

    agents: list[Agent]
    # TODO: Initialise values of props
    props: list[Prop] = dataclasses.field(default_factory=list)

    # TODO: Add prop data


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
        world: World tilemap array.
        reward_function: Reward function for the environment
        agent_count: Number of agents in the environment.
        world_level: World level.
    """

    world: jax.Array
    reward_function: Callable[[ICLandInfo], jax.Array] | None
    agent_spawns: jax.Array
    # TODO(Iajedi): Prop spawns
    # prop_spawns: jax.Array
    world_level: int

    # Without this, model is model=<mujoco._structs.MjModel object at 0x7b61fb18dc70>
    # For some arbitrary memory address. __repr__ provides cleaner output
    # for users and for testing.
    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object.

        Examples:
            >>> from icland.types import ICLandParams, ICLandState
            >>> import jax
            >>> def example_reward_function(state: ICLandState) -> jax.Array:
            ...     return jax.numpy.array(0)
            >>> ICLandParams(jax.numpy.array([[[0, 0, 0, 1]]]), example_reward_function, jax.numpy.array([[0, 0, 0]]), 6)
            ICLandParams(world=ArrayImpl, reward_function=example_reward_function(state: icland.types.ICLandState) -> jax.Array, agent_spawns=[[0 0 0]], world_level=6)
            >>> ICLandParams(jax.numpy.array([[[0, 0, 0, 1]]]), lambda state: jax.numpy.array(0), jax.numpy.array([[0, 0, 0]]), 6)
            ICLandParams(world=ArrayImpl, reward_function=lambda function(state), agent_spawns=[[0 0 0]], world_level=6)
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

        return f"ICLandParams(world={type(self.world).__name__}, reward_function={reward_function_name}{reward_function_signature}, agent_spawns={self.agent_spawns}, world_level={self.world_level})"


class ICLandConfig(PyTreeNode):  # type: ignore[misc]
    """Global configuration object for the ICLand environment.

    Attributes:
        world_width: Width of the world tilemap.
        world_height: Height of the world tilemap.
        max_agent_count: Maximum of agents in the environment.
        prop_counts (TODO): Dictionary storing the number of each prop.
        max_world_level: Maximum level of the world (in terms of 3D height).
    """

    world_width: int
    world_height: int
    max_agent_count: int
    prop_counts: dict[Any, Any]
    max_world_level: int

    def __repr__(self) -> str:
        """Return a string representation of the ICLandConfig object.

        Examples:
            >>> from icland.types import ICLandConfig
            >>> import jax
            >>> ICLandConfig(10, 10, 1, {}, 6)
            ICLandConfig(width=10, height=10, max_agent_count=1, max_world_level=6)
        """
        return f"ICLandConfig(width={self.world_width}, height={self.world_height}, max_agent_count={self.max_agent_count}, max_world_level={self.max_world_level})"
