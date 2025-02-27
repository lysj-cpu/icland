"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

from typing import TypeAlias

import jax
import mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode

"""Type variables from external modules."""

# Replacing with `type` keyword breaks tests
# https://docs.astral.sh/ruff/rules/non-pep695-type-alias/
MjxStateType: TypeAlias = mujoco.mjx._src.types.Data  # noqa: UP040
MjxModelType: TypeAlias = mujoco.mjx._src.types.Model  # noqa: UP040

"""Type aliases for ICLand project."""


class ICLandConfig(PyTreeNode):  # type: ignore[misc]
    """Global configuration object for the ICLand environment.

    Attributes:
        max_world_width: Width of the world tilemap.
        max_world_depth: Depth of the world tilemap.
        max_world_height: Maximum level of the world (in terms of 3D height).
        max_agent_count: Maximum of agents in the environment.
        max_sphere_count: Maximum number of spheres in the environment.
        max_cube_count: Maximum number of cubes in the environment.
    """

    max_world_width: int
    max_world_depth: int
    max_world_height: int
    max_agent_count: int
    max_sphere_count: int
    max_cube_count: int
    no_props: int
    model: MjxModelType

    def __repr__(self) -> str:
        """Return a string representation of the ICLandConfig object.

        Examples:
            >>> from icland.types import ICLandConfig
            >>> ICLandConfig(10, 10, 6, 1, 1, 1)
            ICLandConfig(max_world_width=10, max_world_depth=10, max_world_height=6, max_agent_count=1, max_sphere_count=1, max_cube_count=1)
        """
        return f"ICLandConfig(max_world_width={self.max_world_width}, max_world_depth={self.max_world_depth}, max_world_height={self.max_world_height}, max_agent_count={self.max_agent_count}, max_sphere_count={self.max_sphere_count}, max_cube_count={self.max_cube_count})"


class ICLandWorld(PyTreeNode):
    """Object storing world information for the ICLand environment.

    Attributes:
        tilemap: World tilemap array.
        max_world_width: Width of the world tilemap.
        max_world_depth: Depth of the world tilemap.
        max_world_height: Maximum level of the world (in terms of 3D height).
    """

    tilemap: jax.Array
    max_world_width: int
    max_world_depth: int
    max_world_height: int
    cmap: jax.Array


class ICLandAgentInfo(PyTreeNode):  # type: ignore[misc]
    """Information about agents in the ICLand environment.

    Attributes:
        agent_count: Actual number of agents.
        spawn_points: Array of spawn points for agents.
        spawn_orientations: Array of spawn orientations for agents.
        body_ids: Array of body IDs for agents.
        geom_ids: Array of geometry IDs for agents.
        dof_addresses: Array of degrees of freedom for agents.
        colour: Colour of agents.
    """

    agent_count: jax.Array
    spawn_points: jax.Array
    spawn_orientations: jax.Array
    body_ids: jax.Array
    geom_ids: jax.Array
    dof_addresses: jax.Array
    colour: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandAgentInfo object.

        Examples:
            >>> from icland.types import ICLandAgentInfo
            >>> ICLandAgentInfo(jax.array([[0.0, 0.0, 0.0]]), jax.array([0.0]), jax.array([0]), jax.array([0]), jax.array([0]), jax.array([0]))
            ICLandAgentInfo(spawn_points=DeviceArray([[0. 0. 0.]]), spawn_orientations=DeviceArray([0.]), body_ids=DeviceArray([0]), geom_ids=DeviceArray([0]), dof_addresses=DeviceArray([0]), colour=DeviceArray([0]))
        """
        return f"ICLandAgentInfo(spawn_points={self.spawn_points}, spawn_orientations={self.spawn_orientations}, body_ids={self.body_ids}, geom_ids={self.geom_ids}, dof_addresses={self.dof_addresses}, colour={self.colour})"


class ICLandPropInfo(PyTreeNode):
    """Information about props in the ICLand environment.

    Attributes:
        prop_count: Actual number of props.
        spawn_points: Array of spawn points for props.
        spawn_rotations: Array of spawn rotations for props:
        prop_types: Array of types for props.
        body_ids: Array of body IDs for props.
        geom_ids: Array of geometry IDs for props.
        dof_addresses: Array of degrees of freedom for props.
        colour: Colour of props.
    """

    prop_count: jax.Array
    spawn_points: jax.Array
    spawn_rotations: jax.Array
    prop_types: jax.Array
    body_ids: jax.Array
    geom_ids: jax.Array
    dof_addresses: jax.Array
    colour: jax.Array


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        world: Define the World.
        agent_info: Hold constant information about agents.
            This is a stacked array of  []
        prop_info: Hold constant information about props.
        reward_function: Reward function.
    """

    world: ICLandWorld
    agent_info: ICLandAgentInfo
    prop_info: ICLandPropInfo
    reward_function: int
    mjx_model: MjxModelType

    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object."""
        return f"ICLandParams(world={self.world}, agent_info={self.agent_info}, prop_info={self.prop_info}, reward_function={self.reward_function})"


class ICLandAgentVariables(PyTreeNode):  # type: ignore[misc]
    """Variables for agents in the ICLand environment.

    Attributes:
        pitch: Pitch of the agent.
        is_tagged: Tag status of the agent.
    """

    pitch: jax.Array  # Shape (max_agent_count, )
    is_tagged: jax.Array  # Shape (max_agent_count, )

    def __repr__(self) -> str:
        """Return a string representation of the ICLandAgentVariables object."""
        return f"ICLandAgentVariables(pitch={self.pitch}, is_tagged={self.is_tagged})"


class ICLandPropVariables(PyTreeNode):  # type: ignore[misc]
    """Variables for props in the ICLand environment.

    Attributes:
        prop_owner: Grab status of the prop.
    """

    prop_owner: jax.Array  # Shape (max_prop_count, )

    def __repr__(self) -> str:
        """Return a string representation of the ICLandPropVariables object."""
        return f"ICLandPropVariables(prop_owner={self.prop_owner})"


class ICLandObservation(PyTreeNode):  # type: ignore[misc]
    """Observation set for the ICLand environment.

    Attributes:
        render: Render the environment.
        is_grabbing: Is agent grabbing prop.
    """

    render: jax.Array
    is_grabbing: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandObservation object."""
        return (
            f"ICLandObservation(render={self.render}, is_grabbing={self.is_grabbing})"
        )


class ICLandState(PyTreeNode):  # type: ignore[misc]
    """Information regarding the current step.

    Attributes:
        mjx_data: MJX data
        agent_variables: Variables for agents.
        prop_variables: Variables for props.
    """

    mjx_data: MjxStateType
    agent_variables: ICLandAgentVariables
    prop_variables: ICLandPropVariables
    observation: ICLandObservation
    reward: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandState object."""
        return f"ICLandState(mjx_data={self.mjx_data}, agent_variables={self.agent_variables}, prop_variables={self.prop_variables})"
