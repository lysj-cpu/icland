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
        max_agent_count: Maximum number of agents in the environment.
        max_sphere_count: Maximum number of spheres in the environment.
        max_cube_count: Maximum number of cubes in the environment.
        no_props: Flag indicating whether props are disabled (1) or enabled (0).
        model: MJX model of the environment.
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

        Returns:
            A string with the key configuration parameters.

        Examples:
            >>> from icland.types import ICLandConfig
            >>> config = ICLandConfig(10, 10, 6, 1, 1, 1, 0, None)
            >>> repr(config)
            'ICLandConfig(max_world_width=10, max_world_depth=10, max_world_height=6, max_agent_count=1, max_sphere_count=1, max_cube_count=1, no_props=0)'
        """
        return f"ICLandConfig(max_world_width={self.max_world_width}, max_world_depth={self.max_world_depth}, max_world_height={self.max_world_height}, max_agent_count={self.max_agent_count}, max_sphere_count={self.max_sphere_count}, max_cube_count={self.max_cube_count}, no_props={self.no_props})"


class ICLandWorld(PyTreeNode):  # type: ignore[misc]
    """Object storing world information for the ICLand environment.

    Attributes:
        tilemap: World tilemap array representing the layout.
        max_world_width: Width of the world tilemap.
        max_world_depth: Depth of the world tilemap.
        max_world_height: Maximum level of the world (in terms of 3D height).
        cmap: Tile colormap of the world for rendering.
    """

    tilemap: jax.Array
    max_world_width: int
    max_world_depth: int
    max_world_height: int
    cmap: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandWorld object.

        Returns:
            A string with world dimensions and array shapes.
        """
        return f"ICLandWorld(tilemap.shape={self.tilemap.shape}, max_world_width={self.max_world_width}, max_world_depth={self.max_world_depth}, max_world_height={self.max_world_height}, cmap.shape={self.cmap.shape})"


class ICLandAgentInfo(PyTreeNode):  # type: ignore[misc]
    """Information about agents in the ICLand environment.

    Attributes:
        agent_count: Actual number of agents active in the environment.
        spawn_points: Array of spawn points for agents, shape (max_agent_count, 3).
        spawn_orientations: Array of spawn orientations for agents, shape (max_agent_count,).
        body_ids: Array of body IDs for agents in the MJX model.
        geom_ids: Array of geometry IDs for agents in the MJX model.
        dof_addresses: Array of degrees of freedom addresses for agents.
        colour: Array of RGB colors for agents, shape (max_agent_count, 3).
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

        Returns:
            A string with agent count and array shapes.

        Examples:
            >>> import jax.numpy as jnp
            >>> from icland.types import ICLandAgentInfo
            >>> agent_info = ICLandAgentInfo(
            ...     jnp.array(1),
            ...     jnp.array([[0.0, 0.0, 0.0]]),
            ...     jnp.array([0.0]),
            ...     jnp.array([0]),
            ...     jnp.array([0]),
            ...     jnp.array([0]),
            ...     jnp.array([[1.0, 0.0, 0.0]]),
            ... )
            >>> repr(agent_info)
            'ICLandAgentInfo(agent_count=1, spawn_points.shape=(1, 3), body_ids.shape=(1,), colour.shape=(1, 3))'
        """
        return f"ICLandAgentInfo(agent_count={self.agent_count}, spawn_points.shape={self.spawn_points.shape}, body_ids.shape={self.body_ids.shape}, colour.shape={self.colour.shape})"


class ICLandPropInfo(PyTreeNode):  # type: ignore[misc]
    """Information about props in the ICLand environment.

    Attributes:
        prop_count: Actual number of props active in the environment.
        spawn_points: Array of spawn points for props, shape (max_prop_count, 3).
        spawn_rotations: Array of spawn rotations for props as quaternions, shape (max_prop_count, 4).
        prop_types: Array of types for props (1=cube, 2=sphere).
        body_ids: Array of body IDs for props in the MJX model.
        geom_ids: Array of geometry IDs for props in the MJX model.
        dof_addresses: Array of degrees of freedom addresses for props.
        colour: Array of RGB colors for props, shape (max_prop_count, 3).
    """

    prop_count: jax.Array
    spawn_points: jax.Array
    spawn_rotations: jax.Array
    prop_types: jax.Array
    body_ids: jax.Array
    geom_ids: jax.Array
    dof_addresses: jax.Array
    colour: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandPropInfo object.

        Returns:
            A string with prop count, types, and array shapes.
        """
        return f"ICLandPropInfo(prop_count={self.prop_count}, prop_types.shape={self.prop_types.shape}, spawn_points.shape={self.spawn_points.shape}, body_ids.shape={self.body_ids.shape}, colour.shape={self.colour.shape})"


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        world: The ICLandWorld object defining the environment layout.
        agent_info: ICLandAgentInfo containing constant information about agents.
        prop_info: ICLandPropInfo containing constant information about props.
        reward_function: Reward function (or None if not specified).
        mjx_model: MJX model of the environment physics.
    """

    world: ICLandWorld
    agent_info: ICLandAgentInfo
    prop_info: ICLandPropInfo
    reward_function: int
    mjx_model: MjxModelType

    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object.

        Returns:
            A string with the world, agent_info, prop_info, and reward_function.
        """
        return f"ICLandParams(world={self.world}, agent_info={self.agent_info}, prop_info={self.prop_info}, reward_function={self.reward_function})"


class ICLandAgentVariables(PyTreeNode):  # type: ignore[misc]
    """Variables for agents in the ICLand environment that change during simulation.

    Attributes:
        pitch: Pitch angle of each agent, shape (max_agent_count,).
        time_of_tag: Tag status of each agent (1=tagged, 0=not tagged), shape (max_agent_count,).
    """

    pitch: jax.Array  # Shape (max_agent_count, )
    time_of_tag: jax.Array  # Shape (max_agent_count, )

    def __repr__(self) -> str:
        """Return a string representation of the ICLandAgentVariables object.

        Returns:
            A string with the pitch and time_of_tag array shapes.
        """
        return f"ICLandAgentVariables(pitch.shape={self.pitch.shape}, time_of_tag.shape={self.time_of_tag.shape})"


class ICLandPropVariables(PyTreeNode):  # type: ignore[misc]
    """Variables for props in the ICLand environment that change during simulation.

    Attributes:
        prop_owner: ID of agent grabbing each prop (0=not grabbed), shape (max_prop_count,).
        time_of_grab: Time of grab for each prop (0=not grabbed), shape (max_prop_count,).
    """

    prop_owner: jax.Array  # Shape (max_prop_count, )
    time_of_grab: jax.Array  # Shape (max_prop_count, )

    def __repr__(self) -> str:
        """Return a string representation of the ICLandPropVariables object.

        Returns:
            A string with the prop_owner array shape.
        """
        return f"ICLandPropVariables(prop_owner.shape={self.prop_owner.shape}, time_of_grab.shape={self.time_of_grab.shape})"


class ICLandObservation(PyTreeNode):  # type: ignore[misc]
    """Observation set for the ICLand environment.

    Attributes:
        render: Rendered frames for each agent's viewpoint,
                shape (max_agent_count, height, width, 3).
        is_grabbing: Boolean indicator for whether each agent is grabbing a prop.
        acceleration: Acceleration values for each agent, shape (max_agent_count, 3).
    """

    render: jax.Array
    is_grabbing: jax.Array
    acceleration: jax.Array

    def __repr__(self) -> str:
        """Return a string representation of the ICLandObservation object.

        Returns:
            A string with the render array shape and is_grabbing value.
        """
        return f"ICLandObservation(render.shape={self.render.shape}, is_grabbing={self.is_grabbing}, acceleration.shape={self.acceleration.shape})"


class ICLandState(PyTreeNode):  # type: ignore[misc]
    """Current state of the ICLand environment.

    Attributes:
        mjx_data: MJX physics state data.
        agent_variables: Variables for agents that change during simulation.
        prop_variables: Variables for props that change during simulation.
        observation: Agent observations of the environment.
        reward: Current reward values for all agents, shape (max_agent_count,).
    """

    mjx_data: MjxStateType
    agent_variables: ICLandAgentVariables
    prop_variables: ICLandPropVariables

    def __repr__(self) -> str:
        """Return a string representation of the ICLandState object.

        Returns:
            A string with mjx_data, agent_variables, prop_variables, and observation.
        """
        return f"ICLandState(mjx_data={self.mjx_data}, agent_variables={self.agent_variables}, prop_variables={self.prop_variables})"
