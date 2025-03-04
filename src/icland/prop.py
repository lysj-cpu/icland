"""Prop generation code."""

from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import mujoco

from icland.constants import AGENT_GRAB_DURATION, AGENT_GRAB_RANGE
from icland.types import (
    ICLandAgentInfo,
    ICLandAgentVariables,
    ICLandPropInfo,
    ICLandPropVariables,
)


class PropType(Enum):
    """Enum for the prop type selection for Props."""

    NONE = 0  # Do not render.
    CUBE = 1
    SPHERE = 2

    def _to_geom(cls) -> mujoco.mjtGeom:
        prop_to_geom = {
            PropType.NONE.value: mujoco.mjtGeom.mjGEOM_BOX,
            PropType.CUBE.value: mujoco.mjtGeom.mjGEOM_BOX,
            PropType.SPHERE.value: mujoco.mjtGeom.mjGEOM_SPHERE,
        }
        return prop_to_geom[cls.value]


def create_prop(
    id: int, pos: jax.Array, spec: mujoco.MjSpec, type: PropType
) -> mujoco.MjSpec:
    """Create an prop in the physics environment.

    Args:
        id: The ID of the prop.
        pos: The initial position of the prop.
        spec: The Mujoco specification object.
        type: The integer value of the prop type enum

    Returns:
        The updated Mujoco specification object.
    """
    prop = spec.worldbody.add_body(
        name=f"prop{id}",
        pos=pos[:3],
    )

    prop.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    prop.add_geom(
        name=f"prop{id}_geom",
        type=type._to_geom(),
        size=[0.1, 0.1, 0.1],
        mass=1,
        material="default",
    )

    return spec


@jax.jit
def step_props(
    mjx_data: Any,
    mjx_model: Any,
    agents_data: ICLandAgentInfo,
    agent_variables: ICLandAgentVariables,
    prop_data: ICLandPropInfo,
    prop_variables: ICLandPropVariables,
) -> Any:
    """Step the props in the physics environment.

    Args:
        mjx_data: The Mujoco data object.
        mjx_model: The Mujoco model object.
        agents_data: The agent data object.
        agent_variables: The agent variables object.
        prop_data: The prop data object.
        prop_variables: The prop variables object.

    Returns:
        The updated Mujoco data object.
    """

    def compute_prop_force(
        owner_id: jax.Array,
        prop_body_id: jax.Array,
        time_of_grab: jax.Array,
        prop_dof_addr: jax.Array,
    ) -> jax.Array:
        """Compute the force applied to a prop.

        Args:
            owner_id: The ID of the agent holding the prop.
            prop_body_id: The body ID of the prop.
            time_of_grab: The time the prop was grabbed.
            prop_dof_addr: The degree-of-freedom address of the prop.

        Returns:
            The force applied to the prop.
        """

        # Compute the force if the prop is held (owner_id != -1)
        def held_force() -> jax.Array:
            # The prop is held by an agent; use the agent's index (owner_id)
            agent_idx = owner_id
            # Get the agent's degree-of-freedom address and compute yaw from qpos.
            agent_dof = agents_data.dof_addresses[agent_idx]
            pitch = agent_variables.pitch[agent_idx]
            yaw = mjx_data.qpos[agent_dof + 3]
            # Define an offset and rotate it into world coordinates.
            offset_local = jnp.array([0.1, 0, 0])
            rotated_offset = jnp.array(
                [
                    jnp.cos(yaw) * offset_local[0] - jnp.sin(yaw) * offset_local[1],
                    jnp.sin(yaw) * offset_local[0] + jnp.cos(yaw) * offset_local[1],
                    offset_local[2],
                ]
            )
            # Use the agent's position (not the prop's) as the ray origin.
            ray_origin = mjx_data.xpos[agents_data.body_ids[agent_idx]] + rotated_offset
            # Compute the ray direction from the agent's pitch and yaw.
            ray_direction = (
                jnp.array(
                    [
                        jnp.cos(pitch) * jnp.cos(yaw),
                        jnp.cos(pitch) * jnp.sin(yaw),
                        jnp.sin(pitch),
                    ]
                )
                * 0.5
            )
            # The target position for the prop.
            ray_end = ray_origin + ray_direction * AGENT_GRAB_RANGE

            # Get the prop's current position.
            current_pos = mjx_data.xpos[prop_body_id]
            # Extract the 3-element velocity using a dynamic slice.
            current_vel = jax.lax.dynamic_slice(mjx_data.qvel, (prop_dof_addr,), (3,))
            # PD controller gains.
            kp = 50.0
            kd = 15
            return (
                kp * (ray_end - current_pos) - kd * current_vel + jnp.array([0, 0, 15])
            )

        is_held = jnp.logical_and(
            owner_id != -1, mjx_data.time < time_of_grab + AGENT_GRAB_DURATION
        )

        # If the prop is not held (owner_id == -1), no force is applied.
        return jax.lax.cond(is_held, held_force, lambda: jnp.zeros(3))

    # Compute forces for all props using vectorized mapping.
    forces = jax.vmap(compute_prop_force)(
        prop_variables.prop_owner,
        prop_data.body_ids,
        prop_variables.time_of_grab,
        prop_data.dof_addresses,
    )

    # Update the applied forces for the props similarly to step_agents.
    new_xfrc_applied = mjx_data.xfrc_applied.at[prop_data.body_ids, :3].set(forces)

    # Create a new mjx_data instance with the updated xfrc_applied field.
    new_mjx_data = mjx_data.replace(xfrc_applied=new_xfrc_applied)

    return new_mjx_data
