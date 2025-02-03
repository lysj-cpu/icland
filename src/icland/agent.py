"""This module contains functions for simulating agent behavior in a physics environment."""

from typing import Any

import jax
import jax.numpy as jnp

from .constants import *
from .types import *


@jax.jit
def step_agent(
    mjx_data: MjxStateType,
    action: jnp.ndarray,
    agent_data: jnp.ndarray,
) -> MjxStateType:
    """Perform a simulation step for the agent (optimized).

    Args:
        mjx_data: The simulation data object.
        action: The action to be performed by the agent.
        agent_data: The body and geometry IDs of the agent.

    Returns:
        Updated simulation data object.
    """
    # --------------------------------------------------------------------------
    # (A) Determine local movement and rotate it to world frame
    # --------------------------------------------------------------------------
    # Extract agent info: body id, geometry id, and dof address.
    body_id, geom_id, dof_address = agent_data

    # Extract the intended local movement (XY) from the action.
    local_movement = action[:2]

    # The hinge angle about z is at qpos[3]
    angle = mjx_data.qpos[3]
    # Compute cosine and sine of the angle.
    c, s = jnp.cos(angle), jnp.sin(angle)
    # Rotate the local movement directly into the world frame.
    world_dir = jnp.array(
        [
            c * local_movement[0] - s * local_movement[1],
            s * local_movement[0] + c * local_movement[1],
            0.0,
        ]
    )
    movement_direction = world_dir

    # --------------------------------------------------------------------------
    # (B) Adjust movement based on contacts (handle slopes)
    # --------------------------------------------------------------------------
    # Since the number of contacts is fixed, we vectorize the computation.
    ncon = mjx_data.ncon

    # Extract contact normals for the first ncon contacts.
    # Here, if the second geometry in the contact equals the agent's geom_id,
    # we keep the normal as is; otherwise, we flip its sign.
    normals = jnp.where(
        (mjx_data.contact.geom[:ncon, 1] == geom_id)[:, None],
        mjx_data.contact.frame[:ncon, 0, :],
        -mjx_data.contact.frame[:ncon, 0, :],
    )

    # Compute the projection of the movement onto each contact plane.
    dots = jnp.einsum("i,ni->n", movement_direction, normals)
    slope_components = movement_direction - dots[:, None] * normals
    slope_mags = jnp.linalg.norm(slope_components, axis=1)

    # Determine valid collisions:
    # A valid collision occurs if the agent's geom_id appears in either geom,
    # and the contact distance is negative (touching).
    is_agent_collision = jnp.logical_or(
        mjx_data.contact.geom[:ncon, 0] == geom_id,
        mjx_data.contact.geom[:ncon, 1] == geom_id,
    )
    is_touching = mjx_data.contact.dist[:ncon] < 0.0
    valid_mask = is_agent_collision & is_touching

    # If any valid collision is found, update movement_direction accordingly.
    # If the slope component is large (mag > 0.7), normalize it;
    # otherwise, set the movement direction to zero.
    def collision_true(_: Any) -> Any:
        # Use argmax to pick the first valid collision (safe since at most one is valid).
        idx = jnp.argmax(valid_mask)
        mag = slope_mags[idx]
        new_dir = jnp.where(
            mag > 0.7,
            slope_components[idx] / (mag + SMALL_VALUE),
            jnp.zeros_like(movement_direction),
        )
        return new_dir

    def collision_false(_: Any) -> Any:
        return movement_direction

    movement_direction = jax.lax.cond(
        jnp.any(valid_mask),
        collision_true,
        collision_false,
        operand=None,
    )

    # --------------------------------------------------------------------------
    # (C) Apply linear force (update xfrc_applied)
    # --------------------------------------------------------------------------
    new_xfrc_applied = mjx_data.xfrc_applied.at[body_id, :3].set(
        movement_direction * AGENT_DRIVING_FORCE
    )

    # --------------------------------------------------------------------------
    # (D) Apply rotation torque about the z hinge (update qfrc_applied)
    # --------------------------------------------------------------------------
    rotation_torque = action[3] * jnp.pi
    new_qfrc_applied = mjx_data.qfrc_applied.at[3].set(
        mjx_data.qfrc_applied[3] + rotation_torque
    )

    # --------------------------------------------------------------------------
    # (E) Clamp linear speed in the XY plane
    # --------------------------------------------------------------------------
    # Extract the agent's XY velocity from qvel.
    vel_2d = jax.lax.dynamic_slice(mjx_data.qvel, (dof_address,), (2,))
    speed = jnp.linalg.norm(vel_2d)
    scale = jnp.where(
        speed > AGENT_MAX_MOVEMENT_SPEED,
        AGENT_MAX_MOVEMENT_SPEED / speed,
        1.0,
    )
    new_vel_2d = scale * vel_2d
    qvel_updated = jax.lax.dynamic_update_slice(
        mjx_data.qvel, new_vel_2d, (dof_address,)
    )

    # --------------------------------------------------------------------------
    # (F) Clamp angular velocity about z
    # --------------------------------------------------------------------------
    omega = qvel_updated[dof_address + 3]
    new_omega = jnp.where(
        jnp.abs(omega) > AGENT_MAX_ROTATION_SPEED,
        jnp.sign(omega) * AGENT_MAX_ROTATION_SPEED,
        omega,
    )
    qvel_updated = qvel_updated.at[dof_address + 3].set(new_omega)

    # --------------------------------------------------------------------------
    # (G) Apply linear and rotational friction to the velocities
    # --------------------------------------------------------------------------
    indices = jnp.array([dof_address, dof_address + 1, dof_address + 3])
    qvel_updated = qvel_updated.at[indices].multiply(
        1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT
    )

    # --------------------------------------------------------------------------
    # Combine updates and return the new state
    # --------------------------------------------------------------------------
    return mjx_data.replace(
        xfrc_applied=new_xfrc_applied,
        qfrc_applied=new_qfrc_applied,
        qvel=qvel_updated,
    )
