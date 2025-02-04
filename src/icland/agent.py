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

    This updated version no longer applies a rotational torque. Instead, it
    directly updates the agent's rotation in `qpos` (the hinge angle about z
    at index 3) using the rotation command provided by `action[3]`. This avoids
    rotational inertia. The translational movement is handled exactly as before.

    Args:
        mjx_data: The simulation data object.
        action: The action to be performed by the agent. The first two components
            define the local XY movement, and `action[3]` is an integer in {-1, 0, 1}
            representing rotation (anticlockwise for positive, clockwise for negative).
        agent_data: The body and geometry IDs of the agent, and the agent's DOF address.

    Returns:
        Updated simulation data object.
    """
    # --------------------------------------------------------------------------
    # (A) Determine local movement and rotate it to world frame
    # --------------------------------------------------------------------------
    # Extract agent info: body id, geometry id, and DOF address.
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
    ncon = mjx_data.ncon

    # Extract contact normals for the first ncon contacts.
    normals = jnp.where(
        (mjx_data.contact.geom[:ncon, 1] == geom_id)[:, None],
        mjx_data.contact.frame[:ncon, 0, :],
        -mjx_data.contact.frame[:ncon, 0, :],
    )

    # Compute the projection of the movement onto each contact plane.
    dots = jnp.einsum("i,ni->n", movement_direction, normals)
    slope_components = movement_direction - dots[:, None] * normals
    slope_mags = jnp.linalg.norm(slope_components, axis=1)

    # Determine valid collisions: a valid collision occurs if the agent's geom_id
    # appears in either geom, and the contact distance is negative (touching).
    is_agent_collision = jnp.logical_or(
        mjx_data.contact.geom[:ncon, 0] == geom_id,
        mjx_data.contact.geom[:ncon, 1] == geom_id,
    )
    is_touching = mjx_data.contact.dist[:ncon] < 0.0
    valid_mask = is_agent_collision & is_touching

    def collision_true(_: Any) -> Any:
        # Use argmax to pick the first valid collision.
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
    # (D) Directly update the agent's rotation in qpos (avoiding torque/inertia)
    # --------------------------------------------------------------------------
    # The rotation command is in action[3] (an int in {-1, 0, 1}). We update the
    # hinge angle (qpos[3]) directly, scaled by AGENT_ROTATION_SPEED.
    new_angle = mjx_data.qpos[3] - AGENT_ROTATION_SPEED * action[3]
    new_qpos = mjx_data.qpos.at[3].set(new_angle)

    # Since we are directly setting the rotation, we do not want any angular inertia.
    # We leave qfrc_applied unchanged (i.e. no torque is applied).
    new_qfrc_applied = mjx_data.qfrc_applied

    # --------------------------------------------------------------------------
    # (E) Clamp linear speed in the XY plane
    # --------------------------------------------------------------------------
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
    # (F) Clamp angular velocity about z (for other potential dynamics)
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
    # (H) Remove rotational inertia by zeroing the angular velocity about z.
    # --------------------------------------------------------------------------
    qvel_updated = qvel_updated.at[dof_address + 3].set(0.0)

    # --------------------------------------------------------------------------
    # Combine updates and return the new state
    # --------------------------------------------------------------------------
    return mjx_data.replace(
        xfrc_applied=new_xfrc_applied,
        qfrc_applied=new_qfrc_applied,
        qvel=qvel_updated,
        qpos=new_qpos,
    )
