"""This module contains functions for simulating agent behavior in a physics environment."""

import jax
import jax.numpy as jnp

from .constants import *
from .types import *


@jax.jit
def step_agent(
    mjx_data: MjxStateType, action: jnp.ndarray, agent_data: jnp.ndarray
) -> MjxStateType:
    """Perform a simulation step for the agent.

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
    """
    Calculate the movement direction in the world frame based on the action and
    the agent's current orientation.
    """
    movement_direction = action[:2]

    # Hinge angle about z is in qpos[3]
    angle = mjx_data.qpos[3]

    # 2D rotation matrix for the angle
    R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])

    # Transform local movement direction to world frame
    world_dir = R @ movement_direction

    # Our force is in the XY plane only
    movement_direction = jnp.array([world_dir[0], world_dir[1], 0])

    # --------------------------------------------------------------------------
    # (B) Optionally check contacts to handle slopes
    # --------------------------------------------------------------------------
    """
    Adjust the movement direction if the agent is in contact with a slope that
    is too steep.
    """
    for nth_contact in range(mjx_data.ncon):
        normal = mjx_data.contact.frame[nth_contact][0]

        # Compute the projection onto the contact plane
        slope_component = (
            movement_direction - jnp.dot(movement_direction, normal) * normal
        )
        slope_mag = jnp.linalg.norm(slope_component)

        is_agent_collision = jnp.logical_or(
            mjx_data.contact.geom1[nth_contact] == agent_data.geom_id,
            mjx_data.contact.geom2[nth_contact] == agent_data.geom_id,
        )

        is_touching = mjx_data.contact.dist[nth_contact] < 0.0

        valid_collision = jnp.logical_and(is_agent_collision, is_touching)

        # If slope is too steep (angle > ~45deg), remove that direction
        movement_direction = jnp.where(
            jnp.logical_and(valid_collision, slope_mag > 0.7),
            slope_component / (slope_mag + 1e-10),
            movement_direction,
        )

    # --------------------------------------------------------------------------
    # (C) Apply the linear force in xfrc_applied
    # --------------------------------------------------------------------------
    """
    Apply the calculated linear force to the agent.
    """
    mjx_data = mjx_data.replace(
        xfrc_applied=mjx_data.xfrc_applied.at[agent_data.body_id, :3].set(
            movement_direction
        )
    )

    # --------------------------------------------------------------------------
    # (D) Apply rotation torque about z hinge
    # --------------------------------------------------------------------------
    """
    Apply the rotation torque to the agent based on the action.
    """
    rotation_torque = action[3] * jnp.pi
    mjx_data = mjx_data.replace(
        qfrc_applied=mjx_data.qfrc_applied.at[3].set(
            mjx_data.qfrc_applied[3] + rotation_torque
        )
    )

    # --------------------------------------------------------------------------
    # (E) Clamp linear speed in XY
    # --------------------------------------------------------------------------
    """
    Clamp the agent's linear speed in the XY plane to the maximum allowed speed.
    """
    vel_2d = mjx_data.qvel[0:2]  # [vx, vy]
    speed = jnp.linalg.norm(vel_2d)

    scale = jnp.where(
        speed > AGENT_MAX_MOVEMENT_SPEED, AGENT_MAX_MOVEMENT_SPEED / speed, 1.0
    )
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:2].multiply(scale))

    # --------------------------------------------------------------------------
    # (F) Clamp angular velocity about z
    # --------------------------------------------------------------------------
    """
    Clamp the agent's angular velocity about the z-axis to the maximum allowed
    rotation speed.
    """
    omega = mjx_data.qvel[3]
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[3].set(
            jnp.where(
                abs(omega) > AGENT_MAX_ROTATION_SPEED,
                jnp.sign(omega) * AGENT_MAX_ROTATION_SPEED,
                mjx_data.qvel[3],
            )
        )
    )

    # --------------------------------------------------------------------------
    # (G) Apply linear and rotational friction
    # --------------------------------------------------------------------------
    """
    Apply friction to the agent's linear and rotational velocities.
    """
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[jnp.array([0, 1, 3])].multiply(
            1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT
        )
    )

    return mjx_data
