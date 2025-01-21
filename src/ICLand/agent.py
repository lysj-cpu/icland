"""Main agent file."""

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from .constants import *


@staticmethod
@jax.jit
def step_agent(mjx_data, action, agent_ids):
    """This function updates the mjx_data for an agent."""
    # Extract object IDs
    body_id, geom_id = agent_ids

    # --------------------------------------------------------------------------
    # (A) Determine local movement and rotate it to world frame
    # --------------------------------------------------------------------------
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
    for nth_contact in range(mjx_data.ncon):
        normal = mjx_data.contact.frame[nth_contact][0]

        # Compute the projection onto the contact plane
        slope_component = (
            movement_direction - jnp.dot(movement_direction, normal) * normal
        )
        slope_mag = jnp.linalg.norm(slope_component)

        is_agent_collision = jnp.logical_or(
            mjx_data.contact.geom1[nth_contact] == geom_id,
            mjx_data.contact.geom2[nth_contact] == geom_id,
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
    mjx_data = mjx_data.replace(
        xfrc_applied=mjx_data.xfrc_applied.at[body_id, :3].set(movement_direction)
    )

    # --------------------------------------------------------------------------
    # (D) Apply rotation torque about z hinge
    # --------------------------------------------------------------------------
    rotation_torque = action[3] * jnp.pi
    mjx_data = mjx_data.replace(
        qfrc_applied=mjx_data.qfrc_applied.at[3].set(
            mjx_data.qfrc_applied[3] + rotation_torque
        )
    )

    # --------------------------------------------------------------------------
    # (E) Clamp linear speed in XY
    # --------------------------------------------------------------------------
    vel_2d = mjx_data.qvel[0:2]  # [vx, vy]
    speed = jnp.linalg.norm(vel_2d)

    scale = jnp.where(
        speed > AGENT_MAX_MOVEMENT_SPEED, AGENT_MAX_MOVEMENT_SPEED / speed, 1.0
    )
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:2].multiply(scale))

    # --------------------------------------------------------------------------
    # (F) Clamp angular velocity about z
    # --------------------------------------------------------------------------
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
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[jnp.array([0, 1, 3])].multiply(
            1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT
        )
    )

    return mjx_data
