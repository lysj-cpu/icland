"""This module contains functions for simulating agent behavior in a physics environment."""

from typing import Any

import jax
import jax.numpy as jnp
import mujoco

from .constants import *
from .types import *


def create_agent(
    id: int, pos: jax.Array, specification: mujoco.MjSpec
) -> mujoco.MjSpec:
    """Create an agent in the physics environment.

    Args:
        id: The ID of the agent.
        pos: The initial position of the agent.
        specification: The Mujoco specification object.

    Returns:
        The updated Mujoco specification object.
    """
    # Define the agent's body.
    agent = specification.worldbody.add_body(
        name=f"agent{id}",
        pos=pos[:3],
    )

    # Add transformational freedom.
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])

    # Add rotational freedom.
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

    # Add agent's geometry.
    agent.add_geom(
        name=f"agent{id}_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.06, 0.06, 0.06],
        fromto=[0, 0, 0, 0, 0, -AGENT_HEIGHT],
        mass=1,
    )

    # This is just to make rotation visible.
    agent.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], pos=[0, 0, 0.2], mass=0
    )

    return specification


@jax.jit
def step_agents(
    mjx_data: MjxStateType,
    actions: jax.Array,
    agents_data: ICLandAgentInfo,
    agent_variables: ICLandAgentVariables,
) -> tuple[MjxStateType, jax.Array]:
    """Update the agents in the physics environment based on the provided actions.

    Returns:
        Modified mjx_data and agents_data.
    """
    # Precompute contact data once per simulation step.
    ncon = mjx_data.ncon
    contact_geom = mjx_data.contact.geom[:ncon]  # Shape: (ncon, 2)
    contact_frame = mjx_data.contact.frame[:ncon, 0, :]  # Shape: (ncon, 3)
    contact_dist = mjx_data.contact.dist[:ncon]  # Shape: (ncon,)

    # Precompute friction factor.
    movement_friction = 1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT

    def agent_update(
        body_id: jax.Array,
        geom_id: jax.Array,
        dof: jax.Array,
        pitch: jax.Array,
        action: jax.Array,
        contact_geom: jax.Array,
        contact_frame: jax.Array,
        contact_dist: jax.Array,
    ) -> tuple[
        Any,  # body_id
        Any,  # dof
        Any,  # new_angle
        Any,  # new_vel_2d
        Any,  # new_omega
        Any,  # force
        Any,  # new_pitch
    ]:
        # (A) Determine local movement and rotate it to world frame.
        local_movement = action[:2]
        angle = mjx_data.qpos[dof + 3]
        c, s = jnp.cos(angle), jnp.sin(angle)
        world_dir = jnp.stack(
            [
                c * local_movement[0] - s * local_movement[1],
                s * local_movement[0] + c * local_movement[1],
                0.0,
            ]
        )
        movement_direction = world_dir

        # (B) Adjust movement based on contacts.
        sign = 2 * (contact_geom[:, 1] == geom_id) - 1  # +1 if true, -1 otherwise.
        normals = contact_frame * sign[:, None]

        dots = normals @ movement_direction
        slope_components = movement_direction - dots[:, None] * normals
        slope_mags = jnp.sqrt(jnp.sum(slope_components**2, axis=1))

        is_collision = jnp.logical_or(
            contact_geom[:, 0] == geom_id,
            contact_geom[:, 1] == geom_id,
        )
        is_touching = contact_dist < 0.0
        valid_mask = is_collision & is_touching

        def collision_true(_: Any) -> jnp.ndarray:
            idx = jnp.argmax(valid_mask)
            mag = slope_mags[idx]
            new_dir = jnp.where(
                mag > AGENT_MAX_CLIMBABLE_STEEPNESS,
                slope_components[idx] / (mag + SMALL_VALUE),
                jnp.zeros_like(movement_direction),
            )
            return new_dir

        movement_direction = jax.lax.cond(
            jnp.any(valid_mask),
            collision_true,
            lambda _: movement_direction,
            operand=None,
        )

        # (C) Compute force and update rotation.
        force = movement_direction * AGENT_DRIVING_FORCE
        new_angle = angle - AGENT_ROTATION_SPEED * action[2]

        # (D) Update and clamp translational velocity.
        vel_2d = jax.lax.dynamic_slice(mjx_data.qvel, (dof,), (2,))
        speed = jnp.sqrt(jnp.sum(vel_2d**2))
        scale = jnp.where(
            speed > AGENT_MAX_MOVEMENT_SPEED, AGENT_MAX_MOVEMENT_SPEED / speed, 1.0
        )
        new_vel_2d = vel_2d * scale * movement_friction

        # Angular velocity is set to zero.
        new_omega = 0.0

        # (E) Update pitch:
        # If action[3] is positive, increase pitch but do not exceed pi/2.
        # If action[3] is negative, decrease pitch but do not fall below -pi/2.
        new_pitch = jnp.clip(
            pitch + action[3] * AGENT_PITCH_SPEED, -jnp.pi / 2, jnp.pi / 2
        )

        return body_id, dof, new_angle, new_vel_2d, new_omega, force, new_pitch

    # Vectorize the per-agent update.
    (body_ids, dofs, new_angles, new_vels, new_omegas, forces, new_pitches) = jax.vmap(
        agent_update, in_axes=(0, 0, 0, 0, 0, None, None, None)
    )(
        agents_data.body_ids,
        agents_data.geom_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions,
        contact_geom,
        contact_frame,
        contact_dist,
    )

    # Combine per-agent updates into new simulation arrays.
    new_xfrc_applied = mjx_data.xfrc_applied.at[body_ids, :3].set(forces)
    new_qpos = mjx_data.qpos.at[dofs + 3].set(new_angles)

    new_qvel = mjx_data.qvel
    new_qvel = new_qvel.at[dofs].set(new_vels[:, 0])
    new_qvel = new_qvel.at[dofs + 1].set(new_vels[:, 1])
    new_qvel = new_qvel.at[dofs + 3].set(new_omegas)

    new_mjx_data = mjx_data.replace(
        xfrc_applied=new_xfrc_applied,
        qpos=new_qpos,
        qvel=new_qvel,
        qfrc_applied=mjx_data.qfrc_applied,
    )

    # Update the agents_data with the new pitch.
    new_agents_variables = ICLandAgentVariables(
        pitch=new_pitches,
        is_tagged=agent_variables.is_tagged,
    )

    return new_mjx_data, new_agents_variables
