"""This module contains functions for simulating agent behavior in a physics environment."""

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from .constants import *
from .types import *

@jax.jit
def _agent_raycast(
    body_id: jax.Array,
    dof: jax.Array,
    pitch: jax.Array,
    action_value: jax.Array,
    mjx_data: Any,
    mjx_model: Any,
    offset_local: jax.Array = jnp.array([0.1, 0, 0]),
) -> jax.Array:
    trigger = action_value > 0.5
    yaw = mjx_data.qpos[dof + 3]
    ray_direction = jnp.array(
        [
            jnp.cos(pitch) * jnp.cos(yaw),
            jnp.cos(pitch) * jnp.sin(yaw),
            jnp.sin(pitch),
        ]
    )
    rotated_offset = jnp.array(
        [
            jnp.cos(yaw) * offset_local[0] - jnp.sin(yaw) * offset_local[1],
            jnp.sin(yaw) * offset_local[0] + jnp.cos(yaw) * offset_local[1],
            offset_local[2],
        ]
    )
    ray_origin = mjx_data.xpos[body_id] + rotated_offset
    raycast = mjx.ray(mjx_model, mjx_data, ray_origin, ray_direction)
    return jnp.where(jnp.logical_and(trigger, raycast[0] < AGENT_MAX_TAG_DISTANCE), raycast[1], -1)


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
        material="default",
    )

    # This is just to make rotation visible.
    agent.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], pos=[0, 0, 0.2], mass=0, material="default"
    )

    return specification


@jax.jit
def step_agents(
    mjx_data: Any,
    mjx_model: Any,
    actions: jax.Array,
    agents_data: Any,
    agent_variables: Any,
) -> tuple[Any, jax.Array]:
    """Update the agents in the physics environment based on the provided actions.

    Returns:
        Modified mjx_data and agents_variables.
    """
    # Precompute contact data once per simulation step.
    ncon = mjx_data.ncon
    contact_geom = mjx_data.contact.geom[:ncon]  # Shape: (ncon, 2)
    contact_frame = mjx_data.contact.frame[:ncon, 0, :]  # Shape: (ncon, 3)
    contact_dist = mjx_data.contact.dist[:ncon]  # Shape: (ncon,)

    # Precompute friction factor.
    movement_friction = 1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT

    # --- Tagging Raycast ---
    # Use the new helper to perform a raycast for tagging using action[4].
    # For future grab functionality, a similar call can be made with action[5] and an appropriate max distance.
    geom_raycasted_ids = jax.vmap(
        lambda body_id, dof, pitch, action_value: _agent_raycast(
            body_id,
            dof,
            pitch,
            action_value,
            mjx_data,
            mjx_model,
        ),
        in_axes=(0, 0, 0, 0),
    )(
        agents_data.body_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions[:, 4],
    )

    compute_index = lambda geom: jnp.argmax(agents_data.geom_ids == geom)
    indices = jax.vmap(compute_index)(geom_raycasted_ids)
    tagged_agent_geom_ids = jnp.where(jnp.isin(geom_raycasted_ids, agents_data.geom_ids), indices, -1)



    # Update time_of_tag for agents that were tagged.
    agent_variables = agent_variables.replace(
        time_of_tag=jnp.where(
            jnp.isin(jnp.arange(agent_variables.time_of_tag.shape[0]), tagged_agent_geom_ids),
            mjx_data.time,
            agent_variables.time_of_tag,
        )
    )

    def agent_update(
        body_id: jax.Array,
        geom_id: jax.Array,
        dof: jax.Array,
        pitch: jax.Array,
        action: jax.Array,
        time_of_tag: jax.Array,
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
        sign = 2 * (contact_geom[:, 1] == geom_id) - 1
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

        # (E) Update pitch.
        new_pitch = jnp.clip(
            pitch + action[3] * AGENT_PITCH_SPEED, -jnp.pi / 2, jnp.pi / 2
        )

        return body_id, dof, new_angle, new_vel_2d, new_omega, force, new_pitch

    (body_ids, dofs, new_angles, new_vels, new_omegas, forces, new_pitches) = jax.vmap(
        agent_update, in_axes=(0, 0, 0, 0, 0, 0, None, None, None)
    )(
        agents_data.body_ids,
        agents_data.geom_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions,
        agent_variables.time_of_tag,
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

    # --- Override state for tagged agents (freeze/resume) ---
    # We assume that for each agent, the state is stored in 4 contiguous entries in qpos and qvel:
    # [x, y, z, angle]. The dof_addresses indicate the starting index.
    n_agents = agents_data.body_ids.shape[0]
    # Create an array of indices (shape: [n_agents, 4]) for each agent’s state slice.
    dof_indices = dofs[:, None] + jnp.arange(4)

    # Define masks:
    freeze_mask = (agent_variables.time_of_tag != -AGENT_TAG_SECS_OUT) & (mjx_data.time < agent_variables.time_of_tag + AGENT_TAG_SECS_OUT)
    resume_mask = (agent_variables.time_of_tag != -AGENT_TAG_SECS_OUT) & (mjx_data.time >= agent_variables.time_of_tag + AGENT_TAG_SECS_OUT)

    # Override qpos:
    current_pos = jnp.take(new_qpos, dof_indices)  # shape: (n_agents, 4)
    # For frozen agents, set to [0, 0, 0, 0]. For resumed agents, set to [1, 1, 1, 0].
    resumed_override = jnp.concatenate([agents_data.spawn_points, agents_data.spawn_orientations[:, None]], axis=1)
    frozen_override = resumed_override.at[:, 2].set(-10)
    override_pos = jnp.where(freeze_mask[:, None], frozen_override, current_pos)
    override_pos = jnp.where(resume_mask[:, None], resumed_override, override_pos)
    new_qpos = new_qpos.at[dof_indices].set(override_pos)

    # Override qvel: set the entire 4–component state to 0 for agents being frozen/resumed.
    current_vel = jnp.take(new_qvel, dof_indices)
    override_vel = jnp.where((freeze_mask | resume_mask)[:, None], jnp.zeros((4,)), current_vel)
    new_qvel = new_qvel.at[dof_indices].set(override_vel)

    # Override applied forces for these agents (only the first 3 components).
    freeze_or_resume = freeze_mask | resume_mask
    forces_override = jnp.where(
        freeze_or_resume[:, None],
        jnp.zeros((n_agents, 3)),
        new_xfrc_applied[body_ids, :3],
    )
    new_xfrc_applied = new_xfrc_applied.at[body_ids, :3].set(forces_override)

    # For resumed agents, update the time_of_tag to -AGENT_TAG_SECS_OUT.
    new_time_of_tag = jnp.where(resume_mask, -AGENT_TAG_SECS_OUT, agent_variables.time_of_tag)
    new_agents_variables = ICLandAgentVariables(
        pitch=new_pitches,
        time_of_tag=new_time_of_tag,
    )

    new_mjx_data = mjx_data.replace(
        xfrc_applied=new_xfrc_applied,
        qpos=new_qpos,
        qvel=new_qvel,
        qfrc_applied=mjx_data.qfrc_applied,
    )

    return new_mjx_data, new_agents_variables
