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
    """Perform a raycast for an agent based on its orientation and action trigger.

    Args:
        body_id: ID of the agent's body.
        dof: Address of the agent's degree of freedom.
        pitch: The pitch angle of the agent.
        action_value: The action value that triggers the raycast.
        mjx_data: The Mujoco simulation data.
        mjx_model: The Mujoco simulation model.
        offset_local: Local offset from the body origin to start the ray.

    Returns:
        The distance (or tagged geometry ID) from the raycast or -1 if not triggered or out of range.
    """
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
    # Return raycast result if triggered and within maximum distance; otherwise, -1.
    return jnp.where(
        jnp.logical_and(trigger, raycast[0] < AGENT_MAX_TAG_DISTANCE), raycast[1], -1
    )


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

    # Add translational degrees of freedom.
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])

    # Add rotational freedom.
    agent.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

    # Add the primary geometry (capsule) for the agent.
    agent.add_geom(
        name=f"agent{id}_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.06, 0.06, 0.06],
        fromto=[0, 0, 0, 0, 0, -AGENT_HEIGHT],
        mass=1,
        material="default",
        friction=[0, 0, 0],
        solmix=100,
    )

    # Add a box geometry to help visualize rotation.
    agent.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05, 0.05, 0.05],
        pos=[0, 0, 0.2],
        mass=0,
        material="default",
    )

    return specification


@jax.jit
def step_agents(
    mjx_data: Any,
    mjx_model: Any,
    actions: jax.Array,
    agents_data: ICLandAgentInfo,
    agent_variables: ICLandAgentVariables,
    prop_data: ICLandPropInfo,
    prop_variables: ICLandPropVariables,
) -> tuple[Any, jax.Array, jax.Array]:
    """Update agents in the physics environment based on provided actions.

    Returns:
        A tuple of (updated mjx_data, updated agent_variables, updated prop_variables).
    """
    # --- Precompute contact and friction data ---
    ncon = mjx_data.ncon
    contact_geom = mjx_data.contact.geom[:ncon]  # shape: (ncon, 2)
    contact_frame = mjx_data.contact.frame[:ncon, 0, :]  # shape: (ncon, 3)
    contact_dist = mjx_data.contact.dist[:ncon]  # shape: (ncon,)
    movement_friction = 1.0 - AGENT_MOVEMENT_FRICTION_COEFFICIENT

    # --- Tagging Raycast for agents and props ---
    # --- Separate Raycasts for Tagging and Grabbing ---
    tag_geom_raycasted_ids = jax.vmap(
        lambda body_id, dof, pitch, act_val: _agent_raycast(
            body_id, dof, pitch, act_val, mjx_data, mjx_model
        ),
        in_axes=(0, 0, 0, 0),
    )(
        agents_data.body_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions[:, 4],  # Tag only if action[4] > 0.5
    )

    grab_geom_raycasted_ids = jax.vmap(
        lambda body_id, dof, pitch, act_val: _agent_raycast(
            body_id, dof, pitch, act_val, mjx_data, mjx_model
        ),
        in_axes=(0, 0, 0, 0),
    )(
        agents_data.body_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions[:, 5],  # Grab only if action[5] > 0.5
    )

    # --- Compute Valid Tag and Grab Results ---
    compute_agent_index = lambda geom: jnp.argmax(agents_data.geom_ids == geom)
    tagged_agent_geom_ids = jnp.where(
        jnp.isin(tag_geom_raycasted_ids, agents_data.geom_ids),
        jax.vmap(compute_agent_index)(tag_geom_raycasted_ids),  # type: ignore
        -1,
    )

    compute_prop_index = lambda geom: jnp.argmax(prop_data.geom_ids == geom)
    tagged_prop_geom_ids = jnp.where(
        jnp.isin(grab_geom_raycasted_ids, prop_data.geom_ids),
        jax.vmap(compute_prop_index)(grab_geom_raycasted_ids),  # type: ignore
        -1,
    )

    # --- Update Prop Ownership for Grabs ---
    new_prop_owner = jax.lax.fori_loop(
        0,
        tagged_prop_geom_ids.shape[0],
        lambda i, owner: jax.lax.cond(
            tagged_prop_geom_ids[i] != -1,
            lambda o: o.at[tagged_prop_geom_ids[i]].set(i),
            lambda o: o,
            owner,
        ),
        prop_variables.prop_owner,
    )

    new_time_of_grab = jnp.where(
        new_prop_owner != prop_variables.prop_owner,
        mjx_data.time,
        prop_variables.time_of_grab,
    )

    final_prop_owner = jnp.where(
        new_time_of_grab + AGENT_GRAB_DURATION < mjx_data.time,
        -1,
        new_prop_owner,
    )

    prop_variables = prop_variables.replace(
        prop_owner=final_prop_owner,
        time_of_grab=new_time_of_grab,
    )

    # --- Update Tag Time for Agents ---
    agent_variables = agent_variables.replace(
        time_of_tag=jnp.where(
            jnp.isin(
                jnp.arange(agent_variables.time_of_tag.shape[0]), tagged_agent_geom_ids
            ),
            mjx_data.time,
            agent_variables.time_of_tag,
        )
    )

    # --- Define per-agent update logic ---
    def agent_update(
        body_id: jax.Array,
        geom_id: jax.Array,
        dof: jax.Array,
        pitch: jax.Array,
        action: jax.Array,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        """Update a single agent's state based on its current state and action.

        Returns:
            Tuple containing updated state values.
        """
        # (A) Transform local movement to world direction.
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

        # (B) Adjust movement based on collision contacts.
        # Determine contact normals relative to the agent.
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

        def adjust_for_collision(_: Any) -> jnp.ndarray:
            # Use the first valid contact to adjust movement.
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
            adjust_for_collision,
            lambda _: movement_direction,
            operand=None,
        )

        # (C) Compute driving force and update rotational angle.
        force = movement_direction * AGENT_DRIVING_FORCE
        new_angle = angle - AGENT_ROTATION_SPEED * action[2] / PHYS_PER_CTRL_STEP

        # (D) Update translational velocity (with friction and speed limiting).
        vel_2d = jax.lax.dynamic_slice(mjx_data.qvel, (dof,), (2,))
        speed = jnp.sqrt(jnp.sum(vel_2d**2))
        scale = jnp.where(
            speed > AGENT_MAX_MOVEMENT_SPEED, AGENT_MAX_MOVEMENT_SPEED / speed, 1.0
        )
        new_vel_2d = vel_2d * scale * movement_friction

        # Set angular velocity to zero.
        new_omega = 0.0

        # (E) Update pitch with clamping.
        new_pitch = jnp.clip(
            pitch + action[3] * AGENT_PITCH_SPEED / PHYS_PER_CTRL_STEP,
            -jnp.pi / 2,
            jnp.pi / 2,
        )

        return body_id, dof, new_angle, new_vel_2d, new_omega, force, new_pitch

    # Vectorize agent updates across all agents.
    (body_ids, dofs, new_angles, new_vels, new_omegas, forces, new_pitches) = jax.vmap(
        agent_update, in_axes=(0, 0, 0, 0, 0)
    )(
        agents_data.body_ids,
        agents_data.geom_ids,
        agents_data.dof_addresses,
        agent_variables.pitch,
        actions,
    )

    # --- Combine updates into simulation state arrays ---
    new_xfrc_applied = mjx_data.xfrc_applied.at[body_ids, :3].set(forces)
    new_qpos = mjx_data.qpos.at[dofs + 3].set(new_angles)
    new_qvel = mjx_data.qvel
    new_qvel = new_qvel.at[dofs].set(new_vels[:, 0])
    new_qvel = new_qvel.at[dofs + 1].set(new_vels[:, 1])
    new_qvel = new_qvel.at[dofs + 3].set(new_omegas)

    # --- Override state for agents being tagged (freeze/resume) ---
    n_agents = agents_data.body_ids.shape[0]
    # Compute indices for each agent's 4-component state slice.
    dof_indices = dofs[:, None] + jnp.arange(4)

    freeze_mask = (agent_variables.time_of_tag != -AGENT_TAG_SECS_OUT) & (
        mjx_data.time < agent_variables.time_of_tag + AGENT_TAG_SECS_OUT
    )
    resume_mask = (agent_variables.time_of_tag != -AGENT_TAG_SECS_OUT) & (
        mjx_data.time >= agent_variables.time_of_tag + AGENT_TAG_SECS_OUT
    )

    # Override positions: for frozen agents, use a shifted spawn (e.g. z set to -10); for resumed, restore spawn.
    current_pos = jnp.take(new_qpos, dof_indices)  # shape: (n_agents, 4)
    resumed_override = jnp.concatenate(
        [agents_data.spawn_points, agents_data.spawn_orientations[:, None]], axis=1
    )
    frozen_override = resumed_override.at[:, 2].set(-10)
    override_pos = jnp.where(freeze_mask[:, None], frozen_override, current_pos)
    override_pos = jnp.where(resume_mask[:, None], resumed_override, override_pos)
    new_qpos = new_qpos.at[dof_indices].set(override_pos)

    # Override velocities for agents being frozen/resumed.
    current_vel = jnp.take(new_qvel, dof_indices)
    override_vel = jnp.where(
        (freeze_mask | resume_mask)[:, None], jnp.zeros((4,)), current_vel
    )
    new_qvel = new_qvel.at[dof_indices].set(override_vel)

    # Override applied forces for these agents (only the first 3 components).
    freeze_or_resume = freeze_mask | resume_mask
    forces_override = jnp.where(
        freeze_or_resume[:, None],
        jnp.zeros((n_agents, 3)),
        new_xfrc_applied[body_ids, :3],
    )
    new_xfrc_applied = new_xfrc_applied.at[body_ids, :3].set(forces_override)

    # For resumed agents, reset the tagging timer.
    new_time_of_tag = jnp.where(
        resume_mask, -AGENT_TAG_SECS_OUT, agent_variables.time_of_tag
    )
    new_agents_variables = ICLandAgentVariables(
        pitch=new_pitches,
        time_of_tag=new_time_of_tag,
    )
    new_prop_variables = ICLandPropVariables(
        prop_owner=prop_variables.prop_owner,
        time_of_grab=prop_variables.time_of_grab,
    )

    # Build the updated simulation data.
    new_mjx_data = mjx_data.replace(
        xfrc_applied=new_xfrc_applied,
        qpos=new_qpos,
        qvel=new_qvel,
        qfrc_applied=mjx_data.qfrc_applied,
    )

    return new_mjx_data, new_agents_variables, new_prop_variables
