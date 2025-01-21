"""Generates videos of a single agent interacting with a basic environment."""

import os

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import math
import imageio

XML = r"""
<mujoco>
  <asset>
    <mesh file="ramp.stl" name="ramp" scale="1 1 1"/>
    <material name="rampmat" specular="1" shininess=".3" reflectance="0"
              rgba="0.0 0.5 1.0 1.0"/>
  </asset>
  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="capsule" pos="0 0 1">
      <!-- 3 slides + 1 hinge around Z -->
      <joint name="slide_x" type="slide" axis="1 0 0" />
      <joint name="slide_y" type="slide" axis="0 1 0" />
      <joint name="slide_z" type="slide" axis="0 0 1" />
      <joint name="hinge_z" type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="capsule_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.01"
      />

      <geom
        name="Cap_top"
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.001"
      />
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom name="box" type="box" size="0.5 1 1" pos="0.45 2 -0.2" euler="0 -5 0"
          rgba="1 0.8 0.8 1"
          />

    <geom name="ramp" type="mesh" mesh="ramp" material="rampmat"
          pos="1.5 0 0.1" euler="0 0 90" 
          />

  </worldbody>
</mujoco>
"""

# ------------------------------------------------------------------------------
# 1) Load model, data, and renderer
# ------------------------------------------------------------------------------
mj_model = mujoco.MjModel.from_xml_string(XML)
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# ------------------------------------------------------------------------------
# 2) Initialize camera
# ------------------------------------------------------------------------------
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

# Track the capsule
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "capsule")
geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "capsule_geom")
cam.trackbodyid = body_id
cam.distance = 3
cam.azimuth = 90.0
cam.elevation = -40.0

for g_id in range(mj_model.ngeom):
    geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g_id)  # Get geom name
    print(f"Geom ID: {g_id}, Geom Name: {geom_name}")

# ------------------------------------------------------------------------------
# 3) Visualization options
# ------------------------------------------------------------------------------
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# ------------------------------------------------------------------------------
# 4) Simulation parameters
# ------------------------------------------------------------------------------
duration = 8  # seconds
framerate = 30  # Hz
frames = []

max_speed = 1.0  # Cap on linear speed
max_rot_speed = 5.0  # Optional cap on rotation speed (rad/s)
friction_coefficient = 0.1  # Simple linear + rotational friction
hinge_dof = 3  # The 4th DOF (index=3) is the hinge around z


@jax.jit
def get_local_movement(sim_time):
    conditions = [
        (0 <= sim_time) & (sim_time < 2),  # local +X
        (2 <= sim_time) & (sim_time < 4),  # local +Y
        (4 <= sim_time) & (sim_time < 6),  # local -X
        (6 <= sim_time) & (sim_time < 8)   # local -Y
    ]
    choices = [
        jnp.array([1.0, 0.0]),  # local +X
        jnp.array([0.0, 1.0]),  # local +Y
        jnp.array([-1.0, 0.0]), # local -X
        jnp.array([0.0, -1.0])  # local -Y
    ]
    default = jnp.array([0.0, 0.0])  # Default case
    # return jnp.select(conditions, choices, default)
    return jnp.array([1.0, 0.0])

@jax.jit
def get_rotation_torque(sim_time):
    """
    Return a torque about z-axis depending on sim_time.
    We'll do a simple pattern:
       0-2s: turn left
       2-4s: turn right
       4-6s: turn left
       6-8s: turn right
    """
    return jnp.select(
        [
            jnp.logical_and(1 <= sim_time, sim_time < 4),
            sim_time < 4.05,
            jnp.logical_and(6 <= sim_time, sim_time < 6.05),
        ],
        [0.001, 0.5, 0.5],
        default=0.0,
    )


@jax.jit
def simulate_step(mjx_model, mjx_data):
    # --------------------------------------------------------------------------
    # (A) Determine local movement and rotate it to world frame
    # --------------------------------------------------------------------------
    local_dir = get_local_movement(mjx_data.time)  # local (x,y)

    # Hinge angle about z is in qpos[3]
    angle = mjx_data.qpos[hinge_dof]

    # 2D rotation matrix for the angle
    R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])

    # Transform local movement direction to world frame
    world_dir = R @ local_dir

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
            mjx_data.contact.geom2[nth_contact] == geom_id
        )

        is_touching = mjx_data.contact.dist[nth_contact] < 0.0

        valid_collision = jnp.logical_and(is_agent_collision, is_touching)

        # If slope is too steep (angle > ~45deg), remove that direction
        movement_direction = jnp.where(
            jnp.logical_and(valid_collision, slope_mag > 0.7), 
            slope_component / (slope_mag + 1e-10), 
            movement_direction
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
    rotation_torque = get_rotation_torque(mjx_data.time) * jnp.pi
    mjx_data = mjx_data.replace(
        qfrc_applied=mjx_data.qfrc_applied.at[hinge_dof].set(
            mjx_data.qfrc_applied[hinge_dof] + rotation_torque
        )
    )

    # mjx_data = mjx_data.replace(
    #     xfrc_applied=mjx_data.qfrc_applied.at[hinge_dof].set(mjx_data.qfrc_applied[hinge_dof] + rotation_torque)
    # )

    # --------------------------------------------------------------------------
    # (E) Step the simulator
    # --------------------------------------------------------------------------
    mjx_data = mjx.step(mjx_model, mjx_data)

    # --------------------------------------------------------------------------
    # (F) Clamp linear speed in XY
    # --------------------------------------------------------------------------
    vel_2d = mjx_data.qvel[0:2]  # [vx, vy]
    speed = jnp.linalg.norm(vel_2d)

    scale = jnp.where(speed > max_speed, max_speed / speed, 1.0)
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:2].multiply(scale))

    # --------------------------------------------------------------------------
    # (G) Clamp angular velocity about z
    # --------------------------------------------------------------------------
    omega = mjx_data.qvel[hinge_dof]
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[hinge_dof].set(
            jnp.where(
                abs(omega) > max_rot_speed,
                jnp.sign(omega) * max_rot_speed,
                mjx_data.qvel[hinge_dof],
            )
        )
    )

    # mjx_data = mjx_data.replace(qvel=)

    # --------------------------------------------------------------------------
    # (H) Apply linear and rotational friction
    # --------------------------------------------------------------------------
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[jnp.array([0, 1, hinge_dof])].multiply(
            1.0 - friction_coefficient
        )
    )

    return mjx_data


third_person_frames = []

with mujoco.Renderer(mj_model) as renderer:
    while mjx_data.time < duration:
        print(mjx_data.time)
        mjx_data = simulate_step(mjx_model, mjx_data)
        if len(third_person_frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            mujoco.mjv_updateScene(
                mj_model,
                mj_data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                renderer.scene,
            )
            third_person_frames.append(renderer.render())

# ------------------------------------------------------------------------------
# 6) Show the video
# ------------------------------------------------------------------------------
imageio.mimwrite("new.mp4", third_person_frames, fps=framerate, quality=8)
 