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
      <joint name="slide_x" type="slide" axis="1 0 0" />
      <joint name="slide_y" type="slide" axis="0 1 0" />
      <joint name="slide_z" type="slide" axis="0 0 1" />
      <camera name="first_person" pos="0 0 0.1" mode="fixed"/>

      <geom
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.01"
      />
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
      friction="0.001 0.001 0.0001"
    />

    <geom type="box" size="0.5 1 1" pos="0.45 2 0"
          rgba="1 0.8 0.8 1"/>

    <geom name="ramp" type="mesh" mesh="ramp" material="rampmat"
          pos="1.5 0 0.1" euler="0 0 90" />
  </worldbody>
</mujoco>
"""

# Load model and data
mj_model = mujoco.MjModel.from_xml_string(XML)
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Camera setup
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "capsule")
cam.trackbodyid = body_id
cam.distance = 3.0
cam.azimuth = 90.0
cam.elevation = -40.0

# First-person camera
first_person_cam = mujoco.MjvCamera()
first_person_camera_id = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "first_person"
)
first_person_cam.fixedcamid = first_person_camera_id
first_person_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

# Visualization options
opt = mujoco.MjvOption()
# opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Parameters
duration = 3.0  # [s]
framerate = 30  # [Hz]
max_speed = 1.0
friction_coefficient = 0.1
up_vector = jnp.array([0, 0, 1])


def get_velocity_vector(sim_time):
    """Returns the driving force of the agent depending on the simulation time."""
    return jnp.select(
        [sim_time < 2, sim_time < 4, sim_time < 6, sim_time < 8],
        [
            jnp.array([1, 0, 0], dtype=jnp.float32),
            jnp.array([0, 1, 0], dtype=jnp.float32),
            jnp.array([-1, 0, 0], dtype=jnp.float32),
            jnp.array([0, -1, 0], dtype=jnp.float32),
        ],
        default=jnp.array([0, 0, 0], dtype=jnp.float32),
    )


def get_first_person_camera_dir(sim_time):
    """Returns the angle that the front-facing camera should face based on simulation time."""
    return jnp.select(
        [sim_time < 2, sim_time < 4, sim_time < 6, sim_time < 8],
        [
            jnp.array([0.5, 0.5, -0.5, -0.5], dtype=jnp.float32),  # Face forwards
            jnp.array(
                [math.sqrt(2) / 2, math.sqrt(2) / 2, 0, 0], dtype=jnp.float32
            ),  # Face left
            jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32),  # Face backward
            jnp.array(
                [0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2], dtype=jnp.float32
            ),  # Face right
        ],
    )


@jax.jit
def simulate_step(mjx_model, mjx_data):
    """Steps the simulator after adjusting agent forces."""
    driving_force = get_velocity_vector(mjx_data.time)

    # Handle contact
    if mjx_data.ncon > 0:
        # Access the first contact's normal
        normal = mjx_data.contact.frame[0][0]
        norm_len = jnp.linalg.norm(normal)

        # Normalize the contact normal if its length is significant
        normal = jax.lax.cond(
            norm_len > 1e-8,
            lambda n: n / norm_len,
            lambda n: n,
            normal,
        )

        # Compute slope direction
        slope_direction = up_vector - jnp.dot(up_vector, normal) * normal
        slope_len = jnp.linalg.norm(slope_direction)

        # Normalize the slope direction if its length is significant
        slope_direction = jax.lax.cond(
            jnp.logical_and(norm_len > 1e-8, slope_len > 1e-8),
            lambda s: s / slope_len,
            lambda s: s,
            slope_direction,
        )

        # Check capsule velocity direction
        velocity_3d = mjx_data.qvel[:3]
        speed_3d = jnp.linalg.norm(velocity_3d)
        velocity_dir = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(norm_len > 1e-8, slope_len > 1e-8), speed_3d > 1e-8
            ),
            lambda v: v / speed_3d,
            lambda v: v,
            velocity_3d,
        )

        # Check if velocity is aligned with slope direction and adjust driving force
        dot_with_slope = jnp.dot(velocity_dir, slope_direction)
        driving_force = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(norm_len > 1e-8, slope_len > 1e-8), speed_3d > 1e-8
                ),
                jnp.any(dot_with_slope > 0.0),
            ),
            lambda _: jnp.linalg.norm(driving_force) * slope_direction,
            lambda _: driving_force,
            None,
        )

    # Apply driving force
    mjx_data = mjx_data.replace(
        xfrc_applied=mjx_data.xfrc_applied.at[body_id, :3].set(driving_force)
    )

    # Step the simulator
    mjx_data = mjx.step(mjx_model, mjx_data)

    # Cap speed
    vel_2d = mjx_data.qvel[:2]
    speed = jnp.linalg.norm(vel_2d)
    mjx_data = jax.lax.cond(
        speed > max_speed,
        lambda _: mjx_data.replace(
            qvel=mjx_data.qvel.at[:2].set(vel_2d * (max_speed / speed))
        ),
        lambda _: mjx_data,
        None,
    )

    # Apply linear drag friction
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[:2].set(mjx_data.qvel[:2] * (1.0 - friction_coefficient))
    )

    return mjx_data


# Simulation loop
third_person_frames = []
first_person_frames = []

with mujoco.Renderer(mj_model) as renderer:
    while mjx_data.time < duration:
        mjx_data = simulate_step(mjx_model, mjx_data)

        # Render frames
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

            # # First-person camera
            mj_model.cam_quat[first_person_camera_id] = get_first_person_camera_dir(
                mj_data.time
            )
            mujoco.mjv_updateScene(
                mj_model,
                mj_data,
                opt,
                None,
                first_person_cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                renderer.scene,
            )
            first_person_pixels = renderer.render()
            first_person_frames.append(first_person_pixels)

third_person_output_file = "third_person_view.mp4"
first_person_output_file = "first_person_view.mp4"

# ImageIO warning: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
# Write third-person view
imageio.mimwrite(
    third_person_output_file, third_person_frames, fps=framerate, quality=8
)

# Write first-person view
imageio.mimwrite(
    first_person_output_file, first_person_frames, fps=framerate, quality=8
)

print(f"Third-person view saved to {third_person_output_file}")
print(f"First-person view saved to {first_person_output_file}")
