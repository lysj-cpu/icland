"""This is only temporary while testing."""

import os

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import ICLand
import jax
import mujoco
from mujoco import mjx
import jax.numpy as jnp
import math
import imageio

key = jax.random.PRNGKey(42)

icland_params = ICLand.sample(key)


cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
mj_model = icland_params[0]


icland_state = ICLand.init(key, icland_params)

cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = icland_state[2][0][0]
cam.distance = 3
cam.azimuth = 90.0
cam.elevation = -40.0

opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

icland_state = ICLand.step(key, icland_state, None, jnp.array([1, 0, 0]))
mjx_data = icland_state[1]


third_person_frames = []

print(mj_model)
with mujoco.Renderer(mj_model) as renderer:
    while mjx_data.time < 8:
        print(mjx_data.time)
        icland_state = ICLand.step(key, icland_state, None, jnp.array([1, 0, 0]))
        mjx_data = icland_state[1]
        if len(third_person_frames) < mjx_data.time * 30:
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
imageio.mimwrite("new.mp4", third_person_frames, fps=30, quality=8)
