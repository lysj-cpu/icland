"""This script generates videos of simulations using predefined simulation presets.

It sets up the environment variables for MuJoCo and XLA, imports necessary libraries,
and defines a function to render videos of simulations.

Functions:
    render_video(model_xml, policy, duration, video_name):

Simulation Presets:
    SIMULATION_PRESETS: A list of dictionaries containing simulation configurations.

Execution:
    Iterates over the simulation presets and generates videos for each preset.
"""

import os

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from typing import Any, Dict, List

import imageio
import jax
import jax.numpy as jnp
import mujoco
from assets.policies import *
from assets.worlds import *
from mujoco import mjx

import icland
from icland.types import *

SIMULATION_PRESETS: List[Dict[str, Any]] = [
    {"name": "ramp_30", "world": RAMP_30, "policy": FORWARD_POLICY, "duration": 4},
    {"name": "ramp_45", "world": RAMP_45, "policy": FORWARD_POLICY, "duration": 4},
    {"name": "ramp_60", "world": RAMP_60, "policy": FORWARD_POLICY, "duration": 4},
    {
        "name": "two_agent_move_collide",
        "world": TWO_AGENT_EMPTY_WORLD_COLLIDE,
        "policy": jnp.array([FORWARD_POLICY, BACKWARD_POLICY]),
        "duration": 4,
        "agent_count": 2,
    },
    {
        "name": "two_agent_move_parallel",
        "world": TWO_AGENT_EMPTY_WORLD,
        "policy": jnp.array([FORWARD_POLICY, FORWARD_POLICY]),
        "duration": 4,
        "agent_count": 2,
    },
    {
        "name": "empty_world",
        "world": EMPTY_WORLD,
        "policy": FORWARD_POLICY,
        "duration": 4,
    },
]

key = jax.random.PRNGKey(42)


def render_video(
    model_xml: str,
    policy: jax.Array,
    duration: int,
    video_name: str,
    agent_count: int = 1,
) -> None:
    """Renders a video of a simulation using the given model and policy.

    Args:
        model_xml (str): XML string defining the MuJoCo model.
        policy (callable): Policy function to determine the agent's actions.
        duration (float): Duration of the video in seconds.
        video_name (str): Name of the output video file.
        agent_count (int): Number of agents in the simulation.

    Returns:
        None
    """
    mj_model = mujoco.MjModel.from_xml_string(model_xml)

    icland_params = ICLandParams(model=mj_model, game=None, agent_count=agent_count)

    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, None, policy)
    mjx_data = icland_state.pipeline_state.mjx_data

    third_person_frames: List[Any] = []

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = icland_state.pipeline_state.component_ids[0, 0]
    cam.distance = 1.5
    cam.azimuth = 90.0
    cam.elevation = -40.0

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    with mujoco.Renderer(mj_model) as renderer:
        while mjx_data.time < duration:
            print(mjx_data.time)
            icland_state = icland.step(key, icland_state, None, policy)
            mjx_data = icland_state.pipeline_state.mjx_data
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

    imageio.mimsave(video_name, third_person_frames, fps=30, quality=8)


if __name__ == "__main__":
    for i, preset in enumerate(SIMULATION_PRESETS):
        render_video(
            preset["world"],
            preset["policy"],
            preset["duration"],
            f"tests/video_output/{preset['name']}.mp4",
            agent_count=preset.get("agent_count", 1),
        )
