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
os.environ["MUJOCO_GL"] = "wgl"
os.environ["MUJOCO_GL"] = "wgl"

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
from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame
from icland.types import *
from icland.world_gen.converter import create_world, export_stls
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.tile_data import TILECODES

SIMULATION_PRESETS: List[Dict[str, Any]] = [
    # {"name": "ramp_60", "world": RAMP_60, "policy": FORWARD_POLICY, "duration": 4},
    # {"name": "ramp_30", "world": RAMP_30, "policy": FORWARD_POLICY, "duration": 4},
    # {"name": "ramp_45", "world": RAMP_45, "policy": FORWARD_POLICY, "duration": 4},
    # {
    #     "name": "two_agent_move_collide",
    #     "world": TWO_AGENT_EMPTY_WORLD_COLLIDE,
    #     "policy": jnp.array([FORWARD_POLICY, BACKWARD_POLICY]),
    #     "duration": 4,
    #     "agent_count": 2,
    # },
    {
        "name": "world_42_convex",
        "world": WORLD_42_CONVEX,
        "policy": FORWARD_POLICY,
        "duration": 4,
    },
    # {
    #     "name": "two_agent_move_parallel",
    #     "world": TWO_AGENT_EMPTY_WORLD,
    #     "policy": jnp.array([FORWARD_POLICY, FORWARD_POLICY]),
    #     "duration": 4,
    #     "agent_count": 2,
    # },
    # {
    #     "name": "empty_world",
    #     "world": EMPTY_WORLD,
    #     "policy": FORWARD_POLICY,
    #     "duration": 4,
    # },
    # {"name": "ramp_30", "world": RAMP_30, "policy": FORWARD_POLICY, "duration": 4},
    # {"name": "ramp_45", "world": RAMP_45, "policy": FORWARD_POLICY, "duration": 4},
    # {"name": "ramp_60", "world": RAMP_60, "policy": FORWARD_POLICY, "duration": 4},
]

key = jax.random.PRNGKey(42)


def __generate_mjcf_string(  # pragma: no cover
    tile_map: jax.Array,
    mesh_dir: str = "meshes/",
) -> None:
    """Generates MJCF file from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]

    mjcf = f"""<mujoco model=\"generated_mesh_world\">
    <compiler meshdir=\"{mesh_dir}\"/>
    <default>
        <geom type=\"mesh\" />
    </default>
    
    <worldbody>\n"""

    mjcf += """            <body name="agent0" pos="1.5 1 4">
                <joint type="slide" axis="1 0 0" />
                <joint type="slide" axis="0 1 0" />
                <joint type="slide" axis="0 0 1" />
                <joint type="hinge" axis="0 0 1" stiffness="1"/>

                <geom
                    name="agent0_geom"
                    type="capsule"
                    size="0.06"
                    fromto="0 0 0 0 0 -0.4"
                    mass="1"
                />

                <geom
                    type="box"
                    size="0.05 0.05 0.05"
                    pos="0 0 0.2"
                    mass="0"
                />
            </body>"""

    for i, mesh_file in enumerate(mesh_files):
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += (
            f'        <geom name="{mesh_name}" mesh="{mesh_name}" pos="0 0 0"/>' + "\n"
        )

    mjcf += "    </worldbody>\n\n    <asset>\n"

    for mesh_file in mesh_files:
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += f'        <mesh name="{mesh_name}" file="{mesh_file}"/>' + "\n"

    mjcf += "    </asset>\n</mujoco>\n"

    return mjcf


def render_video_from_world(
    key: jax.random.PRNGKey,
    policy: jax.Array,
    duration: int,
    video_name: str,
) -> None:
    """Renders a video using SDF function."""
    print(f"Sampling world with key {key}")
    model = sample_world(10, 10, 1000, key, True, 1)
    print(f"Exporting tilemap")
    tilemap = export(model, TILECODES, 10, 10)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    print(f"Exporting stls")
    export_stls(pieces, f"{temp_dir}/{temp_dir}")
    xml_str = __generate_mjcf_string(tilemap, f"{temp_dir}/")
    print(f"Init mj model...")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, game=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, None, policy)
    print(f"Init mjX model and data...")
    mjx_data = icland_state.pipeline_state.mjx_data
    frames: List[Any] = []

    print(f"Starting simulation: {video_name}")
    last_printed_time = -0.1

    default_agent_1 = 0
    world_width = tilemap.shape[1]
    print(f"Rendering...")
    get_camera_info = jax.jit(get_agent_camera_from_mjx)
    while mjx_data.time < duration:
        if int(mjx_data.time * 10) != int(last_printed_time * 10):
            print(f"Time: {mjx_data.time:.1f}")
            last_printed_time = mjx_data.time
        icland_state = icland.step(key, icland_state, None, policy)
        mjx_data = icland_state.pipeline_state.mjx_data
        if len(frames) < mjx_data.time * 30:
            print(
                "Agent pos:",
                mjx_data.xpos[icland_state.pipeline_state.component_ids[0, 0]][:3],
            )
            camera_pos, camera_dir = get_camera_info(
                icland_state, world_width, default_agent_1
            )
            print("Got camera angle")
            f = render_frame(camera_pos, camera_dir, tilemap)
            print("Rendered frame")
            frames.append(f)

    imageio.mimsave(video_name, frames, fps=30, quality=8)


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

    print(f"Starting simulation: {video_name}")
    last_printed_time = -0.1

    with mujoco.Renderer(mj_model) as renderer:
        while mjx_data.time < duration:
            if int(mjx_data.time * 10) != int(last_printed_time * 10):
                print(f"Time: {mjx_data.time:.1f}")
                last_printed_time = mjx_data.time
            icland_state = icland.step(key, icland_state, None, policy)
            mjx_data = icland_state.pipeline_state.mjx_data
            if len(third_person_frames) < mjx_data.time * 30:
                mj_data = mjx.get_data(mj_model, mjx_data)
                print(
                    "Agent pos:",
                    mjx_data.xpos[icland_state.pipeline_state.component_ids[0, 0]][:3],
                )
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
    # render_video_from_world(
    #     key, FORWARD_POLICY, 3, "tests/video_output/world_convex_42_mjx.mp4"
    # )
    for i, preset in enumerate(SIMULATION_PRESETS):
        render_video(
            preset["world"],
            preset["policy"],
            preset["duration"],
            f"tests/video_output/{preset['name']}.mp4",
            agent_count=preset.get("agent_count", 1),
        )
