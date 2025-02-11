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
import shutil

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from typing import Any

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
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.tile_data import TILECODES

SIMULATION_PRESETS: list[dict[str, Any]] = [
    {
        "name": "two_agent_move_collide",
        "world": TWO_AGENT_EMPTY_WORLD_COLLIDE,
        "policy": jnp.array([FORWARD_POLICY, BACKWARD_POLICY]),
        "duration": 4,
        "agent_count": 2,
    },
    {"name": "ramp_30", "world": RAMP_30, "policy": FORWARD_POLICY, "duration": 4},
    {"name": "ramp_60", "world": RAMP_60, "policy": FORWARD_POLICY, "duration": 4},
    {"name": "ramp_45", "world": RAMP_45, "policy": FORWARD_POLICY, "duration": 4},
    {
        "name": "world_42_convex",
        "world": WORLD_42_CONVEX,
        "policy": FORWARD_POLICY,
        "duration": 4,
    },
    {
        "name": "two_agent_move_parallel",
        "world": TWO_AGENT_EMPTY_WORLD,
        "policy": jnp.array([FORWARD_POLICY, FORWARD_POLICY]),
        "duration": 4,
        "agent_count": 2,
    },
    {
        "name": "world_42_convex",
        "world": WORLD_42_CONVEX,
        "policy": FORWARD_POLICY,
        "duration": 4,
    },
]


def __generate_mjcf_string(  # pragma: no cover
    tile_map: jax.Array,
    agent_spawns: jax.Array,
    mesh_dir: str = "meshes/",
) -> str:
    """Generates MJCF file from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]

    mjcf = f"""<mujoco model=\"generated_mesh_world\">
    <compiler meshdir=\"{mesh_dir}\"/>
    <default>
        <geom type=\"mesh\" />
    </default>

    <worldbody>\n"""

    for i, s in enumerate(agent_spawns):
        spawn_loc = s.tolist()
        spawn_loc[2] += 1
        mjcf += f'            <body name="agent{i}" pos="{" ".join([str(s) for s in spawn_loc])}">'
        mjcf += """
            <joint type="slide" axis="1 0 0" />
            <joint type="slide" axis="0 1 0" />
            <joint type="slide" axis="0 0 1" />
            <joint type="hinge" axis="0 0 1" stiffness="1"/>

            <geom"""
        mjcf += f'        name="agent{i}_geom"'
        mjcf += """        type="capsule"
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
    key: jax.Array,
    policy: jax.Array,
    duration: int,
    video_name: str,
    height: int = 10,
    width: int = 10,
) -> None:
    """Renders a video using SDF function."""
    print(f"Sampling world with key {key[1]}")
    model = sample_world(height, width, 1000, key, True, 1)
    print(f"Exporting tilemap")
    tilemap = export(model, TILECODES, height, width)
    spawnpoints = sample_spawn_points(key, tilemap, num_objects=1)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    os.makedirs(f"{temp_dir}", exist_ok=True)
    print(f"Exporting stls")
    export_stls(pieces, f"{temp_dir}/{temp_dir}")
    xml_str = __generate_mjcf_string(tilemap, spawnpoints, f"{temp_dir}/")
    print(f"Init mj model...")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, game=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, None, policy)
    print(f"Init mjX model and data...")
    mjx_data = icland_state.pipeline_state.mjx_data
    frames: list[Any] = []

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
                mjx_data.xpos[
                    icland_state.pipeline_state.component_ids[default_agent_1, 0]
                ][:3],
            )
            camera_pos, camera_dir = get_camera_info(
                icland_state, world_width, default_agent_1
            )
            # print("Got camera angle")
            f = render_frame(
                camera_pos, camera_dir, tilemap, view_width=96, view_height=72
            )
            # print("Rendered frame")
            frames.append(f)

    shutil.rmtree(f"{temp_dir}")

    imageio.mimsave(video_name, frames, fps=30, quality=8)


def render_video(
    key: jax.Array,
    model_xml: str,
    policy: jax.Array,
    duration: int,
    video_name: str,
    agent_count: int = 1,
) -> None:
    """Renders a video of a simulation using the given model and policy.

    Args:
        key (jax.Array): Random key for initialization.
        model_xml (str): XML string defining the MuJoCo model.
        policy (callable): Policy function to determine the agent's actions.
        duration (float): Duration of the video in seconds.
        video_name (str): Name of the output video file.
        agent_count (int): Number of agents in the simulation.

    Returns:
        None
    """
    mj_model = mujoco.MjModel.from_xml_string(model_xml)

    icland_params = ICLandParams(
        model=mj_model, reward_function=None, agent_count=agent_count
    )

    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, icland_params, policy)
    mjx_data = icland_state.pipeline_state.mjx_data

    third_person_frames: list[Any] = []

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
            icland_state = icland.step(key, icland_state, icland_params, policy)
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


def render_video_from_world_with_policies(
    key: jax.Array,
    policies: list[jax.Array],
    switch_intervals: list[float],
    duration: int,
    video_name: str,
) -> None:
    """Renders a video where the agent follows multiple policies sequentially.

    Args:
        key: Random seed key.
        policies: A list of policy arrays, applied sequentially.
        switch_intervals: Time intervals at which to switch to the next policy.
        duration: Total duration of the simulation.
        video_name: Output video file name.
    """
    print(f"Sampling world with key {key}")
    model = sample_world(10, 10, 1000, key, True, 1)
    tilemap = export(model, TILECODES, 10, 10)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    export_stls(pieces, f"{temp_dir}/{temp_dir}")

    xml_str = __generate_mjcf_string(tilemap, jnp.array([[1.5, 1, 4]]), f"{temp_dir}/")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, game=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    mjx_data = icland_state.pipeline_state.mjx_data
    frames: list[Any] = []

    current_policy_idx = 0
    policy = policies[current_policy_idx]

    print(f"Starting simulation: {video_name}")
    last_printed_time = -0.1

    default_agent_1 = 0
    world_width = tilemap.shape[1]
    get_camera_info = jax.jit(get_agent_camera_from_mjx)

    while mjx_data.time < duration:
        # Switch policy at defined intervals
        if (
            current_policy_idx < len(switch_intervals)
            and mjx_data.time >= switch_intervals[current_policy_idx]
        ):
            current_policy_idx += 1
            if current_policy_idx < len(policies):
                policy = policies[current_policy_idx]
                print(f"Switching policy at {mjx_data.time:.1f}s")

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
            f = render_frame(camera_pos, camera_dir, tilemap)
            frames.append(f)

    imageio.mimsave(video_name, frames, fps=30, quality=8)


if __name__ == "__main__":
    # keys = [
    #     jax.random.PRNGKey(42),
    #     jax.random.PRNGKey(420),
    #     jax.random.PRNGKey(2004),
    # ]
    # for k in keys:
    #     render_video_from_world(
    #         k, FORWARD_POLICY, 4, f"tests/video_output/world_convex_{k[1]}_mjx.mp4"
    #     )
    for i, preset in enumerate(SIMULATION_PRESETS):
        print(f"Running preset {i + 1}/{len(SIMULATION_PRESETS)}: {preset['name']}")
        render_video(
            jax.random.PRNGKey(42),
            preset["world"],
            preset["policy"],
            preset["duration"],
            f"tests/video_output/{preset['name']}.mp4",
            agent_count=preset.get("agent_count", 1),
        )
