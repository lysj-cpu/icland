"""
This script generates videos of simulations using predefined simulation presets.

It sets up the environment variables for MuJoCo and XLA, imports necessary libraries,
and defines functions to render videos of simulations.

Functions:
    render_video(model_xml, policy, duration, video_name, agent_count):
    render_video_from_world(key, policy, duration, video_name, height, width):
    render_video_from_world_with_policies(key, policies, switch_intervals, duration, video_name):
    render_sdfr(key, policy, duration, video_name, height, width):

Simulation Presets:
    SIMULATION_PRESETS: A list of dictionaries containing simulation configurations.

Execution:
    Depending on command-line arguments, either the SDF renderer is run or each preset is iterated.
"""

import os
import shutil
import sys
from typing import Any, Callable, Optional

import numpy
import imageio
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# N.B. These need to be set before the MuJoCo imports
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM; this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

# Local module imports
import icland
from icland.presets import *
from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame
from icland.types import *
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.tile_data import TILECODES

# ---------------------------------------------------------------------------
# Simulation Presets
# ---------------------------------------------------------------------------
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
        "name": "two_agent_move_parallel",
        "world": TWO_AGENT_EMPTY_WORLD,
        "policy": jnp.array([FORWARD_POLICY, FORWARD_POLICY]),
        "duration": 4,
        "agent_count": 2,
    },
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _generate_mjcf_string(
    tile_map: jax.Array,
    agent_spawns: jax.Array,
    mesh_dir: str = "tests/assets/meshes/",
) -> str:
    """Generates an MJCF string from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]
    w, h = tile_map.shape[0], tile_map.shape[1]

    mjcf = f"""<mujoco model="generated_mesh_world">
    <compiler meshdir="{mesh_dir}"/>
    <default>
        <geom type="mesh" />
    </default>
    <worldbody>
"""
    # Add agent bodies
    for i, s in enumerate(agent_spawns):
        spawn_loc = s.tolist()
        spawn_loc[2] += 1
        mjcf += (
            f'            <body name="agent{i}" pos="{" ".join([str(s) for s in spawn_loc])}">'
            + """
                <joint type="slide" axis="1 0 0" />
                <joint type="slide" axis="0 1 0" />
                <joint type="slide" axis="0 0 1" />
                <joint type="hinge" axis="0 0 1" stiffness="1"/>
                <geom"""
        )
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
            </body>
"""
    # Add mesh geoms
    for mesh_file in mesh_files:
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += f'        <geom name="{mesh_name}" mesh="{mesh_name}" pos="0 0 0"/>\n'

    # Add walls
    mjcf += f'        <geom name="east_wall" type="plane"\n'
    mjcf += f"""            pos="{w} {h / 2} 10"
            quat="0.5 -0.5 -0.5 0.5"
            size="{h / 2} 10 0.01"
            rgba="1 0.819607843 0.859375 0.5" />\n"""
    mjcf += f'        <geom name="west_wall" type="plane"\n'
    mjcf += f"""            pos="0 {h / 2} 10"
            quat="0.5 0.5 0.5 0.5"
            size="{h / 2} 10 0.01"
            rgba="1 0.819607843 0.859375 0.5" />\n"""
    mjcf += f'        <geom name="north_wall" type="plane"\n'
    mjcf += f"""            pos="{w / 2} 0 10"
            quat="0.5 -0.5 0.5 0.5"
            size="10 {w / 2} 0.01"
            rgba="1 0.819607843 0.859375 0.5" />\n"""
    mjcf += f'        <geom name="south_wall" type="plane"\n'
    mjcf += f"""            pos="{w / 2} {h} 10"
            quat="0.5 0.5 -0.5 0.5"
            size="10 {w / 2} 0.01"
            rgba="1 0.819607843 0.859375 0.5" />\n"""
    mjcf += "    </worldbody>\n\n    <asset>\n"
    for mesh_file in mesh_files:
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += f'        <mesh name="{mesh_name}" file="{mesh_file}"/>\n'
    mjcf += "    </asset>\n</mujoco>\n"

    return mjcf


def _simulate_frames(
    key: jax.Array,
    icland_state: Any,
    icland_params: ICLandParams,
    duration: float,
    frame_callback: Callable[[Any], Any],
    policy: jax.Array,
    frame_rate: int = 30,
    policy_switcher: Optional[Callable[[float, jax.Array], jax.Array]] = None,
) -> list[Any]:
    """
    Runs the simulation loop until `duration` and collects frames using `frame_callback`.
    Optionally, a `policy_switcher` function may update the policy based on simulation time.
    """
    frames = []
    mjx_data = icland_state.pipeline_state.mjx_data
    last_printed_time = -0.1
    current_policy = policy

    while mjx_data.time < duration:
        if policy_switcher is not None:
            current_policy = policy_switcher(mjx_data.time, current_policy)
        if int(mjx_data.time * 10) != int(last_printed_time * 10):
            print(f"Time: {mjx_data.time:.1f}")
            last_printed_time = mjx_data.time
        icland_state = icland.step(key, icland_state, icland_params, current_policy)
        mjx_data = icland_state.pipeline_state.mjx_data
        if len(frames) < mjx_data.time * frame_rate:
            frames.append(frame_callback(icland_state))
    return frames


# ---------------------------------------------------------------------------
# Rendering Functions
# ---------------------------------------------------------------------------
def render_sdfr(
    key: jax.Array,
    policy: jax.Array,
    duration: int,
    video_name: str,
    height: int = 10,
    width: int = 10,
) -> None:
    """Renders a video using an SDF function."""
    print(f"Sampling world with key {key[1]}")
    model = sample_world(height, width, 1000, key, True, 1)
    print("Exporting tilemap")
    tilemap = numpy.zeros((width, height, 4), dtype=numpy.int32)
    mj_model = mujoco.MjModel.from_xml_string(EMPTY_WORLD)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)
    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, icland_params, policy)

    default_agent = 0
    world_width = tilemap.shape[1]
    get_camera_info = jax.jit(get_agent_camera_from_mjx)
    frame_callback = lambda state: render_frame(
        *get_camera_info(state, world_width, default_agent), tilemap, view_width=96, view_height=72
    )

    print(f"Starting simulation: {video_name}")
    frames = _simulate_frames(key, icland_state, icland_params, duration, frame_callback, policy)
    imageio.mimsave(video_name, frames, fps=30, quality=8)


def render_video_from_world(
    key: jax.Array,
    policy: jax.Array,
    duration: int,
    video_name: str,
    height: int = 10,
    width: int = 10,
) -> None:
    """Renders a video from a generated world."""
    print(f"Sampling world with key {key[1]}")
    model = sample_world(height, width, 1000, key, True, 1)
    print("Exporting tilemap")
    tilemap = export(model, TILECODES, height, width)
    spawnpoints = sample_spawn_points(key, tilemap, num_objects=1)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    print("Exporting stls")
    export_stls(pieces, f"{temp_dir}/{temp_dir}")
    xml_str = _generate_mjcf_string(tilemap, spawnpoints, f"{temp_dir}/")
    print("Init mj model...")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    icland_state = icland.step(key, icland_state, icland_params, policy)

    default_agent = 0
    world_width = tilemap.shape[1]
    get_camera_info = jax.jit(get_agent_camera_from_mjx)
    frame_callback = lambda state: render_frame(
        *get_camera_info(state, world_width, default_agent), tilemap, view_width=96, view_height=72
    )

    print(f"Starting simulation: {video_name}")
    frames = _simulate_frames(key, icland_state, icland_params, duration, frame_callback, policy)
    shutil.rmtree(temp_dir)
    imageio.mimsave(video_name, frames, fps=30, quality=8)


def render_video(
    key: jax.Array,
    model_xml: str,
    policy: jax.Array,
    duration: int,
    video_name: str,
    agent_count: int = 1,
) -> None:
    """Renders a video of a simulation using a third-person view via MuJoCo's renderer."""
    mj_model = mujoco.MjModel.from_xml_string(model_xml)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=agent_count)

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
    """
    Renders a video where the agent follows multiple policies sequentially.
    The policy is switched at the times specified in `switch_intervals`.
    """
    print(f"Sampling world with key {key}")
    model = sample_world(10, 10, 1000, key, True, 1)
    tilemap = export(model, TILECODES, 10, 10)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    export_stls(pieces, f"{temp_dir}/{temp_dir}")

    xml_str = _generate_mjcf_string(tilemap, jnp.array([[1.5, 1, 4]]), f"{temp_dir}/")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)

    icland_state = icland.init(key, icland_params)

    default_agent = 0
    world_width = tilemap.shape[1]
    get_camera_info = jax.jit(get_agent_camera_from_mjx)
    frame_callback = lambda state: render_frame(
        *get_camera_info(state, world_width, default_agent), tilemap
    )

    # Setup policy switching using a closure.
    current_policy_idx = 0
    current_policy = policies[current_policy_idx]

    def policy_switcher(time: float, current: jax.Array) -> jax.Array:
        nonlocal current_policy_idx
        if current_policy_idx < len(switch_intervals) and time >= switch_intervals[current_policy_idx]:
            current_policy_idx += 1
            if current_policy_idx < len(policies):
                print(f"Switching policy at {time:.1f}s")
                return policies[current_policy_idx]
        return current

    print(f"Starting simulation: {video_name}")
    frames = _simulate_frames(
        key, icland_state, icland_params, duration, frame_callback, current_policy, policy_switcher=policy_switcher
    )

    shutil.rmtree(temp_dir)
    imageio.mimsave(video_name, frames, fps=30, quality=8)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-sdfr":
        render_sdfr(
            jax.random.PRNGKey(42),
            FORWARD_POLICY,
            4,
            "scripts/video_output/sdf.mp4",
        )
    else:
        for i, preset in enumerate(SIMULATION_PRESETS):
            print(f"Running preset {i + 1}/{len(SIMULATION_PRESETS)}: {preset['name']}")
            render_video(
                jax.random.PRNGKey(42),
                preset["world"],
                preset["policy"],
                preset["duration"],
                f"scripts/video_output/{preset['name']}.mp4",
                agent_count=preset.get("agent_count", 1),
            )
