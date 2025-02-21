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

from typing import Any

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import icland
from icland.presets import *
from icland.renderer.renderer import (
    PlayerInfo,
    PropInfo,
    generate_colormap,
    get_agent_camera_from_mjx,
    render_frame,
    render_frame_with_objects,
)
from icland.types import *
from icland.world_gen.converter import sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.model_editing import _edit_mj_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES

SIMULATION_PRESETS: list[dict[str, Any]] = [
    # {
    #     "name": "world_42_convex",
    #     "world": WORLD_42_CONVEX,
    #     "policy": FORWARD_POLICY,
    #     "duration": 4,
    # },
    # {
    #     "name": "world_42_convex",
    #     "world": WORLD_42_CONVEX,
    #     "policy": RIGHT_POLICY,
    #     "duration": 2.5,
    # },
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


def _generate_mjcf_string(  # pragma: no cover
    tile_map: jax.Array,
    agent_spawns: jax.Array,
    mesh_dir: str = "tests/assets/meshes/",
) -> str:
    """Generates MJCF file from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]

    w, h = tile_map.shape[0], tile_map.shape[1]

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
            </body>\n"""

    for i, mesh_file in enumerate(mesh_files):
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += (
            f'        <geom name="{mesh_name}" mesh="{mesh_name}" pos="0 0 0"/>' + "\n"
        )

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
    model = sample_world(width, height, 1000, key, True, 1)
    print(f"Exporting tilemap")
    tilemap = export(model, TILECODES, width, height)
    spawnpoints = sample_spawn_points(key, tilemap, num_objects=1)
    print(f"Init mj model...")
    mjx_model, mj_model = generate_base_model(ICLandConfig(width, height, 1, {}, 6))
    icland_params = ICLandParams(
        world=tilemap,
        reward_function=None,
        agent_spawns=jnp.array([[1.5, 1, 4]]),
        world_level=6,
    )

    icland_state = icland.init(key, icland_params, mjx_model)

    icland_state = icland.step(key, icland_state, icland_params, policy)
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
        icland_state = icland.step(key, icland_state, icland_params, policy)
        mjx_data = icland_state.pipeline_state.mjx_data
        if len(frames) < mjx_data.time * 30:
            camera_pos, camera_dir = get_camera_info(
                icland_state, world_width, default_agent_1
            )
            print("agent pos:", mjx_data.xpos[1])
            # print("Got camera angle")
            f = render_frame(
                camera_pos, camera_dir, tilemap, view_width=96, view_height=72
            )
            # print("Rendered frame")
            frames.append(f)

    imageio.mimsave(video_name, frames, fps=30, quality=8)


def __combine_frames(
    agent_frames: dict[int, list[Any]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Arrange agent frames into a grid."""
    agent_count = len(agent_frames)
    frames = [np.array(agent_frames[aid]) for aid in range(agent_count)]

    # Get the dimensions of the frames
    h, w, c = frames[0][0].shape

    stacked_frames = np.array(frames)  # Shape: (agent_count, timesteps, h, w, c)

    # Calculate grid size
    cols = int(np.ceil(np.sqrt(agent_count)))
    rows = int(np.ceil(agent_count / cols))

    # Pad with blac frames if needed to fill the grid
    pad_frames = np.zeros((rows * cols - agent_count, h, w, c), dtype=np.uint8)
    all_frames = np.vstack([stacked_frames, pad_frames])

    # Reshape int (rows, cols, height, width, channels)
    grid = all_frames.reshape(rows, cols, h, w, c)

    # Merge rows horizontally, then merge all rows vertically
    row_frames = [np.hstack(row) for row in grid]
    result = np.vstack(row_frames)

    return result


def render_video_multi_agent(
    key: jax.Array,
    policy: jax.Array,
    duration: int,
    agent_count: int,
    video_name: str,
    height: int = 10,
    width: int = 10,
) -> None:
    """Renders separate first person view videos for multiple agents."""
    print(f"Sampling world with key {key[1]}")
    model = sample_world(width, height, 1000, key, True, 1)
    print(f"Exporting tilemap")
    tilemap = export(model, TILECODES, width, height)

    spawnpoints = sample_spawn_points(key, tilemap, num_objects=agent_count)
    spawnpoints = spawnpoints.at[:, 2].add(1)
    print(f"Init mj model...")
    mjx_model, mj_model = generate_base_model(
        ICLandConfig(width, height, agent_count, {}, 6)
    )
    icland_params = ICLandParams(
        world=tilemap,
        reward_function=None,
        agent_spawns=jnp.array([[1 + i, 1, 4] for i in range(agent_count)]),
        world_level=6,
    )

    icland_state = icland.init(key, icland_params, mjx_model)

    icland_state = icland.step(key, icland_state, icland_params, policy)
    print(f"Init mjX model and data...")
    mjx_data = icland_state.pipeline_state.mjx_data

    # Store frames for each agent separately
    agent_frames: dict[int, list[Any]] = {
        agent_id: [] for agent_id in range(agent_count)
    }

    print(f"Starting simulation for {agent_count}: {video_name}")
    last_printed_time = -0.1
    world_width = tilemap.shape[1]

    # JIT-compiled
    get_camera_info = jax.jit(get_agent_camera_from_mjx)

    @jax.jit
    def step_sim(
        icland_state: ICLandState, key: jax.Array
    ) -> tuple[ICLandState, MjxStateType]:
        icland_state = icland.step(key, icland_state, icland_params, policy)
        mjx_data = icland_state.pipeline_state.mjx_data
        return icland_state, mjx_data

    print("Rendering multiple agents...")

    # List to store combined frames for each timestep
    all_combined_frames: list[Any] = []

    while mjx_data.time < duration:
        if int(mjx_data.time * 10) != int(last_printed_time * 10):
            print(f"Time: {mjx_data.time:.1f}")
            last_printed_time = mjx_data.time

        icland_state, mjx_data = step_sim(icland_state, key)

        if len(next(iter(agent_frames.values()))) < mjx_data.time * 30:
            camera_pos, camera_dir = jax.vmap(get_camera_info, in_axes=(None, None, 0))(
                icland_state, world_width, jnp.arange(agent_count)
            )
            players = jax.vmap(
                lambda x: PlayerInfo(
                    pos=mjx_data.xpos[icland_state.pipeline_state.component_ids[x, 0]][
                        :3
                    ],
                    col=jnp.array([1, 0, 0]),
                )
            )(jnp.arange(agent_count))
            props = PropInfo(jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([]))

            for aid in range(agent_count):
                f = render_frame_with_objects(
                    camera_pos[aid],
                    camera_dir[aid],
                    tilemap,
                    generate_colormap(key, width, height),
                    players,
                    props,
                    view_width=96,
                    view_height=72,
                )
                # print("Rendered frame")
                agent_frames[aid].append(f)

            # Combine frames for this timestep and append to all_combined_frames
            combined_frame = __combine_frames(agent_frames)
            all_combined_frames.append(combined_frame)

            # print("agent pos:", mjx_data.xpos[1])
            # print("Got camera angle")

    all_combined_frames = [frame for frame in all_combined_frames]

    # Save the combined video
    imageio.mimsave(video_name, all_combined_frames, fps=30, quality=8)


def render_video(
    key: jax.Array,
    tilemap: jax.Array,
    policy: jax.Array,
    duration: int,
    video_name: str,
    agent_count: int = 1,
) -> None:
    """Renders a video of a simulation using the given model and policy.

    Args:
        key (jax.Array): Random key for initialization.
        tilemap (jax.Array): Tilemap of the world terrain.
        policy (callable): Policy function to determine the agent's actions.
        duration (float): Duration of the video in seconds.
        video_name (str): Name of the output video file.
        agent_count (int): Number of agents in the simulation.

    Returns:
        None
    """
    config = ICLandConfig(10, 10, 1, {}, 6)
    mjx_model, mj_model = generate_base_model(config)
    icland_params = ICLandParams(
        world=tilemap,
        reward_function=None,
        agent_spawns=jnp.array([[1.5, 1, 4]]),
        world_level=6,
    )

    icland_state = icland.init(key, icland_params, mjx_model)
    icland_state = icland.step(key, icland_state, icland_params, policy)
    mjx_data = icland_state.pipeline_state.mjx_data
    _edit_mj_model_data(
        tilemap=tilemap,
        base_model=mj_model,
        max_world_level=config.max_world_level,
    )

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
    """Renders a video where the agent follows multiple policies sequentially.

    Args:
        key: Random seed key.
        policies: A list of policy arrays, applied sequentially.
        switch_intervals: Time intervals at which to switch to the next policy.
        duration: Total duration of the simulation.
        video_name: Output video file name.
    """
    print(f"Sampling world with key {key}")
    width, height = 10, 10
    model = sample_world(width, height, 1000, key, True, 1)
    tilemap = export(model, TILECODES, width, height)
    mjx_model, mj_model = generate_base_model(ICLandConfig(width, height, 1, {}, 6))
    icland_params = ICLandParams(
        world=tilemap,
        reward_function=None,
        agent_spawns=jnp.array([[1.5, 1, 4]]),
        world_level=6,
    )

    icland_state = icland.init(key, icland_params, mjx_model)

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

        icland_state = icland.step(key, icland_state, icland_params, policy)
        mjx_data = icland_state.pipeline_state.mjx_data

        if len(frames) < mjx_data.time * 30:
            camera_pos, camera_dir = get_camera_info(
                icland_state, world_width, default_agent_1
            )
            f = render_frame(camera_pos, camera_dir, tilemap)
            frames.append(f)

    imageio.mimsave(video_name, frames, fps=30, quality=8)


if __name__ == "__main__":
    keys = [
        jax.random.PRNGKey(42),
        # jax.random.PRNGKey(420),
        # jax.random.PRNGKey(2004),
    ]
    for k in keys:
        render_video_multi_agent(
            k, RIGHT_POLICY, 3, 4, f"tests/video_output/world_convex_ma_{k[1]}_mjx.mp4"
        )
    # for i, preset in enumerate(SIMULATION_PRESETS):
    #     print(f"Running preset {i + 1}/{len(SIMULATION_PRESETS)}: {preset['name']}")
    #     render_video(
    #         jax.random.PRNGKey(42),
    #         preset["world"],
    #         preset["policy"],
    #         preset["duration"],
    #         f"scripts/video_output/{preset['name']}.mp4",
    #         agent_count=preset.get("agent_count", 1),
    #     )
    # for i, preset in enumerate(SIMULATION_PRESETS):
    #     print(f"Running preset {i + 1}/{len(SIMULATION_PRESETS)}: {preset['name']}")
    #     render_video(
    #         jax.random.PRNGKey(42),
    #         preset["world"],
    #         preset["policy"],
    #         preset["duration"],
    #         f"tests/video_output/{preset['name']}.mp4",
    #         agent_count=preset.get("agent_count", 1),
    #     )
