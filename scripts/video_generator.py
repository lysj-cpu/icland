"""This script generates videos of simulations using predefined simulation presets.

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
from collections.abc import Callable
from typing import Any

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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
from icland.renderer.renderer import (
    PlayerInfo,
    PropInfo,
    generate_colormap,
    get_agent_camera_from_mjx,
    render_frame,
    render_frame_with_objects,
)
from icland.types import *
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.model_editing import _edit_mj_model_data, generate_base_model
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
    policy_switcher: Callable[[float, jax.Array], jax.Array] | None = None,
) -> list[Any]:
    """Runs the simulation loop until `duration` and collects frames using `frame_callback`.

    Optionally, a `policy_switcher` function may update the policy based on simulation time.
    """
    frames = []  # type: list[Any]
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
    tilemap = np.zeros((width, height, 4), dtype=np.int32)
    mjx_model, mj_model = generate_base_model(ICLandConfig(width, height, 1, {}, 6))
    icland_params = ICLandParams(
        world=tilemap,
        reward_function=None,
        agent_spawns=jnp.array([[0, 0, 0]]),
        world_level=6,
    )

    icland_state = icland.init(key, icland_params, mjx_model)
    icland_state = icland.step(key, icland_state, icland_params, policy)

    default_agent = 0
    world_width = tilemap.shape[1]
    # get_camera_info = jax.jit(get_agent_camera_from_mjx)
    frame_callback = lambda state: render_frame(
        *get_agent_camera_from_mjx(state, world_width, default_agent),
        tilemap,
        view_width=96,
        view_height=72,
    )

    print(f"Starting simulation: {video_name}")
    frames = _simulate_frames(
        key, icland_state, icland_params, duration, frame_callback, policy
    )
    print(f"Exporting video: {video_name} number of frame {len(frames)}")
    imageio.mimsave(video_name, frames, fps=30, quality=8)


def __combine_frames(
    frames_list: list[np.ndarray[Any, np.dtype[np.float32]]],
    grid_shape: tuple[int, int] | None = None,
    padding: int = 0,
    pad_value: int = 0,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    frames = np.array(frames_list)  # Ensure frames is an array
    n, h, w, c = frames.shape

    # If no grid shape is given, choose a near-square grid.
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape

    # Compute the dimensions of the output grid image.
    grid_h = rows * h + (rows + 1) * padding
    grid_w = cols * w + (cols + 1) * padding

    # Initialize the grid with the pad_value.
    grid = np.full((grid_h, grid_w, c), pad_value, dtype=frames.dtype)

    # Place each frame in the grid.
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        top = padding + row * (h + padding)
        left = padding + col * (w + padding)
        grid[top : top + h, left : left + w, :] = frame

    return (grid * 255).astype(np.uint8)


def render_video_multi_agent(
    key: jax.Array,
    duration: int,
    policies: list[jax.Array],
    video_name: str,
    height: int = 10,
    width: int = 10,
) -> None:
    """Renders separate first person view videos for multiple agents."""
    agent_count = len(policies)

    # Initialize global config
    config = ICLandConfig(width, height, agent_count, {}, 6)
    mjx_model, _ = generate_base_model(config)
    print(f"Sampling world with key {key[1]}...")

    # Use ICLand API to sample and initialize world
    icland_params = icland.sample(key, config)
    # icland_params = icland_params.replace(
    #     agent_spawns=jnp.array([[1 + i, 1, 4] for i in range(agent_count)])
    # )
    icland_state = icland.init(key, icland_params, mjx_model)
    icland_state = icland.step(key, icland_state, icland_params, jnp.array(policies))
    tilemap = icland_state.pipeline_state.world
    print(f"Init mjx model and data...")
    mjx_data = icland_state.pipeline_state.mjx_data

    # Store frames for each agent
    agent_frames: list[Any] = []

    print(f"Starting simulation for {agent_count} agents: {video_name}")
    last_printed_time = -0.1
    world_width = tilemap.shape[0]

    # JIT-compiled
    get_camera_info = jax.jit(get_agent_camera_from_mjx)

    all_colors = jnp.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    )
    colors = all_colors[
        jax.random.randint(
            jax.random.PRNGKey(0), (agent_count,), 0, all_colors.shape[0]
        )
    ]
    FPS = 60

    while mjx_data.time < duration:
        if int(mjx_data.time * 10) != int(last_printed_time * 10):
            print(f"Time: {mjx_data.time:.1f}")
            last_printed_time = mjx_data.time

        icland_state = icland.step(
            key, icland_state, icland_params, jnp.array(policies)
        )
        mjx_data = icland_state.pipeline_state.mjx_data

        if len(agent_frames) < mjx_data.time * FPS:
            camera_pos, camera_dir = jax.vmap(get_camera_info, in_axes=(None, None, 0))(
                icland_state, world_width, jnp.arange(agent_count)
            )
            players = jax.vmap(
                lambda x: PlayerInfo(
                    pos=camera_pos[x].at[1].add(0.2),
                    col=colors[x],
                )
            )(jnp.arange(agent_count))

            props = PropInfo(
                prop_type=jnp.array([0]),
                pos=jnp.empty((1, 3)),
                rot=jnp.array([[1, 0, 0, 0]]),
                col=jnp.empty((1, 3)),
            )

            temp_agent_frames: list[Any] = []
            for aid in range(agent_count):
                f = render_frame_with_objects(
                    camera_pos[aid],
                    camera_dir[aid],
                    tilemap,
                    generate_colormap(key, width, height),
                    players,
                    props,
                    view_width=360,
                    view_height=240,
                )
                # print("Rendered frame")
                temp_agent_frames.append(f)

            # Combine frames for this timestep and append to all_combined_frames
            combined_frame = __combine_frames(temp_agent_frames)
            agent_frames.append(combined_frame)
        # print("Got camera angle")

    # Save the combined video
    imageio.mimsave(video_name, agent_frames, fps=FPS, quality=8)


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

    The policy is switched at the times specified in `switch_intervals`.
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
        if (
            current_policy_idx < len(switch_intervals)
            and time >= switch_intervals[current_policy_idx]
        ):
            current_policy_idx += 1
            if current_policy_idx < len(policies):
                print(f"Switching policy at {time:.1f}s")
                return policies[current_policy_idx]
        return current

    print(f"Starting simulation: {video_name}")
    frames = _simulate_frames(
        key,
        icland_state,
        icland_params,
        duration,
        frame_callback,
        current_policy,
        policy_switcher=policy_switcher,
    )

    imageio.mimsave(video_name, frames, fps=30, quality=8)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    keys = [
        jax.random.PRNGKey(42),
        jax.random.PRNGKey(420),
        jax.random.PRNGKey(2004),
    ]
    for k in keys:
        render_video_multi_agent(
            k,
            4,
            [FORWARD_POLICY, CLOCKWISE_POLICY, BACKWARD_POLICY, ANTI_CLOCKWISE_POLICY],
            f"tests/video_output/world_convex_ma_{k[1]}_mjx.mp4",
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
