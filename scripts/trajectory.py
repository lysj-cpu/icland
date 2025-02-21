"""Trajectory plotting."""

import os
import shutil

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
from video_generator import _generate_mjcf_string

import icland
from icland.presets import *
from icland.renderer.renderer import get_agent_camera_from_mjx
from icland.types import *
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.tile_data import TILECODES


def plot_trajectories_multi_policies(
    key: jax.Array,
    policies: list[jax.Array],
    switch_intervals: list[float],
    duration: int,
) -> list[jax.Array]:
    """Renders a video where the agent follows multiple policies sequentially.

    Args:
        key: Random seed key.
        policies: A list of policy arrays, applied sequentially.
        switch_intervals: Time intervals at which to switch to the next policy.
        duration: Total duration of the simulation.
        video_name: Output video file name.
    """
    print(f"Sampling world with key {key[1]}")
    model = sample_world(10, 10, 1000, key, True, 1)
    tilemap = export(model, TILECODES, 10, 10)
    spawnpoints = sample_spawn_points(key, tilemap, num_objects=1)
    pieces = create_world(tilemap)
    temp_dir = "temp"
    os.makedirs(f"{temp_dir}", exist_ok=True)
    export_stls(pieces, f"{temp_dir}/{temp_dir}")

    xml_str = _generate_mjcf_string(tilemap, spawnpoints, f"{temp_dir}/")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)

    icland_state = icland.init(key, icland_params, mj_model)
    mjx_data = icland_state.pipeline_state.mjx_data
    trajectory: list[Any] = []

    current_policy_idx = 0
    policy = policies[current_policy_idx]

    print(f"Starting simulation of world with key {key[1]}")
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

        if len(trajectory) < mjx_data.time * 30:
            agent_pos = mjx_data.xpos[
                icland_state.pipeline_state.component_ids[default_agent_1, 0]
            ][:3]
            print("Agent pos:", agent_pos)
            trajectory.append(agent_pos)

    shutil.rmtree(f"{temp_dir}")
    return trajectory


def plot_multiple_trajectories(
    trajectories: list[list[jax.Array]], filename: str = "trajectories"
) -> None:
    """Plots multiple trajectories in 3D space, using different colors and adding a legend.

    Parameters:
    - trajectories: list of lists of JAX numpy arrays,
                    where each inner list represents a trajectory as a sequence of (x, y, z) positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define a color cycle
    colors = itertools.cycle(plt.get_cmap("tab10").colors)  # type: ignore

    for i, trajectory in enumerate(trajectories):
        positions = jnp.stack(trajectory).T  # Shape: (3, N)
        color = next(colors)
        ax.plot(
            positions[0],
            positions[1],
            positions[2],
            marker="o",
            linestyle="-",
            label=f"Trajectory {i + 1}",
            color=color,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_title("Multiple Agent Trajectories")
    ax.legend()

    plt.savefig(f"./tests/trajectory_output/{filename}.png")


# Example usage
if __name__ == "__main__":
    trajectories = []

    # 2D list (num_trajs, num_frames, 3)
    keys = [
        # jax.random.PRNGKey(216),
        jax.random.PRNGKey(42),
        jax.random.PRNGKey(420),
        # jax.random.PRNGKey(2004),
        # jax.random.PRNGKey(141),
        # jax.random.PRNGKey(8),
        # jax.random.PRNGKey(5120),
        # jax.random.PRNGKey(77),
        # jax.random.PRNGKey(926),
        # jax.random.PRNGKey(6561)
    ]
    for k in keys:
        trajectories.append(
            plot_trajectories_multi_policies(
                k,
                [FORWARD_POLICY, LEFT_POLICY, FORWARD_POLICY, RIGHT_POLICY],
                [1.0, 2.0, 3.0],
                4,
            )
        )

    plot_multiple_trajectories(trajectories)
