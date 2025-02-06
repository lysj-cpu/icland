#!/usr/bin/env python3
"""Interactive simulation.

    Renders each frame of the simulation to an OpenCV window and lets you change the agent's policy using keyboard input.

Controls:
    - Hold 'w' to command the agent with FORWARD_POLICY.
    - Hold 's' to command the agent with BACKWARD_POLICY.
    - Hold 'a' to command the agent with LEFT_POLICY.
    - Hold 'd' to command the agent with RIGHT_POLICY.
    - Hold the left arrow key to command the agent with ANTI_CLOCKWISE_POLICY.
    - Hold the right arrow key to command the agent with CLOCKWISE_POLICY.
    - Press 'q' to quit the simulation.

This script is based on video_generator but instead of writing a video file, it displays frames in real time.
"""

import os

# N.B. These need to be set before the mujoco imports.
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM (improves steps/sec on some GPUs)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from typing import Any, Dict, List

import cv2  # For displaying frames and capturing window events.
import jax
import jax.numpy as jnp
import keyboard  # For polling the state of multiple keys simultaneously.
import mujoco

# Import your policies and worlds from your assets.
from assets.policies import *
from assets.worlds import RAMP_30  # You can choose any world you like.
from mujoco import mjx

import icland
from icland.types import *

# (Optional) A list of simulation presets; here we just pick one for interactivity.
SIMULATION_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "ramp_30",
        "world": RAMP_30,
        "policy": FORWARD_POLICY,  # This default will be overridden interactively.
        "duration": 30,  # Duration in seconds (or you can let it run until quit)
        "agent_count": 1,
    },
]


def interactive_simulation(model_xml: str, duration: int, agent_count: int = 1) -> None:
    """Runs an interactive simulation where you can change the agent's policy via keyboard input.

    Args:
        model_xml (str): XML string defining the MuJoCo model.
        duration (int): Maximum duration (in seconds) to run the simulation.
        agent_count (int): Number of agents in the simulation.
    """
    # Create the MuJoCo model from the XML string.
    mj_model = mujoco.MjModel.from_xml_string(model_xml)

    # Set up the simulation parameters.
    icland_params = ICLandParams(model=mj_model, game=None, agent_count=agent_count)
    jax_key = jax.random.PRNGKey(42)
    icland_state = icland.init(jax_key, icland_params)

    # Set up the camera.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # Use the first component id (e.g. the first agent's body) as the track target.
    cam.trackbodyid = icland_state.pipeline_state.component_ids[0, 0]
    cam.distance = 1.5
    cam.azimuth = 0.0
    cam.elevation = -30.0
    # Adjust the camera to be behind the agent.

    # Set up visualization options.
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # Initialize the current policy (action) to NOOP_POLICY.
    current_policy = NOOP_POLICY
    print("Starting interactive simulation.")

    # Create a window using OpenCV.
    window_name = "Interactive Simulation"
    cv2.namedWindow(window_name)

    # Create the renderer.
    with mujoco.Renderer(mj_model) as renderer:
        while True:
            # Process any pending window events.
            cv2.waitKey(1)

            # Stop if simulation time exceeds the duration.
            mjx_data = icland_state.pipeline_state.mjx_data
            if mjx_data.time >= duration:
                print("Reached maximum duration.")
                break

            # Quit if 'q' is pressed.
            if keyboard.is_pressed("q"):
                print("Quitting simulation.")
                break

            # Build up the new policy by checking each key's state.
            new_policy = jnp.zeros_like(current_policy)
            if keyboard.is_pressed("w"):
                new_policy += FORWARD_POLICY
            if keyboard.is_pressed("s"):
                new_policy += BACKWARD_POLICY
            if keyboard.is_pressed("a"):
                new_policy += LEFT_POLICY
            if keyboard.is_pressed("d"):
                new_policy += RIGHT_POLICY
            # Use the key names recognized by the keyboard module for arrow keys.
            if keyboard.is_pressed("left"):
                new_policy += ANTI_CLOCKWISE_POLICY
            if keyboard.is_pressed("right"):
                new_policy += CLOCKWISE_POLICY

            # Update the current policy if it has changed.
            if not jnp.array_equal(new_policy, current_policy):
                current_policy = new_policy
                print(f"Time {mjx_data.time:.2f}: {current_policy}")

            # Step the simulation using the current_policy.
            icland_state = icland.step(jax_key, icland_state, None, current_policy)
            # (Optional) Update the JAX random key.
            jax_key, _ = jax.random.split(jax_key)

            # Get the latest simulation data.
            mjx_data = icland_state.pipeline_state.mjx_data
            mj_data = mjx.get_data(mj_model, mjx_data)

            # Update the scene.
            mujoco.mjv_updateScene(
                mj_model,
                mj_data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                renderer.scene,
            )

            # Render the frame.
            frame = renderer.render()
            # Convert the frame from RGB to BGR for OpenCV.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the frame.
            cv2.imshow(window_name, frame_bgr)

    cv2.destroyWindow(window_name)
    print("Interactive simulation ended.")


if __name__ == "__main__":
    # For this example, we choose the first preset.
    preset = SIMULATION_PRESETS[0]
    print(f"Running preset: {preset['name']}")
    interactive_simulation(
        preset["world"],
        preset["duration"],
        agent_count=preset.get("agent_count", 1),
    )
