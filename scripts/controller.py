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
import sys

from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame

# N.B. These need to be set before the mujoco imports.
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM (improves steps/sec on some GPUs)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import cv2  # For displaying frames and capturing window events.
import jax
import jax.numpy as jnp
import keyboard  # For polling the state of multiple keys simultaneously.
import mujoco
import numpy as np
from mujoco import mjx

import icland

# Import your policies and worlds from your assets.
from icland.presets import *
from icland.types import *
from icland.world_gen.model_editing import generate_base_model


def interactive_simulation(config: ICLandConfig) -> None:
    """Runs an interactive simulation where you can change the agent's policy via keyboard input."""
    # Create the MuJoCo model from the .
    icland_params = icland.sample(jax.random.PRNGKey(42), DEFAULT_CONFIG)
    mjx_model, mj_model = generate_base_model(DEFAULT_CONFIG)

    jax_key = jax.random.PRNGKey(42)
    icland_state = icland.init(jax_key, icland_params, mjx_model)

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
            if keyboard.is_pressed("up"):
                new_policy += LOOK_UP_POLICY
            if keyboard.is_pressed("down"):
                new_policy += LOOK_DOWN_POLICY

            # Update the current policy if it has changed.
            if not jnp.array_equal(new_policy, current_policy):
                current_policy = new_policy
                print(f"Time {mjx_data.time:.2f}: {current_policy}")

            # Step the simulation using the current_policy.
            icland_state = icland.step(
                jax_key, icland_state, icland_params, current_policy
            )
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


def sdfr_interactive_simulation(config: ICLandConfig) -> None:
    """Runs an interactive SDF simulation using a generated world and SDF rendering."""
    # Set up the JAX random key.
    jax_key = jax.random.PRNGKey(42)

    # World configuration.
    height, width = 10, 10

    # Sample the world (we follow render_sdfr's approach).
    # model = sample_world(height, width, 1000, jax_key, True, 1)
    # Create a dummy tilemap (all zeros) as in render_sdfr.
    tilemap = jnp.zeros((width, height, 4), dtype=np.int32)

    # Create the MuJoCo model using an EMPTY_WORLD MJCF string.
    # (Assumes EMPTY_WORLD is imported from icland.presets)
    icland_params = icland.sample(jax.random.PRNGKey(42), DEFAULT_CONFIG)
    mjx_model, mj_model = generate_base_model(DEFAULT_CONFIG)

    jax_key = jax.random.PRNGKey(42)
    icland_state = icland.init(jax_key, icland_params, mjx_model)
    # Take an initial step with the default (no-op) policy.
    current_policy = NOOP_POLICY
    icland_state = icland.step(jax_key, icland_state, icland_params, current_policy)

    # Set up default agent and world width for camera parameters.
    default_agent = 0
    world_width = tilemap.shape[1]

    # Define the frame callback using the SDF rendering functions.
    frame_callback = lambda state: render_frame(
        *get_agent_camera_from_mjx(state, world_width, default_agent),
        tilemap,
        view_width=96,
        view_height=72,
    )

    # Set up an OpenCV window.
    window_name = "SDF Interactive Simulation"
    cv2.namedWindow(window_name)
    print("Starting SDF interactive simulation. Press 'q' to quit.")

    while True:
        # Process any pending OpenCV window events.
        cv2.waitKey(1)

        # Quit if 'q' is pressed.
        if keyboard.is_pressed("q"):
            print("Quitting simulation.")
            break

        # Build the new policy based on keyboard input.
        new_policy = jnp.zeros_like(current_policy)
        if keyboard.is_pressed("w"):
            new_policy += FORWARD_POLICY
        if keyboard.is_pressed("s"):
            new_policy += BACKWARD_POLICY
        if keyboard.is_pressed("a"):
            new_policy += LEFT_POLICY
        if keyboard.is_pressed("d"):
            new_policy += RIGHT_POLICY
        if keyboard.is_pressed("left"):
            new_policy += ANTI_CLOCKWISE_POLICY
        if keyboard.is_pressed("right"):
            new_policy += CLOCKWISE_POLICY
        if keyboard.is_pressed("up"):
            new_policy += LOOK_UP_POLICY
        if keyboard.is_pressed("down"):
            new_policy += LOOK_DOWN_POLICY

        # Update the current policy if it has changed.
        if not jnp.array_equal(new_policy, current_policy):
            current_policy = new_policy
            print(f"Current policy updated: {current_policy}")

        # Step the simulation using the current policy.
        icland_state = icland.step(jax_key, icland_state, icland_params, current_policy)
        jax_key, _ = jax.random.split(jax_key)

        # Render the frame using the SDF rendering callback.
        frame = frame_callback(icland_state)  # type: ignore
        # Frame is of shape (w, h, 3) with values in [0, 1].
        # We repace all NaN values with 0 for OpenCV compatibility
        frame = np.nan_to_num(frame)
        # Convert the frame from RGB to BGR for OpenCV.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, frame_bgr)

    cv2.destroyWindow(window_name)
    print("SDF interactive simulation ended.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-sdfr":
        sdfr_interactive_simulation(DEFAULT_CONFIG)
    else:
        interactive_simulation(DEFAULT_CONFIG)
