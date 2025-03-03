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
from typing import Any

import imageio

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


def interactive_simulation() -> None:
    """Runs an interactive simulation where you can change the agent's policy via keyboard input."""
    # Create the MuJoCo model from the .

    jax_key = jax.random.PRNGKey(0)
    icland_params = icland.sample(jax_key)
    mjx_model, mj_model = generate_base_model(
        icland.DEFAULT_CONFIG.max_world_width,
        icland.DEFAULT_CONFIG.max_world_depth,
        icland.DEFAULT_CONFIG.max_world_height,
        icland.DEFAULT_CONFIG.max_agent_count,
        icland.DEFAULT_CONFIG.max_sphere_count,
        icland.DEFAULT_CONFIG.max_cube_count,
    )

    icland_state = icland.init(icland_params)

    # Set up the camera.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # Use the first component id (e.g. the first agent's body) as the track target.
    cam.trackbodyid = icland_params.agent_info.body_ids[0]
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

    frame_rate = 30
    frames: list[Any] = []
    controlling = 0

    # Create the renderer.
    with mujoco.Renderer(mj_model) as renderer:
        while True:
            # Process any pending window events.
            cv2.waitKey(1)

            # Stop if simulation time exceeds the duration.
            mjx_data = icland_state.mjx_data

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
            if keyboard.is_pressed("1"):
                new_policy += TAG_AGENT_POLICY
            if keyboard.is_pressed("0"):
                controlling += 1
                controlling = controlling % 2
            if keyboard.is_pressed("2"):
                new_policy += GRAB_AGENT_POLICY

            # Update the current policy if it has changed.
            if not jnp.array_equal(new_policy, current_policy):
                current_policy = new_policy
                print(f"Time {mjx_data.time:.2f}: {current_policy}")

            # Step the simulation using the current_policy.
            icland_state, obs, rew = icland.step(
                icland_state,
                icland_params,
                jnp.array(
                    [current_policy, NOOP_POLICY]
                    if controlling == 0
                    else [NOOP_POLICY, current_policy]
                ),
            )

            # Get the latest simulation data.
            mjx_data = icland_state.mjx_data
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

            if len(frames) < mjx_data.time * frame_rate:
                frames.append(frame_bgr)

            # Display the frame.
            cv2.imshow(window_name, frame_bgr)

    cv2.destroyWindow(window_name)
    print("Interactive simulation ended.")
    print(f"Exporting video: {'controller.mp4'} number of frame {len(frames)}")
    imageio.mimsave(
        "scripts/video_output/controller.mp4", frames, fps=frame_rate, quality=8
    )


def sdfr_interactive_simulation() -> None:
    """Runs an interactive SDF simulation using a generated world and SDF rendering."""
    # Set up the JAX random key.
    jax_key = jax.random.PRNGKey(0)
    icland_params = icland.sample(jax_key)

    icland_state = icland.init(icland_params)

    # Take an initial step with the default (no-op) policy.
    current_policy = NOOP_POLICY

    # Set up an OpenCV window.
    window_name = "SDF Interactive Simulation"
    cv2.namedWindow(window_name)
    print("Starting SDF interactive simulation. Press 'q' to quit.")

    framerate = 30
    frames: list[Any] = []

    controlling = 0
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
        if keyboard.is_pressed("1"):
            new_policy += TAG_AGENT_POLICY
        if keyboard.is_pressed("0"):
            controlling += 1
            controlling = controlling % 2
        if keyboard.is_pressed("2"):
            new_policy += GRAB_AGENT_POLICY

        # Update the current policy if it has changed.
        if not jnp.array_equal(new_policy, current_policy):
            current_policy = new_policy
            print(f"Current policy updated: {current_policy}")

        # Step the simulation using the current policy.
        icland_state, obs, _ = icland.step(
            icland_state,
            icland_params,
            jnp.array(
                [current_policy, NOOP_POLICY]
                if controlling == 0
                else [NOOP_POLICY, current_policy]
            ),
        )

        # Render the frame using the SDF rendering callback.
        frame = obs.render[controlling]
        # Frame is of shape (w, h, 3) with values in [0, 1].
        # We repace all NaN values with 0 for OpenCV compatibility
        frame = np.nan_to_num(frame)
        resized_frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_NEAREST)
        # Convert the frame from RGB to BGR for OpenCV.
        frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

        if len(frames) < icland_state.mjx_data.time * framerate:
            frames.append(frame_bgr)

        cv2.imshow(window_name, frame_bgr)

    cv2.destroyWindow(window_name)
    print("SDF interactive simulation ended.")
    print(f"Exporting video: {'controller.mp4'} number of frame {len(frames)}")
    imageio.mimsave(
        "scripts/video_output/controller.mp4", frames, fps=framerate, quality=8
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-sdfr":
        sdfr_interactive_simulation()
    else:
        interactive_simulation()
