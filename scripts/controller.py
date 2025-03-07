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
    - Press '1' to attempt to tag an agent in front of you with TAG_POLICY.
    - Press '2' to attempt to grab a prop in front of you with GRAB_POLICY.
    - Press 'q' to quit the simulation.

This script is based on video_generator but instead of writing a video file, it displays frames in real time.
"""

import argparse
import os
from typing import Any

import imageio
import pygame

# N.B. These need to be set before the mujoco imports.
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM (improves steps/sec on some GPUs)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import cv2  # For displaying frames and capturing window events.
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import icland

# Import your policies and worlds from your assets.
from icland.presets import *
from icland.types import *
from icland.world_gen.model_editing import generate_base_model


def interactive_simulation(jax_key: jax.Array) -> None:
    """Runs an interactive simulation where you can change the agent's policy via keyboard input."""
    # Create the MuJoCo model from the .
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
    window_name = "Interactive Simulation"

    window_size = (960, 720)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption(window_name)
    print("Starting interactive simulation. Press 'q' to quit.")

    framerate = 30
    frames: list[Any] = []
    clock = pygame.time.Clock()

    controlling = 0

    # Create the renderer.
    with mujoco.Renderer(mj_model) as renderer:
        while True:
            # Update mjx data
            mjx_data = icland_state.mjx_data

            # Quit if 'q' is pressed
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            new_policy = jnp.zeros_like(current_policy)
            # Determine which image to display based on key combinations
            if keys[pygame.K_w]:
                new_policy += FORWARD_POLICY
            if keys[pygame.K_a]:
                new_policy += LEFT_POLICY
            if keys[pygame.K_s]:
                new_policy += BACKWARD_POLICY
            if keys[pygame.K_d]:
                new_policy += RIGHT_POLICY
            if keys[pygame.K_LEFT]:
                new_policy += ANTI_CLOCKWISE_POLICY
            if keys[pygame.K_RIGHT]:
                new_policy += CLOCKWISE_POLICY
            if keys[pygame.K_UP]:
                new_policy += LOOK_UP_POLICY
            if keys[pygame.K_DOWN]:
                new_policy += LOOK_DOWN_POLICY
            if keys[pygame.K_1]:
                new_policy += TAG_AGENT_POLICY
            if keys[pygame.K_0]:
                controlling = (controlling + 1) % icland_params.agent_info.agent_count
            if keys[pygame.K_2]:
                new_policy += GRAB_AGENT_POLICY

            # Update the current policy if it has changed.
            if not jnp.array_equal(new_policy, current_policy):
                current_policy = new_policy
                print(f"Current policy updated: {current_policy}")

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
            frame = (frame * 0xFF).astype(np.uint8)
            # Convert the frame from RGB to BGR for OpenCV.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            resized_frame = cv2.resize(
                frame, window_size, interpolation=cv2.INTER_NEAREST
            )
            frame_bytes = resized_frame.tobytes()
            screen.blit(
                pygame.image.frombuffer(frame_bytes, resized_frame.shape[1::-1], "RGB"),
                (0, 0),
            )
            pygame.display.flip()

            clock.tick(framerate)

            if len(frames) < icland_state.mjx_data.time * framerate:
                frames.append(resized_frame)

    cv2.destroyWindow(window_name)
    print("Interactive simulation ended.")
    print(f"Exporting video: {'controller.mp4'} with {len(frames)} frames.")
    imageio.mimsave(
        "scripts/video_output/controller.mp4", frames, fps=framerate, quality=8
    )


def sdfr_interactive_simulation(jax_key: jax.Array) -> None:
    """Runs an interactive SDF simulation using a generated world and SDF rendering."""
    # Set up the JAX random key.
    icland_params = icland.sample(jax_key)

    icland_state = icland.init(icland_params)

    # Take an initial step with the default (no-op) policy.
    current_policy = NOOP_POLICY

    # Set up an OpenCV window.
    window_name = "SDF Interactive Simulation"

    window_size = (960, 720)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption(window_name)
    print("Starting SDF interactive simulation. Press 'q' to quit.")

    framerate = 30
    frames: list[Any] = []
    clock = pygame.time.Clock()

    controlling = 0
    while True:
        # Quit if 'q' is pressed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        new_policy = jnp.zeros_like(current_policy)
        # Determine which image to display based on key combinations
        if keys[pygame.K_w]:
            new_policy += FORWARD_POLICY
        if keys[pygame.K_a]:
            new_policy += LEFT_POLICY
        if keys[pygame.K_s]:
            new_policy += BACKWARD_POLICY
        if keys[pygame.K_d]:
            new_policy += RIGHT_POLICY
        if keys[pygame.K_LEFT]:
            new_policy += ANTI_CLOCKWISE_POLICY
        if keys[pygame.K_RIGHT]:
            new_policy += CLOCKWISE_POLICY
        if keys[pygame.K_UP]:
            new_policy += LOOK_UP_POLICY
        if keys[pygame.K_DOWN]:
            new_policy += LOOK_DOWN_POLICY
        if keys[pygame.K_1]:
            new_policy += TAG_AGENT_POLICY
        if keys[pygame.K_0]:
            controlling = (controlling + 1) % icland_params.agent_info.agent_count
        if keys[pygame.K_2]:
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
        frame = (frame * 0xFF).astype(np.uint8)
        # Convert the frame from RGB to BGR for OpenCV.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        resized_frame = cv2.resize(frame, window_size, interpolation=cv2.INTER_NEAREST)
        frame_bytes = resized_frame.tobytes()
        screen.blit(
            pygame.image.frombuffer(frame_bytes, resized_frame.shape[1::-1], "RGB"),
            (0, 0),
        )
        pygame.display.flip()

        clock.tick(framerate)

        if len(frames) < icland_state.mjx_data.time * framerate:
            frames.append(resized_frame)

    pygame.quit()
    print("SDF interactive simulation ended.")
    print(f"Exporting video: {'controller.mp4'} number of frame {len(frames)}")
    imageio.mimsave(
        "scripts/video_output/controller.mp4", frames, fps=framerate, quality=8
    )


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Run interactive simulations with optional key number."
    )

    # Add optional argument for key seed with default value 0
    parser.add_argument(
        "--key", type=int, default=0, help="Specify the key seed (default is 0)."
    )

    # Add optional flag for SDFR simulation
    parser.add_argument(
        "-sdfr",
        action="store_true",
        help="Run the SDFR interactive simulation (default: run the Mujoco renderer).",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Generate the PRNG key using the specified key_no
    key = jax.random.PRNGKey(args.key)

    # Determine which simulation to run
    if args.sdfr:
        sdfr_interactive_simulation(key)
    else:
        interactive_simulation(key)
