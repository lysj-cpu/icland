"""This is a basic example of how to use the icland environment with video recording."""

from typing import Any

# import cv2  # For displaying frames and capturing window events.
import imageio
import jax
import jax.numpy as jnp

# import keyboard  # For polling the state of multiple keys simultaneously.
import numpy as np

import icland
from icland.types import *

# Create a random key
key = jax.random.PRNGKey(2004)
FPS = 30

# Sample initial conditions
config = icland.config(
    10,
    10,
    6,
    4,
    3,
    3,
)
icland_params: ICLandParams = icland.sample(key, config)

state = icland.init(icland_params)

agent_count = icland_params.agent_info.agent_count

batched_action = jax.vmap(
    lambda k: jnp.concatenate([jax.random.randint(k, (2,), -1, 2), jnp.zeros((4,))])
)(jax.random.split(key, 4))

# Take a step in the environment
# while True:
#     state, obs, rew = icland.step(state, icland_params, batched_action)
#     print(state.mjx_data.time)
# Set up an OpenCV window.
window_name = "ICLand Video"
# cv2.namedWindow(window_name)
# print("Starting SDF interactive simulation. Press 'q' to quit.")


def __combine_frames(
    frames_list: list[np.ndarray[Any, np.dtype[np.float32]]]
    | jax.Array
    | np.ndarray[tuple[int, ...], np.dtype[Any]],
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


agent_frames: list[Any] = []
TIME = 4
timestep = 0
while state.mjx_data.time < TIME:
    # Process any pending OpenCV window events.
    # cv2.waitKey(1)

    # Quit if 'q' is pressed.
    # if keyboard.is_pressed("q"):
    #     print("Quitting simulation.")
    #     break

    # Build the new policy based on keyboard input.

    # Step the simulation using the current policy.
    state, obs, rew = icland.step(state, icland_params, batched_action)

    # Render the frame using the SDF rendering callback.
    frame = obs.render
    frame_rgb = np.nan_to_num(frame)
    frame_rgb_combined = __combine_frames(frame_rgb[:agent_count].astype("float32"))
    # Frame is of shape (w, h, 3) with values in [0, 1].
    # We repace all NaN values with 0 for OpenCV compatibility
    # frames = __combine_frames(frames_list=frame)
    if len(agent_frames) < state.mjx_data.time * FPS:
        print("Time:", state.mjx_data.time)
        agent_frames.append(frame_rgb_combined)
    # Convert the frame from RGB to BGR for OpenCV.
    # frame_bgr = cv2.cvtColor(frame_rgb_combined, cv2.COLOR_RGB2BGR)
    # frame_bgr = cv2.resize(frame_bgr, (256, 144))
    # cv2.imshow(window_name, frame_bgr)
    timestep += 1

# cv2.destroyWindow(window_name)
imageio.mimsave(
    "tests/video_output/ICLand_FPS_ma.mp4", agent_frames, fps=FPS, quality=8
)
print("ICLand video ended.")

# Calculate the reward
# if icland_params.reward_function is not None:
#     reward = icland_params.reward_function(next_state.data)
#     print(reward)
