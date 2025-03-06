"""This is a basic example of how to use the icland environment with agent monitoring."""

import time

import jax
import jax.numpy as jnp

import icland
from icland.server.trajectory_poster import WebSocketTrajectoryPoster
from icland.server.trajectory_server import TrajectoryManager
from icland.types import *

# Create a random key
key = jax.random.PRNGKey(0)

# Sample initial conditions
config = icland.config(
    5,
    5,
    6,
    2,
    2,
    2,
)
icland_params: ICLandParams = icland.sample(key, config)

icland_state = icland.init(icland_params)

agent_count = icland_params.agent_info.agent_count

# Simulate for 200 frames, i.e. 10 seconds
NUM_STEPS = 200

# Start the trajectory manager server
tm = TrajectoryManager()
tm.start()
time.sleep(1)

# Create a poster
poster = WebSocketTrajectoryPoster()

# Start posting simulation data
poster.connect()
step = 0
try:
    while step < NUM_STEPS:
        icland_actions = jnp.array(  # Dummy policy placeholder
            [
                [
                    0.5,
                    -0.5,
                    0,
                    0,
                    0,
                    0,
                ],
                [-1, 0, 0, 0, 1, 0],
            ]
        )
        icland_state, _, rew = icland.step(icland_state, icland_params, icland_actions)

        if poster.stop_event.is_set():
            break

        poster.post_icland_data(step, icland_params, icland_state, icland_actions, rew)

        step += 1

    # Tell the server to terminate the simulation.
    poster.end_simulation()

    # Allow broadcast to propagate
    time.sleep(2)
finally:
    # Disconnect.
    poster.sio.disconnect()

tm.stop()
