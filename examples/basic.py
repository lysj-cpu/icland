"""This is a basic example of how to use the icland environment."""

import jax
import jax.numpy as jnp

import icland
from icland.types import *

# Create a random key
key = jax.random.PRNGKey(42)

# Sample initial conditions
icland_params: ICLandParams = icland.sample(key)

state = icland.init(icland_params)

agent_count = icland_params.agent_info.agent_count

batched_action = jnp.array([1, 0, 0, 0, 0, 0])

# Take a step in the environment
while True:
    state, obs, rew = icland.step(state, icland_params, batched_action)
    print(state.mjx_data.time)

# Calculate the reward
# if icland_params.reward_function is not None:
#     reward = icland_params.reward_function(next_state.data)
#     print(reward)
