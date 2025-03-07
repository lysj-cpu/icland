"""This is a basic example of how to use the icland environment."""

import jax
import jax.numpy as jnp

import icland
from icland.types import *

# Create a random key
key = jax.random.PRNGKey(42)

# Sample initial conditions
config = icland.config(
    5,
    5,
    6,
    1,
    0,
    0,
)
icland_params: ICLandParams = icland.sample(key, config)

state = icland.init(icland_params)

agent_count = icland_params.agent_info.agent_count

batched_action = jnp.array([1, 0, 0, 0, 0, 0])

# Take a step in the environment
state = icland.step(state, icland_params, batched_action)

# Calculate the reward
# if icland_params.reward_function is not None:
#     reward = icland_params.reward_function(next_state.data)
#     print(reward)
