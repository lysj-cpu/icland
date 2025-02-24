"""This is a basic example of how to use the icland environment."""

import jax
import jax.numpy as jnp

import icland
from icland.types import *
from icland.world_gen.model_editing import generate_base_model

# Create a random key
key = jax.random.PRNGKey(42)

# Set global configuration
config = ICLandConfig(2, 2, 1, {}, 6)
# Sample initial conditions
icland_params = icland.sample(key, config)

# Initialize the environment
mjx_model, _ = generate_base_model(config)
init_state = icland.init(key, icland_params, mjx_model)

# Define an action to take
action = jnp.array([1, 0, 0])

# Take a step in the environment
next_state = icland.step(key, init_state, icland_params, action)

# Calculate the reward
if icland_params.reward_function is not None:
    reward = icland_params.reward_function(next_state.data)
    print(reward)
