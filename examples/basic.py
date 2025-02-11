"""This is a basic example of how to use the icland environment."""

import jax
import jax.numpy as jnp

import icland

# Create a random key
key = jax.random.PRNGKey(42)

# Sample initial conditions
icland_params = icland.sample(key)

# Initialize the environment
init_state = icland.init(key, icland_params)

# Define an action to take
action = jnp.array([1, 0, 0])

# Take a step in the environment
next_state = icland.step(key, init_state, icland_params, action)

# Calculate the reward
if icland_params.reward_function is not None:
    reward = icland_params.reward_function(next_state.data)
    print(reward)
