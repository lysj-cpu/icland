"""Example of using the environment in a batched setting."""

import jax

import icland
from icland.types import *

SEED = 42
BATCH_SIZE = 8

# Benchmark parameters
key = jax.random.PRNGKey(SEED)

# Set global configuration
config = icland.config(
    2,
    2,
    6,
    1,
    0,
    0,
)

# Sample initial conditions
keys = jax.random.split(key, BATCH_SIZE)
icland_params = jax.vmap(icland.sample, in_axes=(0, None))(keys, config)

icland_state = jax.vmap(icland.init, in_axes=(0,))(icland_params)

print(icland_state)

# # Initialize the environment
# init_states = jax.vmap(icland.init, in_axes=(0, 0, None))(
#     keys, icland_params, mjx_model
# )

# # Batched step function
# batched_step = jax.vmap(icland.step, in_axes=(0, 0, 0, 0))

# # Define actions to take
# actions = jnp.array([[1, 0, 0] for _ in range(BATCH_SIZE)])

# # Optionally, regenerate the keys
# keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)

# # Take a step in the environment
# icland_states = batched_step(keys, init_states, icland_params, actions)

# # Calculate the reward
# if icland_params.reward_function is not None:
#     reward = icland_params.reward_function(icland_states.data)
#     print(reward)
