"""Example of using the environment in a batched setting."""

import jax
import jax.numpy as jnp

import icland
from icland.types import *
from icland.world_gen.model_editing import generate_base_model

SEED = 42
BATCH_SIZE = 8

# Benchmark parameters
key = jax.random.PRNGKey(SEED)

# Set global configuration
config = ICLandConfig(
    world_width=2, world_height=2, max_agent_count=1, prop_counts={}, max_world_level=6
)
mjx_model, _ = generate_base_model(config)

# Sample initial conditions
keys = jax.random.split(key, BATCH_SIZE)
icland_params = jax.vmap(icland.sample, in_axes=(0, None))(keys, config)

# Initialize the environment
init_states = jax.vmap(icland.init, in_axes=(0, 0, None))(
    keys, icland_params, mjx_model
)

# Batched step function
batched_step = jax.vmap(icland.step, in_axes=(0, 0, 0, 0))

# Define actions to take
actions = jnp.array([[1, 0, 0] for _ in range(BATCH_SIZE)])

# Optionally, regenerate the keys
keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)

# Take a step in the environment
icland_states = batched_step(keys, init_states, icland_params, actions)

# Calculate the reward
if icland_params.reward_function is not None:
    reward = icland_params.reward_function(icland_states.data)
    print(reward)
