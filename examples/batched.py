"""Example of using the environment in a batched setting."""

import jax
import jax.numpy as jnp

import icland

SEED = 42
BATCH_SIZE = 8

# Benchmark parameters
key = jax.random.PRNGKey(SEED)

# Sample initial conditions
icland_params = icland.sample(key)

# Initialize the environment
init_state = icland.init(key, icland_params)

# Batched step function
batched_step = jax.vmap(icland.step, in_axes=(0, 0, icland_params, 0))

# Prepare batch
icland_states = jax.tree.map(lambda x: jnp.stack([x] * BATCH_SIZE), init_state)

# Define actions to take
actions = jnp.array([[1, 0, 0] for _ in range(BATCH_SIZE)])

# Split the key
keys = jax.random.split(key, BATCH_SIZE)

# Take a step in the environment
icland_states = batched_step(keys, icland_states, icland_params, actions)

# Calculate the reward
if icland_params.reward_function is not None:
    reward = icland_params.reward_function(icland_states.data)
    print(reward)
