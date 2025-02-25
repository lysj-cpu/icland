"""This is a basic example of how to use the icland environment."""

import jax
import jax.numpy as jnp

import icland
from icland.types import *
from icland.world_gen.model_editing import generate_base_model

# Create a random key
key = jax.random.PRNGKey(42)

# Sample initial conditions
icland_params = icland.sample(key)

init_state = icland.init(icland_params)

action = ICLandAction(
    forward=1.0,
    right=0.0,
    yaw=0.0,
    pitch=0.0,
    grab=0,
    tag=0
)

batched_action = jax.tree.map(
        lambda x: jnp.stack([x] * int(icland_params.agent_info.agent_count)), action
    )

# Take a step in the environment
next_state = icland.step(init_state, icland_params, batched_action)

# print(next_state)

# Calculate the reward
# if icland_params.reward_function is not None:
#     reward = icland_params.reward_function(next_state.data)
#     print(reward)
