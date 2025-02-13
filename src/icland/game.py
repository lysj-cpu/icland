"""Generates a game for the ICLand environment."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .constants import *
from .types import *

POSITION_RANGE = (-5, 5)
ACCEPTABLE_DISTANCE = 0.5


def generate_game(
    key: PRNGKeyArray, agent_count: int
) -> Callable[[ICLandInfo], jax.Array]:
    """Generate a game using the given random key and agent count.

    This function randomly selects one of two modes:
      - Translation mode: agents are rewarded when their (x, y) position is within
        ACCEPTABLE_DISTANCE of a target position.
      - Rotation mode: agents are rewarded when their rotation is within
        ACCEPTABLE_DISTANCE of a target rotation.

    The selection is made using the provided key.

    Args:
        key: Random key for generating the game.
        agent_count: Number of agents in the environment.

    Returns:
        A function that takes an ICLandInfo and returns a reward array.
    """
    # Split the key so that one part is used to decide the mode and another for sampling the target.
    key, mode_key, target_key = jax.random.split(key, 3)
    # Randomly decide the mode: True for translation mode, False for rotation mode.
    is_translation_mode = jax.random.bernoulli(mode_key)

    if is_translation_mode:
        # Translation mode: sample a target (x, y) position for each agent.
        target_position = jax.random.uniform(
            target_key,
            (agent_count, 2),
            minval=POSITION_RANGE[0],
            maxval=POSITION_RANGE[1],
        )

        def reward_function(info: ICLandInfo) -> jax.Array:
            """Compute reward based on agent position relative to a target position."""
            # Extract the first two coordinates (x, y) of agent positions.
            # agent_positions = info.agent_positions[:, :2]
            agent_positions = jnp.array([agent.position[:2] for agent in info.agents])
            # Compute Euclidean distance from each agent to its target.
            distance = jnp.linalg.norm(agent_positions - target_position, axis=1)
            # Reward is 1 if the distance is less than the acceptable threshold.
            reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1.0, 0.0)
            return reward

        return reward_function

    else:
        # Rotation mode: sample a target rotation for each agent.
        target_rotation = jax.random.uniform(
            target_key, (agent_count, 1), minval=0, maxval=3
        )

        def reward_function(info: ICLandInfo) -> jax.Array:
            """Compute reward based on agent rotation relative to a target rotation."""
            # Extract the agent rotations.
            # agent_rotation = info.agent_rotations
            agent_rotation = jnp.array([agent.rotation for agent in info.agents])
            # Compute the absolute difference between agent rotation and target.
            distance = jnp.abs(agent_rotation - target_rotation)
            # Reward is 1 if the rotation difference is within the acceptable threshold.
            reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1.0, 0.0)
            return reward

        return reward_function
