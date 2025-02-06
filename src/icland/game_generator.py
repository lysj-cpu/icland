"""Generates a game for the ICLand environment."""

from typing import Callable

import jax
import jax.numpy as jnp

from .constants import *
from .types import *

POSITION_RANGE = (-5, 5)
ACCEPTABLE_DISTANCE = 0.5


def generate_game(
    key: jax.Array, agent_count: int
) -> Callable[[ICLandInfo], jax.Array]:
    """Generate a game using the given model and agent count.

    Args:
        key: Random key for generating the game.
        agent_count: Number of agents in the environment.

    Returns:
        Function that generates a game given the initial state.
    """
    # Generate an x, y tuple for each agent to target
    # target_position = jax.random.uniform(key, (agent_count, 2), minval=POSITION_RANGE[0], maxval=POSITION_RANGE[1])

    # def translation_reward_function(info: ICLandInfo) -> jax.Array:
    #     """Generate a game using the given initial state.

    #     Args:
    #         info: Information about the ICLand environment.

    #     Returns:
    #         The reward of the game.
    #     """

    #     # Calculate the distance between the agent and the target
    #     agent_positions = info.agent_positions[:, :2]
    #     distance = jnp.linalg.norm(agent_positions - target_position, axis=1)

    #     # Calculate the reward based on the distance
    #     reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1, 0)
    #     return reward

    target_rotation = jax.random.uniform(key, (agent_count, 1), minval=0, maxval=3)

    def rotation_reward_function(info: ICLandInfo) -> jax.Array:
        """Generate a game using the given initial state.

        Args:
            info: Information about the ICLand environment.

        Returns:
            The reward of the game.
        """
        # Calculate the distance between the agent and the target
        agent_rotation = info.agent_rotations

        print(agent_rotation, target_rotation)

        distance = jnp.abs(agent_rotation - target_rotation)

        # Calculate the reward based on the distance
        reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1, 0)
        return reward

    return rotation_reward_function
