"""Tests for the game_generator module."""

import jax
import jax.numpy as jnp

from icland.game import generate_game
from icland.types import Agent, ICLandInfo


# Define a minimal dummy ICLandInfo type with the attributes used by the reward functions.
class DummyICLandInfo:
    """Minimal dummy ICLandInfo class for testing."""

    def __init__(self, agent_positions: jax.Array, agent_rotations: jax.Array) -> None:
        """Initialize the dummy ICLandInfo object."""
        self.agent_positions = agent_positions
        self.agent_rotations = agent_rotations


def test_generate_game_runs_without_error() -> None:
    """Test that the generate_game function runs without error."""
    # Create a random key.
    key = jax.random.PRNGKey(0)
    agent_count = 3

    for i in range(5):
        key, subkey = jax.random.split(key)

        # Generate the game reward function.
        reward_fn = generate_game(subkey, agent_count)
        # Ensure that a callable reward function is returned.
        assert callable(reward_fn)

        # Create a dummy ICLandInfo input.
        # Both fields are provided even though only one is used depending on mode.
        info = ICLandInfo(
            agents=[
                Agent(
                    position=jnp.zeros(3), rotation=jnp.zeros(1), velocity=jnp.zeros(4)
                )
                for _ in range(agent_count)
            ]
        )

        # Call the reward function. We don't check the output here,
        # only that the function runs without error.
        rewards = reward_fn(info)

        # Optionally, verify that the output is a JAX array with the expected shape.
        assert isinstance(rewards, jax.Array) or isinstance(rewards, jnp.ndarray)
        assert rewards.shape[0] == agent_count
