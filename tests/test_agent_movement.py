"""Tests movement behaviour under different pre-defined movement policies."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest

import icland
from icland.types import ICLandState

# Define movement policies
NOOP_POLICY = jnp.array([0, 0, 0])
FORWARD_POLICY = jnp.array([1, 0, 0])
BACK_POLICY = jnp.array([-1, 0, 0])
LEFT_POLICY = jnp.array([0, -1, 0])
RIGHT_POLICY = jnp.array([0, 1, 0])


@pytest.fixture
def key() -> jax.Array:
    """Fixture to provide a consistent PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def initialize_icland(key: jax.Array) -> Callable[[jnp.ndarray], ICLandState]:
    """Fixture to initialize the ICLand environment."""

    def _init(policy: jnp.ndarray) -> ICLandState:
        icland_params = icland.sample(key)
        icland_state = icland.init(key, icland_params)
        # Perform a warm-up step
        icland_state = icland.step(key, icland_state, None, policy)
        return icland_state

    return _init


@pytest.mark.parametrize(
    "policy, axis, direction, description",
    [
        (FORWARD_POLICY, 0, 1, "Forward Movement"),
        (BACK_POLICY, 0, -1, "Backward Movement"),
        (LEFT_POLICY, 1, -1, "Left Movement"),
        (RIGHT_POLICY, 1, 1, "Right Movement"),
        # (NOOP_POLICY, None, None, "No Movement"),
        # TODO: Noop policy
    ],
)
def test_agent_movement(
    key: jax.Array,
    initialize_icland: Callable[[jnp.ndarray], ICLandState],
    policy: jnp.ndarray,
    axis: int,
    direction: float,
    description: str,
) -> None:
    """Test agent movement in ICLand environment."""
    icland_state = initialize_icland(policy)
    body_id = icland_state.agent_data.body_id[0]

    # Get initial position
    initial_pos = icland_state.mjx_data.xpos[body_id]

    # Step the environment and get new position
    icland_state = icland.step(key, icland_state, None, policy)
    new_pos = icland_state.mjx_data.xpos[body_id]

    if axis is None:
        # No movement expected
        assert jnp.allclose(initial_pos, new_pos), (
            f"{description} Failed: Agent moved when it shouldn't have. "
            f"Initial: {initial_pos}, New: {new_pos}"
        )
    else:
        # Movement expected
        initial_axis = initial_pos[axis]
        new_axis = new_pos[axis]
        if direction > 0:
            assert new_axis > initial_axis, (
                f"{description} Failed: Expected positive movement along axis {axis}. "
                f"Initial: {initial_axis}, New: {new_axis}"
            )
        else:
            assert new_axis < initial_axis, (
                f"{description} Failed: Expected negative movement along axis {axis}. "
                f"Initial: {initial_axis}, New: {new_axis}"
            )
