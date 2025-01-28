"""Tests movement behaviour under different pre-defined movement policies."""

from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import pytest
from assets.policies import *
from assets.worlds import *

import icland
from icland.types import *


@pytest.fixture
def key() -> jax.Array:
    """Fixture to provide a consistent PRNG key for tests."""
    return jax.random.PRNGKey(42)

@pytest.mark.parametrize(
    "name, policy, expected_direction",
    [
        ("Forward Movement", FORWARD_POLICY, jnp.array([1, 0])),
        ("Backward Movement", BACKWARD_POLICY, jnp.array([-1, 0])),
        ("Left Movement", LEFT_POLICY, jnp.array([0, -1])),
        ("Right Movement", RIGHT_POLICY, jnp.array([0, 1])),
        ("No Movement", NOOP_POLICY, jnp.array([0, 0])),
    ],
)
def test_agent_translation(
    key: jax.Array,
    name: str,
    policy: jnp.ndarray,
    expected_direction: jnp.ndarray,
) -> None:
    """Test agent movement in ICLand environment."""

    mj_model = mujoco.MjModel.from_xml_string(EMPTY_WORLD)

    icland_params = ICLandParams(
        mj_model,
        None,
        1
    )

    icland_state = icland.init(key, icland_params)
    body_id = icland_state.agent_data.body_id[0]

    # Get initial position, without height
    initial_pos = icland_state.mjx_data.xpos[body_id][:2]

    # Step the environment to update the agents velocty
    icland_state = icland.step(key, icland_state, None, policy)

    # Check if the correct velocity was applied
    velocity = icland_state.mjx_data.qvel[:2]
    normalised_velocity = velocity / (jnp.linalg.norm(velocity) + 1e-10)
    assert jnp.allclose(normalised_velocity, expected_direction), (
        f"{name} failed: Expected velocity {expected_direction}, "
        f"Actual velocity {normalised_velocity}"
    )

    # Step the environment to update the agents position via the velocity
    icland_state = icland.step(key, icland_state, None, NOOP_POLICY)

    # Get new position
    new_position = icland_state.mjx_data.xpos[body_id][:2]
    
    # Check if the agent moved in the expected direction
    displacement = new_position - initial_pos
    normalised_displacement = displacement / (jnp.linalg.norm(displacement) + 1e-10)
    assert jnp.allclose(normalised_displacement, expected_direction), (
        f"{name} failed: Expected displacement {expected_direction}, "
        f"Actual displacement {normalised_displacement}"
    )
