"""Tests movement behaviour under different pre-defined movement policies."""

import jax
import jax.numpy as jnp
import pytest

import icland
from icland.constants import SMALL_VALUE
from icland.presets import *
from icland.types import *
from icland.world_gen.model_editing import generate_base_model


@pytest.fixture
def key() -> jax.Array:
    """Fixture to provide a consistent PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.mark.parametrize(
    "name, policy, expected_direction",
    [
        ("Forward Movement", FORWARD_POLICY, jnp.array([1, 0])),
        ("Backward Movement", BACKWARD_POLICY, jnp.array([-1, 0])),
        ("Left Movement", LEFT_POLICY, jnp.array([0, 1])),
        ("Right Movement", RIGHT_POLICY, jnp.array([0, -1])),
        ("No Movement", NOOP_POLICY, jnp.array([0, 0])),
    ],
)
def test_agent_translation(
    key: jax.Array,
    name: str,
    policy: jnp.ndarray,
    expected_direction: jax.Array,
) -> None:
    """Test agent movement in ICLand environment."""
    # Create the ICLand environment
    mjx_model, _ = generate_base_model(DEFAULT_CONFIG)
    icland_params = ICLandParams(
        world=TEST_TILEMAP_EMPTY_WORLD,
        reward_function=None,
        agent_spawns=jnp.array([[1, 1, 1]]),
        world_level=6,
    )

    icland_state = icland.init(jax.random.PRNGKey(42), icland_params, mjx_model)
    pipeline_state = icland_state.pipeline_state
    body_id = pipeline_state.component_ids[0, 0].astype(int)

    # Initial step (to apply data from model)
    icland_state = icland.step(key, icland_state, None, policy)

    # Get initial position, without height
    pipeline_state = icland_state.pipeline_state
    initial_pos = pipeline_state.mjx_data.xpos[body_id][:2]

    # Step the environment to update the agents velocty
    icland_state = icland.step(key, icland_state, None, policy)

    # Check if the correct velocity was applied
    velocity = pipeline_state.mjx_data.qvel[:2]
    normalised_velocity = velocity / (jnp.linalg.norm(velocity) + SMALL_VALUE)

    assert jnp.allclose(normalised_velocity, expected_direction), (
        f"{name} failed: Expected velocity {expected_direction}, "
        f"Actual velocity {normalised_velocity}"
    )

    # Step the environment to update the agents position via the velocity
    icland_state = icland.step(key, icland_state, None, NOOP_POLICY)
    pipeline_state = icland_state.pipeline_state

    # Get new position
    new_position = pipeline_state.mjx_data.xpos[body_id][:2]

    # Check if the agent moved in the expected direction
    displacement = new_position - initial_pos
    normalised_displacement = displacement / (
        jnp.linalg.norm(displacement) + SMALL_VALUE
    )
    assert jnp.allclose(normalised_displacement, expected_direction), (
        f"{name} failed: Expected displacement {expected_direction}, "
        f"Actual displacement {normalised_displacement}"
    )


@pytest.mark.parametrize(
    "name, policy, expected_orientation",
    [
        ("Clockwise Rotation", CLOCKWISE_POLICY, -1),
        ("Anti-clockwise Rotation", ANTI_CLOCKWISE_POLICY, 1),
        ("No Rotation", NOOP_POLICY, 0),
    ],
)
def test_agent_rotation(
    key: jax.Array,
    name: str,
    policy: jnp.ndarray,
    expected_orientation: jnp.ndarray,
) -> None:
    """Test agent movement in ICLand environment."""
    # Create the ICLand environment
    mjx_model, _ = generate_base_model(DEFAULT_CONFIG)
    icland_params = ICLandParams(
        world=TEST_TILEMAP_EMPTY_WORLD,
        reward_function=None,
        agent_spawns=jnp.array([[1, 1, 1]]),
        world_level=6,
    )

    icland_state = icland.init(jax.random.PRNGKey(42), icland_params, mjx_model)
    pipeline_state = icland_state.pipeline_state

    # Get initial orientation
    initial_orientation = pipeline_state.mjx_data.qpos[3]

    # Step the environment to update the agents angular velocity
    icland_state = icland.step(key, icland_state, None, policy)
    pipeline_state = icland_state.pipeline_state

    # Get new orientation
    new_orientation = pipeline_state.mjx_data.qpos[3]
    orientation_delta = new_orientation - initial_orientation
    normalised_orientation_delta = orientation_delta / (
        jnp.linalg.norm(orientation_delta) + SMALL_VALUE
    )
    assert jnp.allclose(normalised_orientation_delta, expected_orientation), (
        f"{name} failed: Expected orientation {expected_orientation}, "
        f"Actual orientation {normalised_orientation_delta}"
    )


@pytest.mark.parametrize(
    "name, policies",
    [
        ("Move In Parallel", jnp.array([FORWARD_POLICY, FORWARD_POLICY])),
        ("Two Agents Colide", jnp.array([FORWARD_POLICY, BACKWARD_POLICY])),
    ],
)
def test_two_agents(key: jax.Array, name: str, policies: jnp.ndarray) -> None:
    """Test two agents movement in ICLand environment."""
    # Create the ICLand environment
    config = DEFAULT_CONFIG.replace(max_agent_count=2)
    mjx_model, _ = generate_base_model(config)
    icland_params = ICLandParams(
        world=TEST_TILEMAP_EMPTY_WORLD,
        reward_function=None,
        agent_spawns=jnp.array([[1, 1, 1], [1, 1.5, 1]]),
        world_level=6,
    )

    icland_state = icland.init(jax.random.PRNGKey(42), icland_params, mjx_model)
    icland_state = icland.step(key, icland_state, None, policies)
    pipeline_state = icland_state.pipeline_state

    # Simulate 2 seconds
    while pipeline_state.mjx_data.time < 2:
        icland_state = icland.step(key, icland_state, None, policies)
        pipeline_state = icland_state.pipeline_state

    # Get the positions of the two agents
    body_id_1, body_id_2 = pipeline_state.component_ids[:, 0].astype(int)
    agent_1_pos = pipeline_state.mjx_data.xpos[body_id_1][:2]
    agent_2_pos = pipeline_state.mjx_data.xpos[body_id_2][:2]

    # Simulate one more second.
    while pipeline_state.mjx_data.time < 1:
        icland_state = icland.step(key, icland_state, None, NOOP_POLICY)
        pipeline_state = icland_state.pipeline_state

    agent_1_new_pos = pipeline_state.mjx_data.xpos[body_id_1][:2]
    agent_2_new_pos = pipeline_state.mjx_data.xpos[body_id_2][:2]

    # Get the displacements
    displacement_1 = agent_1_new_pos - agent_1_pos
    displacement_2 = agent_2_new_pos - agent_2_pos

    if name == "Move In Parallel":
        # Check the two agents moved in parallel
        assert jnp.allclose(displacement_1 - displacement_2, 0, atol=1e-2), (
            f"{name} failed: Expected displacement difference 0, "
            f"Agent 1 displacement {displacement_1}, Agent 2 displacement {displacement_2}"
        )
    elif name == "Two Agents Colide":
        # Check agents do not move (they have collided)
        assert jnp.allclose(displacement_1 + displacement_2, 0, atol=1e-2), (
            f"{name} failed: Expected displacement difference 0, "
            f"Agent 1 displacement {displacement_1}, Agent 2 displacement {displacement_2}"
        )

    else:
        raise ValueError("Invalid test case name")
