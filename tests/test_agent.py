"""Tests movement behaviour under different pre-defined movement policies."""

import jax
import jax.numpy as jnp
import pytest

import icland
from icland.constants import *
from icland.presets import *
from icland.types import *
from icland.world_gen.model_editing import edit_model_data, generate_base_model


@pytest.fixture
def key() -> jax.Array:
    """Fixture to provide a consistent PRNG key for tests."""
    return jax.random.PRNGKey(42)


def world(num_agents: int) -> ICLandParams:
    """Helper function to provide a consistent world for tests."""
    WORLD_WIDTH = 10
    WORLD_DEPTH = 10
    WORLD_HEIGHT = 6
    MAX_AGENT_COUNT = num_agents
    MAX_SPHERE_COUNT = 0
    MAX_CUBE_COUNT = 0
    mjx_model, mj_model = generate_base_model(
        WORLD_WIDTH,
        WORLD_DEPTH,
        WORLD_HEIGHT,
        MAX_AGENT_COUNT,
        MAX_SPHERE_COUNT,
        MAX_CUBE_COUNT,
    )
    agent_info = ICLandAgentInfo(
        agent_count=num_agents,
        spawn_points=jnp.array([[5, 2 + 1 * i, 3.5] for i in range(num_agents)]),
        spawn_orientations=jnp.zeros((num_agents,), dtype="float32"),
        body_ids=jnp.arange(1, num_agents + 1, dtype="int32"),
        geom_ids=(jnp.arange(num_agents) + WORLD_WIDTH * WORLD_DEPTH) * 2 + WALL_OFFSET,
        dof_addresses=jnp.arange(num_agents) * AGENT_DOF_OFFSET,
        colour=jnp.zeros((num_agents, 3)),
    )
    prop_info = ICLandPropInfo(
        prop_count=1,
        prop_types=jnp.array([0]),
        spawn_points=-jnp.ones((1, 3)),
        spawn_rotations=jnp.zeros((1, 4), dtype="float32"),
        body_ids=BODY_OFFSET + num_agents + jnp.arange(1, dtype="int32"),
        geom_ids=(1 + WORLD_WIDTH * WORLD_DEPTH) * 2
        + jnp.arange(1, dtype="int32")
        + WALL_OFFSET,
        dof_addresses=jnp.arange(1, dtype="int32") * PROP_DOF_MULTIPLIER
        + PROP_DOF_OFFSET,
        colour=jnp.zeros((1, 3)),
    )
    mjx_model = edit_model_data(
        TEST_TILEMAP_FLAT, mjx_model, agent_info, prop_info, WORLD_HEIGHT
    )
    icland_params = ICLandParams(
        world=ICLandWorld(
            TEST_TILEMAP_FLAT,
            WORLD_WIDTH,
            WORLD_DEPTH,
            WORLD_HEIGHT,
            jnp.zeros((WORLD_WIDTH, WORLD_DEPTH, 3)),
        ),
        agent_info=agent_info,
        prop_info=prop_info,
        reward_function=None,
        mjx_model=mjx_model,
    )
    return icland_params


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
    icland_params = world(1)

    icland_state = icland.init(icland_params)
    body_id = icland_params.agent_info.body_ids[0]

    # Initial step (to apply data from model)
    icland_state, obs, rew = icland.step(icland_state, icland_params, jnp.array(policy))

    # Get initial position, without height
    initial_pos = icland_state.mjx_data.xpos[body_id][:2]

    # Step the environment to update the agents velocty
    icland_state, obs, rew = icland.step(icland_state, icland_params, jnp.array(policy))

    # Check if the correct velocity was applied
    velocity = icland_state.mjx_data.qvel[:2]

    if jnp.all(policy == NOOP_POLICY):
        assert jnp.allclose(velocity, jnp.array([0, 0]), 0, 1e-2), (
            f"{name} failed: Expected velocity [0, 0], Actual velocity {velocity}"
        )
        return

    normalised_velocity = velocity / (jnp.linalg.norm(velocity) + SMALL_VALUE)

    assert jnp.allclose(normalised_velocity, expected_direction, 0, 1e-2), (
        f"{name} failed: Expected velocity {expected_direction}, "
        f"Actual velocity {normalised_velocity}"
    )

    # Step the environment to update the agents position via the velocity
    icland_state, obs, rew = icland.step(icland_state, icland_params, jnp.array(policy))
    new_position = icland_state.mjx_data.xpos[body_id][:2]  # Get new position

    # Check if the agent moved in the expected direction
    displacement = new_position - initial_pos
    normalised_displacement = displacement / (
        jnp.linalg.norm(displacement) + SMALL_VALUE
    )
    assert jnp.allclose(normalised_displacement, expected_direction, 0, 1e-2), (
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
    icland_params = world(1)
    icland_state = icland.init(icland_params)

    # Get initial orientation
    initial_orientation = icland_state.mjx_data.qpos[3]

    # Step the environment to update the agents angular velocity
    icland_state, obs, rew = icland.step(icland_state, icland_params, jnp.array(policy))

    # Get new orientation
    new_orientation = icland_state.mjx_data.qpos[3]
    orientation_delta = new_orientation - initial_orientation
    normalised_orientation_delta = orientation_delta / (
        jnp.linalg.norm(orientation_delta) + SMALL_VALUE
    )
    assert jnp.allclose(normalised_orientation_delta, expected_orientation, 0, 1e-2), (
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
    icland_params = world(2)
    icland_state = icland.init(icland_params)
    icland_state, obs, rew = icland.step(icland_state, icland_params, policies)

    # Simulate 2 seconds
    while icland_state.mjx_data.time < 0.5:
        icland_state, obs, rew = icland.step(icland_state, icland_params, policies)

    # Get the positions of the two agents
    body_id_1, body_id_2 = icland_params.agent_info.body_ids
    agent_1_pos = icland_state.mjx_data.xpos[body_id_1][:2]
    agent_2_pos = icland_state.mjx_data.xpos[body_id_2][:2]

    # Simulate one more second.
    while icland_state.mjx_data.time < 0.5:
        icland_state, obs, rew = icland.step(
            icland_state, icland_params, jnp.array([NOOP_POLICY, NOOP_POLICY])
        )

    agent_1_new_pos = icland_state.mjx_data.xpos[body_id_1][:2]
    agent_2_new_pos = icland_state.mjx_data.xpos[body_id_2][:2]

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
