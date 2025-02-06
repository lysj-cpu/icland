"""Tests movement behaviour under different pre-defined movement policies in ICLand Brax environment."""

# TODO: Reduce code duplication between test_brax.py and test_agent.py

import jax
import jax.numpy as jnp
import mujoco
import pytest
from assets.policies import *
from assets.worlds import EMPTY_WORLD, TWO_AGENT_EMPTY_WORLD
from brax.envs import get_environment, register_environment
from brax.mjx.base import State as mjx_state

from icland.brax_env import ICLand
from icland.constants import SMALL_VALUE
from icland.types import ICLandBraxState, ICLandParams


@pytest.fixture(params=[EMPTY_WORLD, TWO_AGENT_EMPTY_WORLD])
def env(
    request: pytest.FixtureRequest,
) -> tuple[jax._src.pjit.JitWrapped, ICLandBraxState, mjx_state]:
    """Fixture to provide a consistent ICLand Brax environment for tests."""
    world = request.param
    register_environment("icland", ICLand)
    mj_model = mujoco.MjModel.from_xml_string(world)
    agent_count = 1 if world == EMPTY_WORLD else 2
    icland_params = ICLandParams(mj_model, None, agent_count, None)
    env = get_environment("icland", rng=jax.random.key(42), params=icland_params)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state: ICLandBraxState = jit_reset(jax.random.PRNGKey(0))
    pipeline_state = state.pipeline_state

    return jit_step, state, pipeline_state


@pytest.mark.parametrize(
    "name, policy, expected_direction, env",
    [
        ("Forward Movement", FORWARD_POLICY, jnp.array([1, 0]), EMPTY_WORLD),
        ("Backward Movement", BACKWARD_POLICY, jnp.array([-1, 0]), EMPTY_WORLD),
        ("Left Movement", LEFT_POLICY, jnp.array([0, 1]), EMPTY_WORLD),
        ("Right Movement", RIGHT_POLICY, jnp.array([0, -1]), EMPTY_WORLD),
        ("No Movement", NOOP_POLICY, jnp.array([0, 0]), EMPTY_WORLD),
    ],
    indirect=["env"],
)
def test_agent_translation(
    env: tuple[jax._src.pjit.JitWrapped, ICLandBraxState, mjx_state],
    name: str,
    policy: jnp.ndarray,
    expected_direction: jnp.ndarray,
) -> None:
    """Test agent movement in ICLand Brax environment."""
    jit_step, state, pipeline_state = env
    body_id = state.component_ids[0, 0]

    # Get initial position, without height
    initial_pos = pipeline_state.xpos[body_id][:2]

    # Step the environment to update the agents velocty
    state = jit_step(state, policy)
    pipeline_state = state.pipeline_state

    # Check if the correct velocity was applied
    velocity = pipeline_state.qvel[:2]
    normalised_velocity = velocity / (jnp.linalg.norm(velocity) + SMALL_VALUE)

    assert jnp.allclose(normalised_velocity, expected_direction), (
        f"{name} failed: Expected velocity {expected_direction}, "
        f"Actual velocity {normalised_velocity}"
    )

    # Step the environment to update the agents position via the velocity
    state = jit_step(state, NOOP_POLICY)
    pipeline_state = state.pipeline_state

    # Get new position
    new_position = pipeline_state.xpos[body_id][:2]

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
    "name, policy, expected_orientation, env",
    [
        ("Clockwise Rotation", CLOCKWISE_POLICY, -1, EMPTY_WORLD),
        ("Anti-clockwise Rotation", ANTI_CLOCKWISE_POLICY, 1, EMPTY_WORLD),
        ("No Rotation", NOOP_POLICY, 0, EMPTY_WORLD),
    ],
    indirect=["env"],
)
def test_agent_rotation(
    env: tuple[jax._src.pjit.JitWrapped, ICLandBraxState, mjx_state],
    policy: jnp.ndarray,
    expected_orientation: jnp.ndarray,
    name: str,
) -> None:
    """Test agent rotation in ICLand Brax environment."""
    jit_step, state, pipeline_state = env
    initial_orientation = pipeline_state.qpos[3]
    state = jit_step(state, policy)
    pipeline_state = state.pipeline_state

    new_orientation = pipeline_state.qpos[3]
    orientation_delta = new_orientation - initial_orientation
    normalised_orientation_delta = orientation_delta / (
        jnp.linalg.norm(orientation_delta) + SMALL_VALUE
    )
    assert jnp.allclose(normalised_orientation_delta, expected_orientation), (
        f"{name} failed: Expected orientation -1, "
        f"Actual orientation {normalised_orientation_delta}"
    )


@pytest.mark.parametrize(
    "name, policies, env",
    [
        (
            "Move In Parallel",
            jnp.array([FORWARD_POLICY, FORWARD_POLICY]),
            TWO_AGENT_EMPTY_WORLD,
        ),
        (
            "Two Agents Colide",
            jnp.array([FORWARD_POLICY, BACKWARD_POLICY]),
            TWO_AGENT_EMPTY_WORLD,
        ),
    ],
    indirect=["env"],
)
def test_two_agents(
    env: tuple[jax._src.pjit.JitWrapped, ICLandBraxState, mjx_state],
    name: str,
    policies: jnp.ndarray,
) -> None:
    """Test two agents movement in ICLand Brax environment."""
    jit_step, state, pipeline_state = env

    # Simulate 2 seconds
    while pipeline_state.time < 2:
        state = jit_step(state, policies)
        pipeline_state = state.pipeline_state

    # Get the positions of the two agents
    body_id_1, body_id_2 = state.component_ids[:, 0]
    agent_1_pos = pipeline_state.xpos[body_id_1][:2]
    agent_2_pos = pipeline_state.xpos[body_id_2][:2]

    # Simulate one more step.
    state = jit_step(state, NOOP_POLICY)
    pipeline_state = state.pipeline_state

    agent_1_new_pos = pipeline_state.xpos[body_id_1][:2]
    agent_2_new_pos = pipeline_state.xpos[body_id_2][:2]

    # Get the displacements
    displacement_1 = agent_1_new_pos - agent_1_pos
    displacement_2 = agent_2_new_pos - agent_2_pos

    if name == "Move In Parallel":
        # Check the two agents moved in parallel
        assert jnp.allclose(displacement_1 - displacement_2, 0), (
            f"{name} failed: Expected displacement difference 0, "
            f"Agent 1 displacement {displacement_1}, Agent 2 displacement {displacement_2}"
        )
    elif name == "Two Agents Colide":
        # Check agents do not move (they have collided)
        assert jnp.allclose(displacement_1 + displacement_2, 0), (
            f"{name} failed: Expected displacement difference 0, "
            f"Agent 1 displacement {displacement_1}, Agent 2 displacement {displacement_2}"
        )

    else:
        raise ValueError("Invalid test case name")
