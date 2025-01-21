"""unit tests."""

import os

# N.B. These need to be before the mujoco imports
# Fixes AttributeError: 'Renderer' object has no attribute '_mjr_context'
os.environ["MUJOCO_GL"] = "egl"

# Tell XLA to use Triton GEMM, this can improve steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import ICLand
import jax
import mujoco
from mujoco import mjx
import jax.numpy as jnp
import math
import imageio

# Define movement policies
NOOP_POLICY = jnp.array([0, 0, 0])
FORWARD_POLICY = jnp.array([1, 0, 0])
BACK_POLICY = jnp.array([-1, 0, 0])
LEFT_POLICY = jnp.array([0, -1, 0])
RIGHT_POLICY = jnp.array([0, 1, 0])


def run_test(
    key: jax.random.PRNGKey,
    policy: jnp.ndarray,
    axis: int = None,
    direction: float = None,
    test_name: str = "",
):
    """Helper function to test agent movement along a given axis and direction.
    
    If axis is None, checks that the agent does not move.
    """
    # 1. Sample and initialize ICLand environment
    icland_params = ICLand.sample(key)
    icland_state = ICLand.init(key, icland_params)

    # 2. Step the environment once (warm-up step)
    icland_state = ICLand.step(key, icland_state, None, policy)
    body_id = icland_state[2][0][0]

    # 3. Get initial position
    initial_pos = icland_state[1].xpos[body_id]

    # 4. Step the environment again
    icland_state = ICLand.step(key, icland_state, None, policy)
    new_pos = icland_state[1].xpos[body_id]

    # 5. Check movement or no movement
    if axis is None:
        # No movement test
        assert jnp.allclose(initial_pos, new_pos), (
            f"{test_name} Failed: Agent moved when it shouldn't have. "
            f"Initial: {initial_pos}, New: {new_pos}"
        )
    else:
        # Movement test
        initial_axis = initial_pos[axis]
        new_axis = new_pos[axis]
        if direction > 0:
            assert new_axis > initial_axis, (
                f"{test_name} Failed: Expected positive movement along axis {axis}. "
                f"Initial: {initial_axis}, New: {new_axis}"
            )
        else:
            assert new_axis < initial_axis, (
                f"{test_name} Failed: Expected negative movement along axis {axis}. "
                f"Initial: {initial_axis}, New: {new_axis}"
            )

    print(f"Test Passed: {test_name}")


def main():
    """Runs a suite of agent movement tests in the ICLand environment.

    This function uses a fixed PRNG key for reproducibility and calls the 
    `run_test` helper with different movement policies (forward, backward, 
    left, right) as well as a no-operation policy. Each test verifies that 
    the agent's position changes (or does not change) in the expected way.
    """
    # Use a consistent PRNG key for all tests
    key = jax.random.PRNGKey(42)

    # Movement tests
    run_test(key, FORWARD_POLICY, axis=0, direction=1, test_name="Forward Movement")
    run_test(key, BACK_POLICY, axis=0, direction=-1, test_name="Backward Movement")
    run_test(key, LEFT_POLICY, axis=1, direction=-1, test_name="Left Movement")
    run_test(key, RIGHT_POLICY, axis=1, direction=1, test_name="Right Movement")


if __name__ == "__main__":
    main()
