"""A temporary script showing how users would interact with the ICLand environment."""

import jax
from brax.envs import get_environment, register_environment

from icland.brax_env import ICLand

if __name__ == "__main__":
    # Adapted from https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
    register_environment("icland", ICLand)
    env = get_environment("icland", rng=jax.random.key(42))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    # TODO: Keep adapting from aforementioned source
