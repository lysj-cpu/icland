"""A temporary script showing how users would interact with the ICLand environment."""

import functools
from datetime import datetime
from typing import Any, Dict

import jax
import matplotlib.pyplot as plt
from brax.envs import get_environment, register_environment
from brax.training.agents.ppo import train as ppo

from icland.brax_env import ICLand

# Training parameters
NUM_TIMESTEPS = 100_000_000
NUM_EVALS = 10
REWARD_SCALING = 1
EPISODE_LENGTH = 1000
NORMALIZE_OBSERVATIONS = False
ACTION_REPEAT = 1
UNROLL_LENGTH = 20
NUM_MINIBATCHES = 32
NUM_UPDATES_PER_BATCH = 8
DISCOUNTING = 0.97
LEARNING_RATE = 3e-4
ENTROPY_COST = 1e-4
NUM_ENVS = 8192
BATCH_SIZE = 256
SEED = 42


def run_brax_test() -> None:
    """Run a simple test of the ICLand environment using Brax."""
    # Register the custom environment and create an instance.
    register_environment("icland", ICLand)
    env = get_environment("icland", rng=jax.random.PRNGKey(SEED))

    # Create a partial training function with fixed parameters.
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=NUM_TIMESTEPS,
        num_evals=NUM_EVALS,
        reward_scaling=REWARD_SCALING,
        episode_length=EPISODE_LENGTH,
        normalize_observations=NORMALIZE_OBSERVATIONS,
        action_repeat=ACTION_REPEAT,
        unroll_length=UNROLL_LENGTH,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=NUM_UPDATES_PER_BATCH,
        discounting=DISCOUNTING,
        learning_rate=LEARNING_RATE,
        entropy_cost=ENTROPY_COST,
        num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # Containers for tracking training progress.
    progress_data: Dict[str, Any] = {
        "steps": [],
        "rewards": [],
        "reward_std": [],
        "timestamps": [datetime.now()],
    }

    # Set y-axis bounds for plotting.
    min_reward, max_reward = -10, 10

    def progress_callback(num_steps: int, metrics: Dict[str, Any]) -> None:
        """Callback function to track and plot training progress."""
        progress_data["timestamps"].append(datetime.now())
        progress_data["steps"].append(num_steps)
        progress_data["rewards"].append(metrics["eval/episode_reward"])
        progress_data["reward_std"].append(metrics["eval/episode_reward_std"])

        # Create a new figure for each progress update.
        plt.figure()
        plt.xlim(0, NUM_TIMESTEPS * 1.25)
        plt.ylim(min_reward, max_reward)
        plt.xlabel("# environment steps")
        plt.ylabel("Reward per episode")
        plt.title(f"Reward = {progress_data['rewards'][-1]:.3f}")

        # Plot the rewards with error bars.
        plt.errorbar(
            progress_data["steps"],
            progress_data["rewards"],
            yerr=progress_data["reward_std"],
            fmt="-o",
        )
        plt.savefig(f"training_progress_{num_steps}.png")
        plt.close()  # Close the figure to free up memory.

    # Run training.
    inference_fn, params, _ = train_fn(environment=env, progress_fn=progress_callback)

    # Print timing information.
    jit_time = progress_data["timestamps"][1] - progress_data["timestamps"][0]
    training_time = progress_data["timestamps"][-1] - progress_data["timestamps"][1]
    print(f"Time to jit: {jit_time}")
    print(f"Time to train: {training_time}")


if __name__ == "__main__":
    run_brax_test()
