"""Benchmark functions for the IC-LAND environment."""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import psutil
import pynvml

import icland

SEED = 42

# # # Enable JAX debug flags
# # # jax.config.update("jax_debug_nans", True)  # Check for NaNs
# jax.config.update("jax_log_compiles", True)  # Log compilations
# # # jax.config.update("jax_debug_infs", True)  # Check for infinities

@dataclass
class BenchmarkMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    batched_steps_per_second: float
    max_memory_usage_mb: float
    max_cpu_usage_percent: float
    max_gpu_usage_percent: list[float]
    max_gpu_memory_usage_mb: list[float]


def benchmark_batch_size(batch_size: int) -> BenchmarkMetrics:
    """Benchmark the performance of our step with varying batch sizes."""
    NUM_STEPS = 100

    key = jax.random.PRNGKey(SEED)
    icland_params = icland.sample(key)
    init_state = icland.init(key, icland_params)

    # Batched step function
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, icland_params, 0))

    # Prepare batch
    def replicate(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    icland_states = jax.tree_map(replicate, init_state)
    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    # Old code
    # icland_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), init_state)
    # actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])

    keys = jax.random.split(key, batch_size)

    process = psutil.Process()
    max_memory_usage_mb = 0.0
    max_cpu_usage_percent = 0.0

    # Attempt to initialize NVML for GPU usage
    gpu_available = True
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        max_gpu_usage_percent: list[float] = [0.0] * num_gpus
        max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
    except pynvml.NVMLError:
        gpu_available = False
        max_gpu_usage_percent = []
        max_gpu_memory_usage_mb = []

    # Timed run
    total_time = 0
    for i in range(NUM_STEPS):
        # The elements in each of the four arrays are the same, except for those in keys
        
        print(f'Start of batched step {i}')
        step_start_time = time.time()
        icland_states = batched_step(keys, icland_states, icland_params, actions)
        step_time = time.time() - step_start_time
        total_time += step_time

        print(f'End of batched step {i}. Time taken: {step_time}')
        
        # CPU Memory & Usage
        memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        max_memory_usage_mb = max(max_memory_usage_mb, memory_usage_mb)
        max_cpu_usage_percent = max(max_cpu_usage_percent, cpu_usage_percent)

        # GPU Usage & Memory
        if gpu_available:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util_percent = util_rates.gpu
                gpu_mem_usage_mb = mem_info.used / (1024**2)
                max_gpu_usage_percent[i] = max(
                    max_gpu_usage_percent[i], gpu_util_percent
                )
                max_gpu_memory_usage_mb[i] = max(
                    max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
                )

    if gpu_available:
        pynvml.nvmlShutdown()

    batched_steps_per_second = NUM_STEPS / total_time

    return BenchmarkMetrics(
        batch_size=batch_size,
        batched_steps_per_second=batched_steps_per_second,
        max_memory_usage_mb=max_memory_usage_mb,
        max_cpu_usage_percent=max_cpu_usage_percent,
        max_gpu_usage_percent=max_gpu_usage_percent,
        max_gpu_memory_usage_mb=max_gpu_memory_usage_mb,
    )
