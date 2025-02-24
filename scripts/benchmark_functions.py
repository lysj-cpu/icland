"""Benchmark functions for the IC-LAND environment."""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import psutil
import pynvml

import icland
from icland.types import *
from icland.world_gen.model_editing import generate_base_model

SEED = 42


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
    config = ICLandConfig(2, 2, 1, {}, 6)
    mjx_model, _ = generate_base_model(config)
    icland_params = icland.sample(key, config)
    init_state = icland.init(key, icland_params, mjx_model)

    # Batched step function
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, icland_params, 0))

    # Prepare batch
    icland_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), init_state)
    actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])
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
    start_time = time.time()
    for _ in range(NUM_STEPS):
        icland_states = batched_step(keys, icland_states, icland_params, actions)

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

    total_time = time.time() - start_time

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
