"""Benchmark functions for the IC-LAND environment."""

import os
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import psutil
import pynvml

import icland
from icland.agent import collect_body_scene_info, create_agent
from icland.constants import AGENT_OBSERVATION_DIM
from icland.game import generate_game
from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame
from icland.types import ICLandParams, ICLandState, PipelineState
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.model_editing import edit_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES
from video_generator import generate_mjcf_string

SEED = 42

# # Enable JAX debug flags
# # jax.config.update("jax_debug_nans", True)  # Check for NaNs
# jax.config.update("jax_log_compiles", True)  # Log compilations
# # jax.config.update("jax_debug_infs", True)  # Check for infinities

@dataclass
class BenchmarkMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    batched_steps_per_second: float
    max_memory_usage_mb: float
    max_cpu_usage_percent: float
    max_gpu_usage_percent: list[float]
    max_gpu_memory_usage_mb: list[float]

@dataclass
class SampleWorldBenchmarkMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    sample_world_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: list[float]
    gpu_memory_usage_mb: list[float]

def benchmark_renderer_non_empty_world(batch_size: int) -> BenchmarkMetrics:
    NUM_STEPS = 20
    height = 1
    width = 1
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    agent_count = 1
    print(f"Benchmarking non-empty world renderer of size {height}x{width}, with agent count of {agent_count}")

    # Maybe switch to use np ops instead of list comprehension
    print(f"Before sample_world...")
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
    models = batched_sample_world(height, width, 1000, keys, True, 1) 

    print(f"Before export...")
    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, height, width)

    print(f"Before generate_base_model...")
    mjx_model, mj_model = generate_base_model(height, width, agent_count)
    agent_components = icland.collect_agent_components(mj_model, agent_count)
    
    batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(
        tilemaps,
        mjx_model,
    )

    # Batched mjx data
    batch_data = jax.vmap(mujoco.mjx.make_data)(batch)
    
    # icland_params = ICLandParams(
    #     model=mj_model, 
    #     reward_function=None, 
    #     agent_count=agent_count
    # )

    # actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    def create_icland_state(mjx_model, mjx_data):
        return ICLandState(
            PipelineState(mjx_model, mjx_data, agent_components),
            jnp.zeros(AGENT_OBSERVATION_DIM),
            collect_body_scene_info(agent_components, mjx_data)
        )

    # Emulate our own step and run once
    icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)   # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))

    # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))
    batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))
    batched_render_frame = jax.vmap(render_frame, in_axes=(0, 0, 0))
    print(f"Starting simulation...")

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


    default_agent_1 = 0

    camera_poses, camera_dirs = batched_get_camera_info(
        icland_states, width, default_agent_1
    )

    # Timed run
    total_time = 0
    for i in range(NUM_STEPS):
        # The elements in each of the four arrays are the same, except for those in keys
        # icland_states = batched_step(keys, icland_states, icland_params, actions)

        # camera_poses, camera_dirs = batched_get_camera_info(
        #     icland_states, width, default_agent_1
        # )

        # print("Got camera angle")
        step_start_time = time.time()
        f = batched_render_frame(camera_poses, camera_dirs, tilemaps)        
        step_time = time.time() - step_start_time
        total_time += step_time
        print(f'End of batched render frame step {i}. Time taken: {step_time}')

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

def benchmark_sample_world(batch_size: int) -> SampleWorldBenchmarkMetrics:
    height = 2
    width = 2
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    print(f"Benchmarking sample world of size {height}x{width}")

    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))

    process = psutil.Process()
    memory_usage_mb = 0.0
    cpu_usage_percent = 0.0

    # Attempt to initialize NVML for GPU usage
    gpu_available = True
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        gpu_usage_percent: list[float] = [0.0] * num_gpus
        gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
    except pynvml.NVMLError:
        gpu_available = False
        gpu_usage_percent = []
        gpu_memory_usage_mb = []
    
    process.cpu_percent(interval=None)

    start_time = time.time()
    models = batched_sample_world(height, width, 1000, keys, True, 1) 
    total_time = time.time() - start_time
    print(f"sample_world of batch {batch_size} took {total_time}s")

    # CPU Memory & Usage
    memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
    cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
    memory_usage_mb = memory_usage_mb
    cpu_usage_percent = cpu_usage_percent

    # GPU Usage & Memory
    if gpu_available:
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_util_percent = util_rates.gpu
            gpu_mem_usage_mb = mem_info.used / (1024**2)
            gpu_usage_percent[i] = gpu_util_percent
            gpu_memory_usage_mb[i] = gpu_mem_usage_mb
    
    if gpu_available:
        pynvml.nvmlShutdown()

    return SampleWorldBenchmarkMetrics(
        batch_size=batch_size,
        sample_world_time=total_time,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent,
        gpu_usage_percent=gpu_usage_percent,
        gpu_memory_usage_mb=gpu_memory_usage_mb,
    )

def benchmark_step_non_empty_world(batch_size: int) -> BenchmarkMetrics:
    NUM_STEPS = 20
    height = 2
    width = 2
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    agent_count = 4
    print(f"Benchmarking non-empty world of size {height}x{width}, with agent count of {agent_count}")

    # Maybe switch to use np ops instead of list comprehension
    print(f"Before sample_world...")
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
    models = batched_sample_world(height, width, 1000, keys, True, 1) 

    print(f"Before export...")
    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, height, width)

    print(f"Before generate_base_model...")
    mjx_model, mj_model = generate_base_model(height, width, agent_count)
    agent_components = icland.collect_agent_components(mj_model, agent_count)
    
    batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(
        tilemaps,
        mjx_model,
    )

    # Batched mjx data
    batch_data = jax.vmap(mujoco.mjx.make_data)(batch)
    
    icland_params = ICLandParams(
        model=mj_model, 
        reward_function=None, 
        agent_count=agent_count
    )

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    def create_icland_state(mjx_model, mjx_data):
        return ICLandState(
            PipelineState(mjx_model, mjx_data, agent_components),
            jnp.zeros(AGENT_OBSERVATION_DIM),
            collect_body_scene_info(agent_components, mjx_data)
        )

    # Emulate our own step and run once
    icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)   # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))

    batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))
    print(f"Starting simulation...")

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
        icland_states = batched_step(None, icland_states, icland_params, actions)
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



def benchmark_batch_size(batch_size: int) -> BenchmarkMetrics:
    """Benchmark the performance of our step with varying batch sizes."""
    NUM_STEPS = 20

    key = jax.random.PRNGKey(SEED)
    icland_params = icland.sample(key)
    init_state = icland.init(key, icland_params)

    # Prepare batch
    def replicate(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    icland_states = jax.tree_map(replicate, init_state)
    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    # Old code
    # icland_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), init_state)
    # actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])

    keys = jax.random.split(key, batch_size)

    # Batched step function
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))

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
