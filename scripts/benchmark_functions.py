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
from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame
from icland.types import ICLandParams
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.tile_data import TILECODES
from video_generator import generate_mjcf_string

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

def benchmark_step_non_empty_world(batch_size: int) -> BenchmarkMetrics:
    NUM_STEPS = 100
    height = 10
    width = 10
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, batch_size)

    print(f"Before sample_world...")
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))

    # model = sample_world(height, width, 1000, key, True, 1)
    models = batched_sample_world(height, width, 1000, keys, True, 1)

    print(f"Before export...")
    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, height, width)

    print(f"Before sample spawn points...")
    batched_sample_spawn_points = jax.vmap(sample_spawn_points, in_axes=(0, 0))
    spawnpoints = batched_sample_spawn_points(keys, tilemaps)

    # create_world is not fully jitted
    print(f"Before create world...")

    batched_create_world = jax.vmap(create_world, in_axes=(0))
    pieces = batched_create_world(tilemaps)

    temp_dir = "temp"
    os.makedirs(f"{temp_dir}", exist_ok=True)

    print(f"Before export stls...")
    def batch_export_stls(pieces, temp_dir):
        def body_fn(carry, piece):
            export_stls(piece, f"{temp_dir}/{temp_dir}")
            return carry, None  # We don't need to return anything from each step

        # Use scan to apply 'export_stls' over the 'pieces'
        _, _ = jax.lax.scan(body_fn, None, pieces)

    batch_export_stls(pieces, f"{temp_dir}/{temp_dir}")

    print(f"Before generate mjcf string...")
    def batch_generate_mjcf_string(tilemaps, spawnpoints, temp_dir):
        def body_fn(carry, data):
            tilemap, spawnpoint = data
            xml_str = generate_mjcf_string(tilemap, spawnpoint, f"{temp_dir}/")
            return carry, xml_str

        # Combine tilemaps and spawnpoints into a list of tuples for scanning
        combined_data = list(zip(tilemaps, spawnpoints))
        
        # Use scan to apply 'generate_mjcf_string' over the combined data
        _, xml_strs = jax.lax.scan(body_fn, None, combined_data)
        
        return xml_strs

    xml_strs = batch_generate_mjcf_string(tilemaps, spawnpoints, f"{temp_dir}/")

    print(f"Before batched from xml string...")
    batched_from_xml_string = jax.vmap(mujoco.MjModel.from_xml_string, in_axes=(0))
    mj_models = batched_from_xml_string(xml_strs)

    def create_icland_params(mj_model):
        return ICLandParams(model=mj_model, game=None, agent_count=1)

    batched_create_icland_params = jax.vmap(create_icland_params, in_axes=(0))
    icland_params = batched_create_icland_params(mj_models)

    batched_init = jax.vmap(icland.init, in_axes=(0, 0))
    icland_states = batched_init(keys, icland_params)

    def create_mjx_data(state):
        return state.pipeline_state.mjx_data
    
    batched_create_mjx_data = jax.vmap(create_mjx_data, in_axes=(0))
    mjx_data = batched_create_mjx_data(icland_states)

    frames: list[Any] = []

    print(f"Starting simulation...")
    last_printed_time = -0.1

    default_agent_1 = 0
    world_width = width
    print(f"Rendering...")
    batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None, None, None))

    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    # Old code
    # icland_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), init_state)
    # actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])

    batched_render_frame = jax.vmap(render_frame, in_axes=(0, 0, 0))

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
        print(f"Start of batched step {i}")
        step_start_time = time.time()
        icland_states = batched_step(keys, icland_states, icland_params, actions)
        step_time = time.time() - step_start_time
        print(f"End of batched step {i}. Time taken: {step_time}")
        total_time += step_time

        # mjx_data is not really used
        # mjx_data = batched_create_mjx_data(icland_states)
        # if len(frames) < mjx_data.time * 30:
        #     print(f"Start of batched render {i}")
        #     render_start_time = time.time()
        #     camera_poses, camera_dirs = batched_get_camera_info(
        #         icland_states, world_width, default_agent_1
        #     )
        #     f = batched_render_frame(
        #         camera_poses, camera_dirs, tilemaps, view_width=96, view_height=72
        #     )
        #     render_time = time.time() - render_start_time
        #     print(f"End of batched render {i}. Time taken: {render_time}")
        #     total_time += render_time
        #     frames.append(f)

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
