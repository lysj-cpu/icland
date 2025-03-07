"""Benchmark functions for the IC-LAND environment."""

import os
import time
from dataclasses import dataclass
from typing import Any

from icland.presets import TEST_TILEMAP_EMPTY
import jax
import jax.numpy as jnp
import mujoco
import psutil
import pynvml

import icland
from icland.types import ICLandParams, ICLandState
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.model_editing import edit_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES

SEED = 42

# # Enable JAX debug flags
# # jax.config.update("jax_debug_nans", True)  # Check for NaNs
# jax.config.update("jax_log_compiles", True)  # Log compilations
# # jax.config.update("jax_debug_infs", True)  # Check for infinities

print("JAX devices:")
print(jax.devices())

@dataclass
class SimpleStepMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    num_steps: int
    total_time: float

@dataclass
class ComplexStepMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    num_steps: int
    total_time: float
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

# def benchmark_renderer_non_empty_world(batch_size: int) -> BenchmarkMetrics:
#     NUM_STEPS = 20
#     height = 1
#     width = 1
#     key = jax.random.key(SEED)
#     keys = jax.random.split(key, batch_size)
#     agent_count = 1
#     print(f"Benchmarking non-empty world renderer of size {height}x{width}, with agent count of {agent_count}")

#     # Maybe switch to use np ops instead of list comprehension
#     print(f"Before sample_world...")
#     batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
#     models = batched_sample_world(height, width, 1000, keys, True, 1) 

#     print(f"Before export...")
#     batched_export = jax.vmap(export, in_axes=(0, None, None, None))
#     tilemaps = batched_export(models, TILECODES, height, width)

#     print(f"Before generate_base_model...")
#     mjx_model, mj_model = generate_base_model(height, width, agent_count)
#     agent_components = icland.collect_agent_components(mj_model, agent_count)
    
#     batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(
#         tilemaps,
#         mjx_model,
#     )

#     # Batched mjx data
#     batch_data = jax.vmap(mujoco.mjx.make_data)(batch)
    
#     # icland_params = ICLandParams(
#     #     model=mj_model, 
#     #     reward_function=None, 
#     #     agent_count=agent_count
#     # )

#     # actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

#     def create_icland_state(mjx_model, mjx_data):
#         return ICLandState(
#             PipelineState(mjx_model, mjx_data, agent_components),
#             jnp.zeros(AGENT_OBSERVATION_DIM),
#             collect_body_scene_info(agent_components, mjx_data)
#         )

#     # Emulate our own step and run once
#     icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)   # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))

#     # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))
#     batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))
#     batched_render_frame = jax.vmap(render_frame, in_axes=(0, 0, 0))
#     print(f"Starting simulation...")

#     process = psutil.Process()
#     max_memory_usage_mb = 0.0
#     max_cpu_usage_percent = 0.0

#     # Attempt to initialize NVML for GPU usage
#     gpu_available = True
#     try:
#         pynvml.nvmlInit()
#         num_gpus = pynvml.nvmlDeviceGetCount()
#         max_gpu_usage_percent: list[float] = [0.0] * num_gpus
#         max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
#     except pynvml.NVMLError:
#         gpu_available = False
#         max_gpu_usage_percent = []
#         max_gpu_memory_usage_mb = []


#     default_agent_1 = 0

#     camera_poses, camera_dirs = batched_get_camera_info(
#         icland_states, width, default_agent_1
#     )

#     # Timed run
#     total_time = 0
#     for i in range(NUM_STEPS):
#         # The elements in each of the four arrays are the same, except for those in keys
#         # icland_states = batched_step(keys, icland_states, icland_params, actions)

#         # camera_poses, camera_dirs = batched_get_camera_info(
#         #     icland_states, width, default_agent_1
#         # )

#         # print("Got camera angle")
#         step_start_time = time.time()
#         f = batched_render_frame(camera_poses, camera_dirs, tilemaps)        
#         step_time = time.time() - step_start_time
#         total_time += step_time
#         print(f'End of batched render frame step {i}. Time taken: {step_time}')

#         # CPU Memory & Usage
#         memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
#         cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
#         max_memory_usage_mb = max(max_memory_usage_mb, memory_usage_mb)
#         max_cpu_usage_percent = max(max_cpu_usage_percent, cpu_usage_percent)

#         # GPU Usage & Memory
#         if gpu_available:
#             for i in range(num_gpus):
#                 handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#                 util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
#                 mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

#                 gpu_util_percent = util_rates.gpu
#                 gpu_mem_usage_mb = mem_info.used / (1024**2)
#                 max_gpu_usage_percent[i] = max(
#                     max_gpu_usage_percent[i], gpu_util_percent
#                 )
#                 max_gpu_memory_usage_mb[i] = max(
#                     max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
#                 )

#     if gpu_available:
#         pynvml.nvmlShutdown()

#     batched_steps_per_second = NUM_STEPS / total_time

#     return BenchmarkMetrics(
#         batch_size=batch_size,
#         batched_steps_per_second=batched_steps_per_second,
#         max_memory_usage_mb=max_memory_usage_mb,
#         max_cpu_usage_percent=max_cpu_usage_percent,
#         max_gpu_usage_percent=max_gpu_usage_percent,
#         max_gpu_memory_usage_mb=max_gpu_memory_usage_mb,
#     )

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

def benchmark_simple_step_non_empty_world(batch_size: int, agent_count: int, num_steps: int) -> SimpleStepMetrics:
    """Benchmark the performance of our step with varying batch sizes, in an empty world."""
    height = 2
    width = 2
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    print(f"Benchmarking non-empty world of size {height}x{width}, with agent count of {agent_count}")

    print("generate_base_model and misc...")
    start_time = time.time()

    mjx_model, mj_model = generate_base_model(height, width, agent_count)
    agent_components = icland.collect_agent_components(mj_model, agent_count)

    icland_params = ICLandParams(
        model=mj_model, 
        reward_function=None, 
        agent_count=agent_count
    )

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    time_taken = time.time() - start_time
    print(f"generate_base_model & misc. took {time_taken}s")


    print("Batched sample_world...")
    start_time = time.time()
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
    models = batched_sample_world(height, width, 1000, keys, True, 1) 
    time_taken = time.time() - start_time
    print(f"Batched sample_world took {time_taken}s")

    print("Batched export...")
    start_time = time.time()
    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, height, width)
    time_taken = time.time() - start_time
    print(f"Batched export took {time_taken}s")

    
    print("Batched edit_model_data...")
    start_time = time.time()
    batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(
        tilemaps,
        mjx_model,
    )
    time_taken = time.time() - start_time
    print(f"Batched edit_model_data took {time_taken}s")

    print("Batched make mjx_data...")
    start_time = time.time()
    batch_data = jax.vmap(mujoco.mjx.make_data)(batch)
    time_taken = time.time() - start_time
    print(f"Batched make mjx_data took {time_taken}s")
    
    def create_icland_state(mjx_model, mjx_data):
        return ICLandState(
            PipelineState(mjx_model, mjx_data, agent_components),
            jnp.zeros(AGENT_OBSERVATION_DIM),
            collect_body_scene_info(agent_components, mjx_data)
        )

    icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)

    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))

    def scan_step(carry, i):
        icland_states, icland_params, actions = carry

        icland_states = batched_step(keys, icland_states, icland_params, actions)

        return (icland_states, icland_params, actions), None
    
    steps = jax.numpy.arange(num_steps)

    print(f"Starting simulation...")
    start_time = time.time()
    # icland_states = batched_step(keys, icland_states, icland_params, actions)
    initial_carry = (icland_states, icland_params, actions)  # Starting total_time as 0
    output = jax.lax.scan(scan_step, initial_carry, (steps,))
    jax.tree_util.tree_map(
      lambda x: x.block_until_ready(), output
    )
    total_time = time.time() - start_time

    return SimpleStepMetrics(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
    )

def benchmark_simple_step_empty_world(batch_size: int, agent_count: int, num_steps: int) -> SimpleStepMetrics:
    """Benchmark the performance of our step with varying batch sizes, in an empty world."""
    key = jax.random.PRNGKey(SEED)
    icland_params = icland.sample(agent_count, key)
    init_state = icland.init(key, icland_params)

    # Prepare batch
    def replicate(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    icland_states = jax.tree_map(replicate, init_state)
    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    keys = jax.random.split(key, batch_size)

    # Batched step function
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))

    def scan_step(carry, i):
        icland_states, icland_params, actions = carry

        icland_states = batched_step(keys, icland_states, icland_params, actions)

        return (icland_states, icland_params, actions), None
    
    steps = jax.numpy.arange(num_steps-1)

    start_time = time.time()
    icland_states = batched_step(keys, icland_states, icland_params, actions)
    initial_carry = (icland_states, icland_params, actions)  # Starting total_time as 0
    output = jax.lax.scan(scan_step, initial_carry, (steps,))
    jax.tree_util.tree_map(
      lambda x: x.block_until_ready(), output
    )
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time}")

    return SimpleStepMetrics(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
    )

def benchmark_complex_step_empty_world(batch_size: int, agent_count: int, num_steps: int) -> ComplexStepMetrics:
    key = jax.random.PRNGKey(SEED)
    icland_params = icland.sample(agent_count, key)
    init_state = icland.init(key, icland_params)

    # Prepare batch
    def replicate(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    icland_states = jax.tree_map(replicate, init_state)
    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

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
    for s in range(num_steps):
        # The elements in each of the four arrays are the same, except for those in keys

        print(f'Start of batched step {s}')

        step_start_time = time.time()
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

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), icland_states)
        step_time = time.time() - step_start_time
        total_time += step_time

        print(f'End of batched step {s}. Time taken: {step_time}')
        

    if gpu_available:
        pynvml.nvmlShutdown()

    return ComplexStepMetrics(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
        max_memory_usage_mb=max_memory_usage_mb,
        max_cpu_usage_percent=max_cpu_usage_percent,
        max_gpu_usage_percent=max_gpu_usage_percent,
        max_gpu_memory_usage_mb=max_gpu_memory_usage_mb,
    )

def benchmark_render_frame_non_empty_world(batch_size: int, width: int, agent_count: int, num_steps: int) -> tuple[ComplexStepMetrics, ComplexStepMetrics]:
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    print(f"Benchmarking render_frame non-empty world of size {width}x{width}, with agent count of {agent_count}")

    print("generate_base_model and misc...")
    start_time = time.time()

    mjx_model, mj_model = generate_base_model(width, width, agent_count)
    agent_components = icland.collect_agent_components(mj_model, agent_count)

    icland_params = ICLandParams(
        model=mj_model, 
        reward_function=None, 
        agent_count=agent_count
    )

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    time_taken = time.time() - start_time
    print(f"generate_base_model & misc. took {time_taken}s")


    print("Batched sample_world...")
    start_time = time.time()
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
    models = batched_sample_world(width, width, 1000, keys, True, 1) 
    time_taken = time.time() - start_time
    print(f"Batched sample_world took {time_taken}s")

    print("Batched export...")
    start_time = time.time()
    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, width, width)
    time_taken = time.time() - start_time
    print(f"Batched export took {time_taken}s")
    
    print("Batched edit_model_data...")
    start_time = time.time()
    batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(
        tilemaps,
        mjx_model,
    )
    time_taken = time.time() - start_time
    print(f"Batched edit_model_data took {time_taken}s")

    print("Batched make mjx_data...")
    start_time = time.time()
    batch_data = jax.vmap(mujoco.mjx.make_data)(batch)
    time_taken = time.time() - start_time
    print(f"Batched make mjx_data took {time_taken}s")
    
    def create_icland_state(mjx_model, mjx_data):
        return ICLandState(
            PipelineState(mjx_model, mjx_data, agent_components),
            jnp.zeros(AGENT_OBSERVATION_DIM),
            collect_body_scene_info(agent_components, mjx_data)
        )

    # Emulate our own step and run once
    icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)   # batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))

    batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))
    batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))
    batched_render_frame = jax.vmap(render_frame, in_axes=(0, 0, 0))
    print(f"Starting simulation...")

    process = psutil.Process()
    step_max_memory_usage_mb = 0.0
    step_max_cpu_usage_percent = 0.0
    render_frame_max_memory_usage_mb = 0.0
    render_frame_max_cpu_usage_percent = 0.0

    # Attempt to initialize NVML for GPU usage
    gpu_available = True
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        step_max_gpu_usage_percent: list[float] = [0.0] * num_gpus
        step_max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
        render_frame_max_gpu_usage_percent: list[float] = [0.0] * num_gpus
        render_frame_max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
    except pynvml.NVMLError:
        gpu_available = False
        step_max_gpu_usage_percent = []
        step_max_gpu_memory_usage_mb = []
        render_frame_max_gpu_usage_percent = []
        render_frame_max_gpu_memory_usage_mb = []

    default_agent_1 = 0

    camera_poses, camera_dirs = batched_get_camera_info(
        icland_states, width, default_agent_1
    )

    # Timed run
    total_step_time = 0
    total_render_frame_time = 0
    for s in range(num_steps):
        step_start_time = time.time()
        icland_states = batched_step(keys, icland_states, icland_params, actions)

        # CPU Memory & Usage
        memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        step_max_memory_usage_mb = max(step_max_memory_usage_mb, memory_usage_mb)
        step_max_cpu_usage_percent = max(step_max_cpu_usage_percent, cpu_usage_percent)

        # GPU Usage & Memory
        if gpu_available:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util_percent = util_rates.gpu
                gpu_mem_usage_mb = mem_info.used / (1024**2)
                step_max_gpu_usage_percent[i] = max(
                    step_max_gpu_usage_percent[i], gpu_util_percent
                )
                step_max_gpu_memory_usage_mb[i] = max(
                    step_max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
                )

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), icland_states)
        step_time = time.time() - step_start_time
        total_step_time += step_time

        camera_poses, camera_dirs = batched_get_camera_info(
            icland_states, width, default_agent_1
        )

        print(f'Start of batched render_frame step {s}')
        render_frame_start_time = time.time()
        f = batched_render_frame(camera_poses, camera_dirs, tilemaps)        

        # CPU Memory & Usage
        memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        render_frame_max_memory_usage_mb = max(render_frame_max_memory_usage_mb, memory_usage_mb)
        render_frame_max_cpu_usage_percent = max(render_frame_max_cpu_usage_percent, cpu_usage_percent)

        # GPU Usage & Memory
        if gpu_available:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util_percent = util_rates.gpu
                gpu_mem_usage_mb = mem_info.used / (1024**2)
                render_frame_max_gpu_usage_percent[i] = max(
                    render_frame_max_gpu_usage_percent[i], gpu_util_percent
                )
                render_frame_max_gpu_memory_usage_mb[i] = max(
                    render_frame_max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
                )

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), f)
        render_frame_time = time.time() - render_frame_start_time
        total_render_frame_time += render_frame_time
        print(f'End of batched render_frame step {s}. Time taken: {render_frame_time}')

    if gpu_available:
        pynvml.nvmlShutdown()

    return (
        ComplexStepMetrics(
            batch_size=batch_size,
            num_steps=num_steps,
            total_time=total_step_time,
            max_memory_usage_mb=step_max_memory_usage_mb,
            max_cpu_usage_percent=step_max_cpu_usage_percent,
            max_gpu_usage_percent=step_max_gpu_usage_percent,
            max_gpu_memory_usage_mb=step_max_gpu_memory_usage_mb,
        ),
        ComplexStepMetrics(
            batch_size=batch_size,
            num_steps=num_steps,
            total_time=total_render_frame_time,
            max_memory_usage_mb=render_frame_max_memory_usage_mb,
            max_cpu_usage_percent=render_frame_max_cpu_usage_percent,
            max_gpu_usage_percent=render_frame_max_gpu_usage_percent,
            max_gpu_memory_usage_mb=render_frame_max_gpu_memory_usage_mb,
        )
    )

def benchmark_render_frame_empty_world(batch_size: int, agent_count: int, num_steps: int) -> tuple[ComplexStepMetrics, ComplexStepMetrics]:
    width = 2
    key = jax.random.PRNGKey(SEED)
    icland_params = icland.sample(agent_count, key)
    init_state = icland.init(key, icland_params)

    # Prepare batch
    def replicate(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    icland_states = jax.tree_map(replicate, init_state)
    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    keys = jax.random.split(key, batch_size)

    # Batched step function
    batched_step = jax.vmap(icland.step, in_axes=(None, 0, None, 0))
    batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))
    batched_render_frame = jax.vmap(render_frame, in_axes=(0, 0, None))

    process = psutil.Process()
    step_max_memory_usage_mb = 0.0
    step_max_cpu_usage_percent = 0.0
    render_frame_max_memory_usage_mb = 0.0
    render_frame_max_cpu_usage_percent = 0.0

    # Attempt to initialize NVML for GPU usage
    gpu_available = True
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        step_max_gpu_usage_percent: list[float] = [0.0] * num_gpus
        step_max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
        render_frame_max_gpu_usage_percent: list[float] = [0.0] * num_gpus
        render_frame_max_gpu_memory_usage_mb: list[float] = [0.0] * num_gpus
    except pynvml.NVMLError:
        gpu_available = False
        step_max_gpu_usage_percent = []
        step_max_gpu_memory_usage_mb = []
        render_frame_max_gpu_usage_percent = []
        render_frame_max_gpu_memory_usage_mb = []

    default_agent_1 = 0

    camera_poses, camera_dirs = batched_get_camera_info(
        icland_states, width, default_agent_1
    )

    # Timed run
    total_step_time = 0
    total_render_frame_time = 0
    for s in range(num_steps):
        step_start_time = time.time()
        icland_states = batched_step(keys, icland_states, icland_params, actions)

        # CPU Memory & Usage
        memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        step_max_memory_usage_mb = max(step_max_memory_usage_mb, memory_usage_mb)
        step_max_cpu_usage_percent = max(step_max_cpu_usage_percent, cpu_usage_percent)

        # GPU Usage & Memory
        if gpu_available:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util_percent = util_rates.gpu
                gpu_mem_usage_mb = mem_info.used / (1024**2)
                step_max_gpu_usage_percent[i] = max(
                    step_max_gpu_usage_percent[i], gpu_util_percent
                )
                step_max_gpu_memory_usage_mb[i] = max(
                    step_max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
                )

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), icland_states)
        step_time = time.time() - step_start_time
        total_step_time += step_time

        camera_poses, camera_dirs = batched_get_camera_info(
            icland_states, width, default_agent_1
        )

        print(f'Start of batched render_frame step {s}')
        render_frame_start_time = time.time()
        f = batched_render_frame(camera_poses, camera_dirs, TEST_TILEMAP_EMPTY)

        # CPU Memory & Usage
        memory_usage_mb = process.memory_info().rss / (1024**2)  # in MB
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        render_frame_max_memory_usage_mb = max(render_frame_max_memory_usage_mb, memory_usage_mb)
        render_frame_max_cpu_usage_percent = max(render_frame_max_cpu_usage_percent, cpu_usage_percent)

        # GPU Usage & Memory
        if gpu_available:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util_percent = util_rates.gpu
                gpu_mem_usage_mb = mem_info.used / (1024**2)
                render_frame_max_gpu_usage_percent[i] = max(
                    render_frame_max_gpu_usage_percent[i], gpu_util_percent
                )
                render_frame_max_gpu_memory_usage_mb[i] = max(
                    render_frame_max_gpu_memory_usage_mb[i], gpu_mem_usage_mb
                )

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), f)
        render_frame_time = time.time() - render_frame_start_time
        total_render_frame_time += render_frame_time
        print(f'End of batched render_frame step {s}. Time taken: {render_frame_time}')

    if gpu_available:
        pynvml.nvmlShutdown()

    return (
        ComplexStepMetrics(
            batch_size=batch_size,
            num_steps=num_steps,
            total_time=total_step_time,
            max_memory_usage_mb=step_max_memory_usage_mb,
            max_cpu_usage_percent=step_max_cpu_usage_percent,
            max_gpu_usage_percent=step_max_gpu_usage_percent,
            max_gpu_memory_usage_mb=step_max_gpu_memory_usage_mb,
        ),
        ComplexStepMetrics(
            batch_size=batch_size,
            num_steps=num_steps,
            total_time=total_render_frame_time,
            max_memory_usage_mb=render_frame_max_memory_usage_mb,
            max_cpu_usage_percent=render_frame_max_cpu_usage_percent,
            max_gpu_usage_percent=render_frame_max_gpu_usage_percent,
            max_gpu_memory_usage_mb=render_frame_max_gpu_memory_usage_mb,
        )
    )

def benchmark_entire_step_non_empty_world(batch_size: int, width: int, agent_count: int, num_steps: int) -> ComplexStepMetrics:
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)

    # Sample initial conditions
    config = icland.config(
        5,
        5,
        6,
        1,
        0,
        0,
    )

    batched_sample = jax.vmap(icland.sample, in_axes=(0, None))
    icland_params = batched_sample(keys, config)

    batched_init = jax.vmap(icland.init, in_axes=(0))
    icland_states = batched_init(icland_params)

    batched_agent_count = jax.vmap(lambda params: params.agent_info.agent_count)(icland_params)

    action = jnp.array([1, 0, 0, 0, 0, 0])

    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None))

    def scan_step(carry, i):
        icland_states, icland_params, actions = carry

        icland_states = batched_step(icland_states, icland_params, actions)

        return (icland_states, icland_params, actions), None
    
    steps = jax.numpy.arange(int(num_steps / 5))

    print(f"Starting simulation...")
    start_time = time.time()
    # icland_states = batched_step(keys, icland_states, icland_params, actions)
    initial_carry = (icland_states, icland_params, actions)  # Starting total_time as 0
    output = jax.lax.scan(scan_step, initial_carry, (steps,))
    jax.tree_util.tree_map(
      lambda x: x.block_until_ready(), output
    )
    total_time = time.time() - start_time
    print(f"End of simulation. Total time: ${total_time}")

    return SimpleStepMetrics(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
    )

