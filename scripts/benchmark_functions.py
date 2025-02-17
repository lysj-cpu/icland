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
from icland.agent import create_agent
from icland.game import generate_game
from icland.renderer.renderer import get_agent_camera_from_mjx, render_frame
from icland.types import ICLandParams
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.converter import create_world, export_stls, sample_spawn_points
from icland.world_gen.tile_data import TILECODES
from video_generator import generate_mjcf_string

SEED = 42

TWO_AGENT_EMPTY_WORLD_COLLIDE = """
<mujoco>
  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="agent0" pos="0 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent0_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        mass="1"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        mass="0"
      />
    </body>

    <body name="agent1" pos="1 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent1_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        mass="1"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        mass="0"
      />
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

  </worldbody>
</mujoco>
"""

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

# def benchmark_renderer_empty_world(batch_size: int) -> BenchmarkMetrics:
#     NUM_STEPS = 100

#     key = jax.random.PRNGKey(SEED)
#     icland_params = icland.sample(key)
#     init_state = icland.init(key, icland_params)

#     # Batched step function
#     batched_step = jax.vmap(icland.step, in_axes=(0, 0, icland_params, 0))

#     # Prepare batch
#     def replicate(x):
#         return jnp.broadcast_to(x, (batch_size,) + x.shape)

#     icland_states = jax.tree_map(replicate, init_state)
#     actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

#     # Old code
#     # icland_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), init_state)
#     # actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])

#     keys = jax.random.split(key, batch_size)

#     batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))

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
#     world_width = 10

#     # Timed run
#     total_time = 0
#     for i in range(NUM_STEPS):
#         # The elements in each of the four arrays are the same, except for those in keys
#         icland_states = batched_step(keys, icland_states, icland_params, actions)

#         camera_pos, camera_dir = batched_get_camera_info(
#             icland_states, world_width, default_agent_1
#         )
#         # print("Got camera angle")
#         f = render_frame(
#             camera_pos, camera_dir, tilemap, view_width=96, view_height=72
#         )
        
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



def benchmark_step_non_empty_world(batch_size: int) -> BenchmarkMetrics:
    NUM_STEPS = 100
    height = 5
    width = 5
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)

    # Maybe switch to use np ops instead of list comprehension
    print(f"Before sample_world...")
    models = [sample_world(height, width, 1000, key, True, 1) for _ in range(batch_size)]

    print(f"Before export...")
    tilemaps = [export(model, TILECODES, height, width) for model in models]

    print(f"Before sample spawn points...")
    spawnpoints = [sample_spawn_points(key, tilemap) for tilemap in tilemaps]

    # create_world is not fully jitted
    print(f"Before create world...")
    pieces = [create_world(tilemap) for tilemap in tilemaps]

    temp_dir = "temp"
    os.makedirs(f"{temp_dir}", exist_ok=True)

    print(f"Before export stls...")
    for i in range(batch_size):
        os.makedirs(f"{temp_dir}/world_{i}", exist_ok=True)
        export_stls(pieces[i], f"{temp_dir}/world_{i}/stl")

    print(f"Before generate mjcf string...")
    xml_strs = [generate_mjcf_string(spawnpoints[i], f"{temp_dir}/world_{i}") for i in range(batch_size)]
    
    print(f"Before batched from xml string...")
    mj_models = [mujoco.MjModel.from_xml_string(xml_str) for xml_str in xml_strs]
    
    icland_params = [
        ICLandParams(
            model=mj_model, 
            reward_function=None, 
            agent_count=1
        ) for mj_model in mj_models
    ]

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    print(f"Before icland.init...")
    icland_states = [icland.init(key, params) for params in icland_params]

    print(f"Before batched step...")
    icland_states = [icland.step(key, state, None, action) for state, action in zip(icland_states, actions)]
    icland_states = jax.tree_map(lambda *x: jnp.stack(x), *icland_states)

    # This seems bad
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))

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



def benchmark_batch_size(batch_size: int) -> BenchmarkMetrics:
    """Benchmark the performance of our step with varying batch sizes."""
    NUM_STEPS = 100

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
