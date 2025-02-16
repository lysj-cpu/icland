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
WORLD_42_CONVEX = """
<mujoco model="generated_mesh_world">
    <compiler meshdir="tests/assets/meshes/"/>
    <default>
        <geom type="mesh" />
    </default>
    
    <worldbody>
            <body name="agent0" pos="1.5 1 4">
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
        <geom name="world_0" mesh="world_0" pos="0 0 0"/>
        <geom name="world_1" mesh="world_1" pos="0 0 0"/>
        <geom name="world_10" mesh="world_10" pos="0 0 0"/>
        <geom name="world_11" mesh="world_11" pos="0 0 0"/>
        <geom name="world_12" mesh="world_12" pos="0 0 0"/>
        <geom name="world_13" mesh="world_13" pos="0 0 0"/>
        <geom name="world_14" mesh="world_14" pos="0 0 0"/>
        <geom name="world_15" mesh="world_15" pos="0 0 0"/>
        <geom name="world_16" mesh="world_16" pos="0 0 0"/>
        <geom name="world_17" mesh="world_17" pos="0 0 0"/>
        <geom name="world_18" mesh="world_18" pos="0 0 0"/>
        <geom name="world_19" mesh="world_19" pos="0 0 0"/>
        <geom name="world_2" mesh="world_2" pos="0 0 0"/>
        <geom name="world_20" mesh="world_20" pos="0 0 0"/>
        <geom name="world_21" mesh="world_21" pos="0 0 0"/>
        <geom name="world_22" mesh="world_22" pos="0 0 0"/>
        <geom name="world_23" mesh="world_23" pos="0 0 0"/>
        <geom name="world_24" mesh="world_24" pos="0 0 0"/>
        <geom name="world_25" mesh="world_25" pos="0 0 0"/>
        <geom name="world_26" mesh="world_26" pos="0 0 0"/>
        <geom name="world_27" mesh="world_27" pos="0 0 0"/>
        <geom name="world_28" mesh="world_28" pos="0 0 0"/>
        <geom name="world_29" mesh="world_29" pos="0 0 0"/>
        <geom name="world_3" mesh="world_3" pos="0 0 0"/>
        <geom name="world_30" mesh="world_30" pos="0 0 0"/>
        <geom name="world_31" mesh="world_31" pos="0 0 0"/>
        <geom name="world_32" mesh="world_32" pos="0 0 0"/>
        <geom name="world_33" mesh="world_33" pos="0 0 0"/>
        <geom name="world_34" mesh="world_34" pos="0 0 0"/>
        <geom name="world_35" mesh="world_35" pos="0 0 0"/>
        <geom name="world_36" mesh="world_36" pos="0 0 0"/>
        <geom name="world_37" mesh="world_37" pos="0 0 0"/>
        <geom name="world_38" mesh="world_38" pos="0 0 0"/>
        <geom name="world_39" mesh="world_39" pos="0 0 0"/>
        <geom name="world_4" mesh="world_4" pos="0 0 0"/>
        <geom name="world_40" mesh="world_40" pos="0 0 0"/>
        <geom name="world_41" mesh="world_41" pos="0 0 0"/>
        <geom name="world_42" mesh="world_42" pos="0 0 0"/>
        <geom name="world_43" mesh="world_43" pos="0 0 0"/>
        <geom name="world_44" mesh="world_44" pos="0 0 0"/>
        <geom name="world_45" mesh="world_45" pos="0 0 0"/>
        <geom name="world_46" mesh="world_46" pos="0 0 0"/>
        <geom name="world_47" mesh="world_47" pos="0 0 0"/>
        <geom name="world_48" mesh="world_48" pos="0 0 0"/>
        <geom name="world_49" mesh="world_49" pos="0 0 0"/>
        <geom name="world_5" mesh="world_5" pos="0 0 0"/>
        <geom name="world_50" mesh="world_50" pos="0 0 0"/>
        <geom name="world_51" mesh="world_51" pos="0 0 0"/>
        <geom name="world_52" mesh="world_52" pos="0 0 0"/>
        <geom name="world_53" mesh="world_53" pos="0 0 0"/>
        <geom name="world_54" mesh="world_54" pos="0 0 0"/>
        <geom name="world_55" mesh="world_55" pos="0 0 0"/>
        <geom name="world_56" mesh="world_56" pos="0 0 0"/>
        <geom name="world_57" mesh="world_57" pos="0 0 0"/>
        <geom name="world_58" mesh="world_58" pos="0 0 0"/>
        <geom name="world_59" mesh="world_59" pos="0 0 0"/>
        <geom name="world_6" mesh="world_6" pos="0 0 0"/>
        <geom name="world_60" mesh="world_60" pos="0 0 0"/>
        <geom name="world_61" mesh="world_61" pos="0 0 0"/>
        <geom name="world_62" mesh="world_62" pos="0 0 0"/>
        <geom name="world_63" mesh="world_63" pos="0 0 0"/>
        <geom name="world_64" mesh="world_64" pos="0 0 0"/>
        <geom name="world_65" mesh="world_65" pos="0 0 0"/>
        <geom name="world_66" mesh="world_66" pos="0 0 0"/>
        <geom name="world_67" mesh="world_67" pos="0 0 0"/>
        <geom name="world_68" mesh="world_68" pos="0 0 0"/>
        <geom name="world_69" mesh="world_69" pos="0 0 0"/>
        <geom name="world_7" mesh="world_7" pos="0 0 0"/>
        <geom name="world_70" mesh="world_70" pos="0 0 0"/>
        <geom name="world_71" mesh="world_71" pos="0 0 0"/>
        <geom name="world_72" mesh="world_72" pos="0 0 0"/>
        <geom name="world_73" mesh="world_73" pos="0 0 0"/>
        <geom name="world_74" mesh="world_74" pos="0 0 0"/>
        <geom name="world_75" mesh="world_75" pos="0 0 0"/>
        <geom name="world_76" mesh="world_76" pos="0 0 0"/>
        <geom name="world_77" mesh="world_77" pos="0 0 0"/>
        <geom name="world_78" mesh="world_78" pos="0 0 0"/>
        <geom name="world_79" mesh="world_79" pos="0 0 0"/>
        <geom name="world_8" mesh="world_8" pos="0 0 0"/>
        <geom name="world_80" mesh="world_80" pos="0 0 0"/>
        <geom name="world_81" mesh="world_81" pos="0 0 0"/>
        <geom name="world_82" mesh="world_82" pos="0 0 0"/>
        <geom name="world_83" mesh="world_83" pos="0 0 0"/>
        <geom name="world_84" mesh="world_84" pos="0 0 0"/>
        <geom name="world_85" mesh="world_85" pos="0 0 0"/>
        <geom name="world_86" mesh="world_86" pos="0 0 0"/>
        <geom name="world_87" mesh="world_87" pos="0 0 0"/>
        <geom name="world_88" mesh="world_88" pos="0 0 0"/>
        <geom name="world_89" mesh="world_89" pos="0 0 0"/>
        <geom name="world_9" mesh="world_9" pos="0 0 0"/>
        <geom name="world_90" mesh="world_90" pos="0 0 0"/>
        <geom name="world_91" mesh="world_91" pos="0 0 0"/>
        <geom name="world_92" mesh="world_92" pos="0 0 0"/>
        <geom name="world_93" mesh="world_93" pos="0 0 0"/>
        <geom name="world_94" mesh="world_94" pos="0 0 0"/>
        <geom name="world_95" mesh="world_95" pos="0 0 0"/>
        <geom name="world_96" mesh="world_96" pos="0 0 0"/>
        <geom name="world_97" mesh="world_97" pos="0 0 0"/>
        <geom name="world_98" mesh="world_98" pos="0 0 0"/>
        <geom name="world_99" mesh="world_99" pos="0 0 0"/>
    </worldbody>

    <asset>
        <mesh name="world_0" file="world_0.stl"/>
        <mesh name="world_1" file="world_1.stl"/>
        <mesh name="world_10" file="world_10.stl"/>
        <mesh name="world_11" file="world_11.stl"/>
        <mesh name="world_12" file="world_12.stl"/>
        <mesh name="world_13" file="world_13.stl"/>
        <mesh name="world_14" file="world_14.stl"/>
        <mesh name="world_15" file="world_15.stl"/>
        <mesh name="world_16" file="world_16.stl"/>
        <mesh name="world_17" file="world_17.stl"/>
        <mesh name="world_18" file="world_18.stl"/>
        <mesh name="world_19" file="world_19.stl"/>
        <mesh name="world_2" file="world_2.stl"/>
        <mesh name="world_20" file="world_20.stl"/>
        <mesh name="world_21" file="world_21.stl"/>
        <mesh name="world_22" file="world_22.stl"/>
        <mesh name="world_23" file="world_23.stl"/>
        <mesh name="world_24" file="world_24.stl"/>
        <mesh name="world_25" file="world_25.stl"/>
        <mesh name="world_26" file="world_26.stl"/>
        <mesh name="world_27" file="world_27.stl"/>
        <mesh name="world_28" file="world_28.stl"/>
        <mesh name="world_29" file="world_29.stl"/>
        <mesh name="world_3" file="world_3.stl"/>
        <mesh name="world_30" file="world_30.stl"/>
        <mesh name="world_31" file="world_31.stl"/>
        <mesh name="world_32" file="world_32.stl"/>
        <mesh name="world_33" file="world_33.stl"/>
        <mesh name="world_34" file="world_34.stl"/>
        <mesh name="world_35" file="world_35.stl"/>
        <mesh name="world_36" file="world_36.stl"/>
        <mesh name="world_37" file="world_37.stl"/>
        <mesh name="world_38" file="world_38.stl"/>
        <mesh name="world_39" file="world_39.stl"/>
        <mesh name="world_4" file="world_4.stl"/>
        <mesh name="world_40" file="world_40.stl"/>
        <mesh name="world_41" file="world_41.stl"/>
        <mesh name="world_42" file="world_42.stl"/>
        <mesh name="world_43" file="world_43.stl"/>
        <mesh name="world_44" file="world_44.stl"/>
        <mesh name="world_45" file="world_45.stl"/>
        <mesh name="world_46" file="world_46.stl"/>
        <mesh name="world_47" file="world_47.stl"/>
        <mesh name="world_48" file="world_48.stl"/>
        <mesh name="world_49" file="world_49.stl"/>
        <mesh name="world_5" file="world_5.stl"/>
        <mesh name="world_50" file="world_50.stl"/>
        <mesh name="world_51" file="world_51.stl"/>
        <mesh name="world_52" file="world_52.stl"/>
        <mesh name="world_53" file="world_53.stl"/>
        <mesh name="world_54" file="world_54.stl"/>
        <mesh name="world_55" file="world_55.stl"/>
        <mesh name="world_56" file="world_56.stl"/>
        <mesh name="world_57" file="world_57.stl"/>
        <mesh name="world_58" file="world_58.stl"/>
        <mesh name="world_59" file="world_59.stl"/>
        <mesh name="world_6" file="world_6.stl"/>
        <mesh name="world_60" file="world_60.stl"/>
        <mesh name="world_61" file="world_61.stl"/>
        <mesh name="world_62" file="world_62.stl"/>
        <mesh name="world_63" file="world_63.stl"/>
        <mesh name="world_64" file="world_64.stl"/>
        <mesh name="world_65" file="world_65.stl"/>
        <mesh name="world_66" file="world_66.stl"/>
        <mesh name="world_67" file="world_67.stl"/>
        <mesh name="world_68" file="world_68.stl"/>
        <mesh name="world_69" file="world_69.stl"/>
        <mesh name="world_7" file="world_7.stl"/>
        <mesh name="world_70" file="world_70.stl"/>
        <mesh name="world_71" file="world_71.stl"/>
        <mesh name="world_72" file="world_72.stl"/>
        <mesh name="world_73" file="world_73.stl"/>
        <mesh name="world_74" file="world_74.stl"/>
        <mesh name="world_75" file="world_75.stl"/>
        <mesh name="world_76" file="world_76.stl"/>
        <mesh name="world_77" file="world_77.stl"/>
        <mesh name="world_78" file="world_78.stl"/>
        <mesh name="world_79" file="world_79.stl"/>
        <mesh name="world_8" file="world_8.stl"/>
        <mesh name="world_80" file="world_80.stl"/>
        <mesh name="world_81" file="world_81.stl"/>
        <mesh name="world_82" file="world_82.stl"/>
        <mesh name="world_83" file="world_83.stl"/>
        <mesh name="world_84" file="world_84.stl"/>
        <mesh name="world_85" file="world_85.stl"/>
        <mesh name="world_86" file="world_86.stl"/>
        <mesh name="world_87" file="world_87.stl"/>
        <mesh name="world_88" file="world_88.stl"/>
        <mesh name="world_89" file="world_89.stl"/>
        <mesh name="world_9" file="world_9.stl"/>
        <mesh name="world_90" file="world_90.stl"/>
        <mesh name="world_91" file="world_91.stl"/>
        <mesh name="world_92" file="world_92.stl"/>
        <mesh name="world_93" file="world_93.stl"/>
        <mesh name="world_94" file="world_94.stl"/>
        <mesh name="world_95" file="world_95.stl"/>
        <mesh name="world_96" file="world_96.stl"/>
        <mesh name="world_97" file="world_97.stl"/>
        <mesh name="world_98" file="world_98.stl"/>
        <mesh name="world_99" file="world_99.stl"/>
    </asset>
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
    height = 10
    width = 10
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)

    # Maybe switch to use np ops instead of list comprehension
    print(f"Before sample_world...")
    models = [sample_world(height, width, 1000, key, True, 1) for key in keys]

    print(f"Before export...")
    tilemaps = [export(model, TILECODES, height, width) for model in models]

    print(f"Before sample spawn points...")
    spawnpoints = [sample_spawn_points(key, tilemap) for key, tilemap in zip(keys, tilemaps)]

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

    actions = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

    icland_params = [icland.sample(key) for key in keys]

    print(f"Before icland.init...")
    icland_states = [icland.init(key, params) for key, params in zip(keys, icland_params)]

    icland_states = [icland.step(key, state, None, action) for key, state, action in zip(keys, icland_states, actions)]
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
