import jax
import icland
import mujoco
import jax.numpy as jnp
from icland import *
from icland.world_gen.JITModel import sample_world, export
from icland.world_gen.model_editing import edit_model_data, generate_base_model
from icland.world_gen.tile_data import TILECODES
from icland.renderer.renderer import generate_colormap, get_agent_camera_from_mjx
from mujoco_playground._src.mjx_env import MjxEnv, State, config_dict
from typing import Any, Dict, Optional, Union
import psutil
import pynvml
import time

class PlaygroundMjxEnv(MjxEnv):
    def __init__(self, config: config_dict.ConfigDict, config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._mjx_model = mjx.put_model(self._mj_model)

    def reset(self, rng: jax.Array) -> State:
        data = init(self._mjx_model)
        obs = self._get_observation(data)
        return State(data=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()), metrics={}, info={})

    def step(self, state: State, action: jax.Array) -> State:
        data = step(self._mjx_model, state.data, action, self.n_substeps)
        obs = self._get_observation(data)
        reward = self._get_reward(state, action, data)
        done = self._get_done(state, action, data)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    @property
    def xml_path(self) -> str:
        return "path/to/your/environment.xml"

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def _get_observation(self, data: mjx.Data) -> jax.Array:
        # Implement your observation extraction logic here
        return jnp.zeros((self.observation_size,))

    def _get_reward(self, state: State, action: jax.Array, data: mjx.Data) -> jax.Array:
        # Implement your reward calculation logic here
        return jnp.zeros(())

    def _get_done(self, state: State, action: jax.Array, data: mjx.Data) -> jax.Array:
        # Implement your done condition logic here
        return jnp.zeros(())

def benchmark(func):
    def wrapper():
        func()
        return wrapper
    return wrapper

def benchmark_renderer_mujoco_playground(batch_size: int) -> None:
    NUM_STEPS = 20
    SEED = 42
    height, width, agent_count = 1, 1, 1
    key = jax.random.key(SEED)
    keys = jax.random.split(key, batch_size)
    print(f"Benchmarking Mujoco Playground renderer with batch size {batch_size}")

    # Generate environment
    batched_sample_world = jax.vmap(sample_world, in_axes=(None, None, None, 0, None, None))
    models = batched_sample_world(height, width, 1000, keys, True, 1)

    batched_export = jax.vmap(export, in_axes=(0, None, None, None))
    tilemaps = batched_export(models, TILECODES, height, width)

    mjx_model, mj_model = generate_base_model(height, width, agent_count)
    agent_components = icland.collect_agent_components(mj_model, agent_count)

    batch = jax.jit(jax.vmap(edit_model_data, in_axes=(0, None)))(tilemaps, mjx_model)
    batch_data = jax.vmap(mujoco.mjx.make_data)(batch)

    def create_icland_state(mjx_model, mjx_data):
        return ICLandState(
            PipelineState(mjx_model, mjx_data, agent_components),
            jnp.zeros(AGENT_OBSERVATION_DIM),
            collect_body_scene_info(agent_components, mjx_data)
        )

    icland_states = jax.vmap(create_icland_state, in_axes=(0, 0))(batch, batch_data)
    batched_get_camera_info = jax.vmap(get_agent_camera_from_mjx, in_axes=(0, None, None))
    batched_render_frame = jax.vmap(mujoco_playground_render, in_axes=(0, 0, 0))

    print("Starting simulation...")
    process = psutil.Process()
    max_memory_usage_mb, max_cpu_usage_percent = 0.0, 0.0

    gpu_available = True
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        max_gpu_usage_percent = [0.0] * num_gpus
        max_gpu_memory_usage_mb = [0.0] * num_gpus
    except pynvml.NVMLError:
        gpu_available = False
        max_gpu_usage_percent, max_gpu_memory_usage_mb = [], []

    camera_poses, camera_dirs = batched_get_camera_info(icland_states, width, 0)

    total_time = 0
    for i in range(NUM_STEPS):
        step_start_time = time.time()
        _ = batched_render_frame(camera_poses, camera_dirs, tilemaps)
        step_time = time.time() - step_start_time
        total_time += step_time

        memory_usage_mb = process.memory_info().rss / (1024**2)
        cpu_usage_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
        max_memory_usage_mb = max(max_memory_usage_mb, memory_usage_mb)
        max_cpu_usage_percent = max(max_cpu_usage_percent, cpu_usage_percent)

        if gpu_available:
            for j in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(j)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                max_gpu_usage_percent[j] = max(max_gpu_usage_percent[j], util_rates.gpu)
                max_gpu_memory_usage_mb[j] = max(max_gpu_memory_usage_mb[j], mem_info.used / (1024**2))

    if gpu_available:
        pynvml.nvmlShutdown()

    batched_steps_per_second = NUM_STEPS / total_time
    print(f"Average steps per second: {batched_steps_per_second}")


    return



if __name__ == "__main__":
    benchmark_renderer_mujoco_playground(2)
