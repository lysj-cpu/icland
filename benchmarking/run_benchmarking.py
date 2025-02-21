# type: ignore
"""Benchmarking script for ICLand."""

import os
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import psutil
import pynvml

import icland
from icland.presets import DEFAULT_CONFIG

# --------------------------------------------------------------------------------------
# SCENARIOS: a dictionary for all your benchmark scenarios.
#   Each scenario has:
#       - "description": short text describing the scenario
#       - "parameters": dictionary of relevant parameters
# --------------------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "vary_batch_size": {
        "description": "Benchmark performance across different batch sizes.",
        "parameters": {
            "batch_sizes": [2 ** (i * 3) for i in range(0, 6)],
            "num_steps": 100,
            "warmup_steps": 10,
            "seed": 42,
        },
    },
    # In the future, add additional scenarios here.
    # "vary_agent_count": {
    #     "description": "Benchmark performance with varying number of agents.",
    #     "parameters": {
    #         "agent_counts": [10, 50, 100],
    #         "num_steps": 100,
    #         "warmup_steps": 10,
    #         "seed": 42
    #     }
    # }
}


# --------------------------------------------------------------------------------------
# Helper: Set up environment & JAX step function (for example scenario).
# For other scenarios, you might customize or override this part.
# --------------------------------------------------------------------------------------
def setup_environment(seed: int) -> tuple[jnp.ndarray, Any, Any, Callable[..., Any]]:
    """Set up the environment for the benchmarking scenario.

    Args:
        seed (int): Random seed for the environment.

    Returns:
        Tuple containing:
            - key (jnp.ndarray): A JAX PRNG key.
            - icland_params (Any): Sampled parameters from icland.
            - init_state (Any): The initial state for the simulation.
            - batched_step (Callable): The vectorized step function.
    """
    key = jax.random.PRNGKey(seed)
    icland_params = icland.sample(key, DEFAULT_CONFIG)
    base_model = generate_base_model(DEFAULT_CONFIG)
    init_state = icland.init(key, icland_params)

    # Vmap the step function to handle batched states
    batched_step = jax.vmap(icland.step, in_axes=(0, 0, None, 0))
    return key, icland_params, init_state, batched_step


# --------------------------------------------------------------------------------------
# Benchmark / Measurement Functions
# --------------------------------------------------------------------------------------
def measure_compile_time(
    key: jnp.ndarray,
    icland_params: Any,
    init_state: Any,
    batched_step: Callable[..., Any],
    warmup_steps: int,
) -> float:
    """Measure the JAX compile time by calling batched_step on a small batch (size=1).

    Args:
        key (jnp.ndarray): JAX PRNG key.
        icland_params (Any): Parameters for the icland simulation.
        init_state (Any): Initial state for the simulation.
        batched_step (Callable): The vectorized step function.
        warmup_steps (int): Number of warmup steps.

    Returns:
        float: The measured compile time in seconds.
    """
    batch_size = 1
    icland_states = jax.tree_map(lambda x: jnp.stack([x] * batch_size), init_state)
    actions = jnp.array([[1, 0, 0] for _ in range(batch_size)])
    keys = jax.random.split(key, batch_size)

    # Measure the time of the first call (compile time)
    start_compile = time.time()
    _ = batched_step(keys, icland_states, icland_params, actions)
    compile_time_s = time.time() - start_compile

    # Warm up
    for _ in range(warmup_steps):
        icland_states = batched_step(keys, icland_states, icland_params, actions)

    return compile_time_s


def measure_batched_steps_per_second_and_resources(
    batch_size: int,
    key: jnp.ndarray,
    icland_params: Any,
    init_state: Any,
    batched_step: Callable[..., Any],
    num_steps: int = 100,
) -> dict[str, int | float | list[float]]:
    """Measure batched steps per second and resource usage.

    Measures:
      - Steps per second (batched)
      - Max memory usage (CPU)
      - Max CPU usage (percentage)
      - Max GPU usage (percentage)
      - Max GPU memory usage (MB)
    for a given `batch_size` over `num_steps` steps.

    Args:
        batch_size (int): The size of the batch.
        key (jnp.ndarray): JAX PRNG key.
        icland_params (Any): Parameters for the icland simulation.
        init_state (Any): Initial state for the simulation.
        batched_step (Callable): The vectorized step function.
        num_steps (int, optional): Number of steps to run. Defaults to 100.

    Returns:
        Dict[str, Union[int, float, List[float]]]: A dictionary with benchmark metrics.
    """
    # Prepare batch
    icland_states = jax.tree_map(lambda x: jnp.stack([x] * batch_size), init_state)
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
    for _ in range(num_steps):
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

    batched_steps_per_second = num_steps / total_time

    return {
        "batch_size": batch_size,
        "batched_steps_per_second": batched_steps_per_second,
        "max_memory_usage_mb": max_memory_usage_mb,
        "max_cpu_usage_percent": max_cpu_usage_percent,
        "max_gpu_usage_percent": max_gpu_usage_percent,
        "max_gpu_memory_usage_mb": max_gpu_memory_usage_mb,
    }


def plot_metric(
    x_values: list[int | float],
    metric_list: list[float | list[float]],
    ylabel: str,
    xlabel: str,
    title: str,
    filename: str,
    labels: list[str] | None = None,
) -> str:
    """General plotting utility for one or more lines on the same figure.

    Saves figure in PNG format.

    Args:
        x_values (List[Union[int, float]]): Values for the x-axis.
        metric_list (List[Union[float, List[float]]]): Metric values to plot.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        title (str): Plot title.
        filename (str): Base filename for saving the plot.
        labels (Optional[List[str]], optional): Labels for each line if multiple. Defaults to None.

    Returns:
        str: The final saved PNG file path.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(8, 5))
    if labels:
        for i, label in enumerate(labels):
            plt.plot(x_values, [m[i] for m in metric_list], marker="o", label=label)
        plt.legend()
    else:
        plt.plot(x_values, metric_list, marker="o")
    plt.xlabel(xlabel)  # Will adjust per scenario
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    png_filename = filename + ".png"
    plt.savefig(png_filename, format="png")
    plt.close()
    return png_filename


# --------------------------------------------------------------------------------------
# SCENARIO RUNNER
#   - We’ll define a single function that, given a scenario name + details,
#     runs the relevant benchmarking steps, collects results, and returns them.
# --------------------------------------------------------------------------------------
def run_scenario_benchmark(
    scenario_name: str, scenario_def: dict[str, Any]
) -> dict[str, Any]:
    """Runs one scenario's benchmarks, including compile time measurement.

    Returns a dictionary with scenario results: compile_time, benchmark_runs, graphics, etc.

    Args:
        scenario_name (str): The name of the scenario.
        scenario_def (Dict[str, Any]): The scenario definition.

    Returns:
        Dict[str, Any]: The results of the scenario benchmark.
    """
    desc = scenario_def["description"]
    params = scenario_def["parameters"]

    # Setup environment (with scenario-specific seed, etc.)
    key, icland_params, init_state, batched_step = setup_environment(params["seed"])

    # 1) Measure compile time (unique per scenario)
    compile_time_s = measure_compile_time(
        key=key,
        icland_params=icland_params,
        init_state=init_state,
        batched_step=batched_step,
        warmup_steps=params["warmup_steps"],
    )

    # 2) Actual runs (example: vary batch size)
    benchmark_runs: list[dict[str, int | float | list[float]]] = []
    if "batch_sizes" in params:
        for b_size in params["batch_sizes"]:
            metrics = measure_batched_steps_per_second_and_resources(
                batch_size=b_size,
                key=key,
                icland_params=icland_params,
                init_state=init_state,
                batched_step=batched_step,
                num_steps=params["num_steps"],
            )
            benchmark_runs.append(metrics)
    else:
        # Future scenario: e.g., "agent_counts"
        # You or a future function would handle it similarly.
        pass

    # 3) Generate scenario-specific plots (optional)
    #    For "vary_batch_size", we can plot e.g. Steps/sec vs. batch_size
    graphics: dict[str, str | None] = {}
    if "batch_sizes" in params and len(benchmark_runs) > 0:
        output_dir = f"benchmarking/output/graphics/{scenario_name}"
        # X-axis is batch_sizes
        batch_sizes = [run["batch_size"] for run in benchmark_runs]

        # Steps/sec
        steps_per_sec_list = [run["batched_steps_per_second"] for run in benchmark_runs]
        steps_graph_path = plot_metric(
            x_values=batch_sizes,
            metric_list=steps_per_sec_list,
            xlabel="Batch Size",
            ylabel="Batched Steps per Second",
            title=f"{scenario_name}: Steps/sec vs. Batch Size",
            filename=os.path.join(output_dir, "steps_per_second_vs_batch_size"),
        )
        graphics["batched_steps_per_second"] = steps_graph_path

        # Memory
        mem_list = [run["max_memory_usage_mb"] for run in benchmark_runs]
        mem_graph_path = plot_metric(
            x_values=batch_sizes,
            metric_list=mem_list,
            xlabel="Batch Size",
            ylabel="Max Memory Usage (MB)",
            title=f"{scenario_name}: Max Memory vs. Batch Size",
            filename=os.path.join(output_dir, "max_memory_usage_vs_batch_size"),
        )
        graphics["max_memory_usage"] = mem_graph_path

        # CPU
        cpu_list = [run["max_cpu_usage_percent"] for run in benchmark_runs]
        cpu_graph_path = plot_metric(
            x_values=batch_sizes,
            xlabel="Batch Size",
            metric_list=cpu_list,
            ylabel="Max CPU Usage (%)",
            title=f"{scenario_name}: Max CPU vs. Batch Size",
            filename=os.path.join(output_dir, "max_cpu_usage_vs_batch_size"),
        )
        graphics["max_cpu_usage"] = cpu_graph_path

        # GPU
        # This is a list of lists, one per GPU
        has_gpu = any(len(run["max_gpu_usage_percent"]) > 0 for run in benchmark_runs)
        if has_gpu:
            max_gpu_usage_list = [
                run["max_gpu_usage_percent"] for run in benchmark_runs
            ]
            # Number of GPUs from first non-empty
            num_gpus = len(
                next(
                    r["max_gpu_usage_percent"]
                    for r in benchmark_runs
                    if len(r["max_gpu_usage_percent"]) > 0
                )
            )
            gpu_labels = [f"GPU {i}" for i in range(num_gpus)]
            gpu_graph_path = plot_metric(
                x_values=batch_sizes,
                metric_list=max_gpu_usage_list,
                ylabel="Max GPU Usage (%)",
                xlabel="Batch Size",
                title=f"{scenario_name}: Max GPU vs. Batch Size",
                filename=os.path.join(output_dir, "max_gpu_usage_vs_batch_size"),
                labels=gpu_labels,
            )
            graphics["max_gpu_usage"] = gpu_graph_path
        else:
            graphics["max_gpu_usage"] = None
    # For a different scenario, you'd generate different plots.

    # Return scenario results
    scenario_results: dict[str, Any] = {
        "description": desc,
        "compile_time_s": compile_time_s,
        "benchmark_runs": benchmark_runs,
        "graphics": graphics,
    }
    return scenario_results


# --------------------------------------------------------------------------------------
# MAIN ENTRY POINT
#   - run_all_scenarios: iterates over SCENARIOS, calls run_scenario_benchmark
#   - returns a dictionary keyed by scenario name
# --------------------------------------------------------------------------------------
def run_all_scenarios(
    scenarios_dict: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Runs all scenarios in the SCENARIOS dictionary.

    Args:
        scenarios_dict (Optional[Dict[str, Dict[str, Any]]], optional): A dictionary
            of scenarios. Defaults to None, in which case the SCENARIOS global is used.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary of scenario results keyed by scenario name.
    """
    if scenarios_dict is None:
        scenarios_dict = SCENARIOS

    all_results: dict[str, dict[str, Any]] = {}
    for scenario_name, scenario_def in scenarios_dict.items():
        print(f"\n--- Running scenario: {scenario_name} ---")
        scenario_results = run_scenario_benchmark(scenario_name, scenario_def)
        all_results[scenario_name] = scenario_results

    return all_results


# --------------------------------------------------------------------------------------
# If you'd like to run from CLI (optional):
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     results = run_all_scenarios()
#     print("All scenario results:", results)
