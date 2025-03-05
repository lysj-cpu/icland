import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math

def linux_to_windows_path(linux_path):
   print(linux_path)
   return linux_path.replace('/', '\\')


def round_to_sig_figs(num, sig_figs=3):
    if num == 0:
        return 0
    magnitude = math.floor(math.log10(abs(num)))
    roundingPos = -magnitude + (sig_figs - 1)
    rounded = round(num, roundingPos)
    if rounded.is_integer():
      rounded = int(rounded)
    return '{:,}'.format(rounded)

def plot_simple_complex_step_graph(output_dir: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    # Can't involve CPU because it was infeasible to do a 5000 step run on CPU
    gpu_simple_100_steps_json = "step_gpu_1_agent_simple_100_steps.json"
    gpu_complex_100_steps_json = "step_gpu_1_agent_complex_100_steps.json"
    gpu_simple_5000_steps_json = "step_gpu_1_agent_simple_5000_steps.json"
    gpu_complex_5000_steps_json = "step_gpu_1_agent_complex_5000_steps.json"

    gpu_simple_100_steps_json = os.path.join(input_dir, gpu_simple_100_steps_json)
    gpu_complex_100_steps_json = os.path.join(input_dir, gpu_complex_100_steps_json)
    gpu_simple_5000_steps_json = os.path.join(input_dir, gpu_simple_5000_steps_json)
    gpu_complex_5000_steps_json = os.path.join(input_dir, gpu_complex_5000_steps_json)


    with open(gpu_simple_100_steps_json, "r") as file:
        gpu_simple_100_steps_data = json.load(file)
    with open(gpu_complex_100_steps_json, "r") as file:
        gpu_complex_100_steps_data = json.load(file)
    with open(gpu_simple_5000_steps_json, "r") as file:
        gpu_simple_5000_steps_data = json.load(file)
    with open(gpu_complex_5000_steps_json, "r") as file:  
        gpu_complex_5000_steps_data = json.load(file)

    gpu_simple_100_steps_results = gpu_simple_100_steps_data['benchmark_results']
    gpu_complex_100_steps_results = gpu_complex_100_steps_data['benchmark_results']
    gpu_simple_5000_steps_results = gpu_simple_5000_steps_data['benchmark_results']
    gpu_complex_5000_steps_results = gpu_complex_5000_steps_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    _, gpu_simple_100_steps_metrics_list = list(gpu_simple_100_steps_results.items())[0]
    _, gpu_complex_100_steps_metrics_list = list(gpu_complex_100_steps_results.items())[0]
    _, gpu_simple_5000_steps_metrics_list = list(gpu_simple_5000_steps_results.items())[0]
    scenario_name, gpu_complex_5000_steps_metrics_list = list(gpu_complex_5000_steps_results.items())[0]

    batch_sizes = [m['batch_size'] for m in gpu_complex_5000_steps_metrics_list]

    gpu_simple_100_steps = [m['total_time'] for m in gpu_simple_100_steps_metrics_list]
    gpu_complex_100_steps = [m['total_time'] for m in gpu_complex_100_steps_metrics_list]
    gpu_simple_5000_steps = [m['total_time'] for m in gpu_simple_5000_steps_metrics_list]
    gpu_complex_5000_steps = [m['total_time'] for m in gpu_complex_5000_steps_metrics_list]

    plt.figure()
    # Use colour to differentiate between simple and complex, as it is the most important comparison
    plt.plot(batch_sizes, gpu_simple_100_steps, color="green", marker="x", linestyle=':', label="jax.lax.scan (100 steps)")
    plt.plot(batch_sizes, gpu_complex_100_steps, color="red", marker="x", linestyle=':', label="for loop (100 steps)")
    plt.plot(batch_sizes, gpu_simple_5000_steps, color="green", marker="o", label="jax.lax.scan (5000 steps)")
    plt.plot(batch_sizes, gpu_complex_5000_steps, color="red", marker="o", label="for loop (5000 steps)")

    legend_elements = [
      mpatches.Patch(color='green', label='jax.lax.scan'),
      mpatches.Patch(color='red', label='for loop'),
      Line2D([0], [0], color='black', lw=1, marker='x', linestyle=':', label='100 steps'),
      Line2D([0], [0], color='black', lw=1, marker='o', label='5000 steps'),
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Total time (in s)")
    plt.title(f"Total time vs Batch Size ({scenario_name})")
    plot_path = os.path.join(
        output_dir, f"{scenario_name}_simple_vs_complex_time.png"
    )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_simple_step_graph(output_dir: str, cpu_json: str, gpu_json: str, special_points: list[tuple[int, str]], y_axis: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(cpu_json, "r") as file:
        cpu_data = json.load(file)
    with open(gpu_json, "r") as file:
        gpu_data = json.load(file)


    cpu_benchmark_results = cpu_data['benchmark_results']
    gpu_benchmark_results = gpu_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    cpu_scenario_name, cpu_metrics_list = list(cpu_benchmark_results.items())[0]
    gpu_scenario_name, gpu_metrics_list = list(gpu_benchmark_results.items())[0]

    batch_sizes = [m['batch_size'] for m in gpu_metrics_list]

    if y_axis == "time":
      cpu_steps = [m['total_time'] for m in cpu_metrics_list]
      gpu_steps = [m['total_time'] for m in gpu_metrics_list]
    elif y_axis == "steps":
      cpu_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in cpu_metrics_list]
      gpu_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    plt.plot(batch_sizes, cpu_steps, color="blue", marker="o", label="CPU")
    plt.plot(batch_sizes, gpu_steps, color="green", marker="s", label="GPU")
    plt.legend(loc="upper left")

    # Special points
    for x_val, pu in special_points:
        if pu == "cpu":
            y_val = cpu_steps[batch_sizes.index(x_val)]
            opp_y_val = gpu_steps[batch_sizes.index(x_val)]
            color = "blue"
        elif pu == "gpu":
            y_val = gpu_steps[batch_sizes.index(x_val)]
            opp_y_val = cpu_steps[batch_sizes.index(x_val)]
            color = "green"
        else:
            raise ValueError(f"Invalid value for pu: {pu}")

        plt.annotate(
            round_to_sig_figs(y_val), 
            (x_val, y_val), 
            textcoords="offset points",
            xytext=(0, 10) if y_val >= opp_y_val else (0, -20),
            ha="center", 
            fontsize=10,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
        )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({gpu_scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{gpu_scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({gpu_scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{gpu_scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_cpu_gpu_step_graph(output_dir: str, y_axis: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    cpu_empty_world_json = "step_cpu_1_agent_simple_100_steps.json"
    cpu_non_empty_world_json = "step_cpu_1_agent_non_empty_world_simple_100_steps.json"
    gpu_empty_world_json = "step_gpu_1_agent_simple_100_steps.json"
    gpu_non_empty_world_json = "step_gpu_1_agent_simple_non_empty_world_100_steps.json"

    cpu_empty_world_json = os.path.join(input_dir, cpu_empty_world_json)
    cpu_non_empty_world_json = os.path.join(input_dir, cpu_non_empty_world_json)
    gpu_empty_world_json = os.path.join(input_dir, gpu_empty_world_json)
    gpu_non_empty_world_json = os.path.join(input_dir, gpu_non_empty_world_json)

    with open(cpu_empty_world_json, "r") as file:
      cpu_empty_world_data = json.load(file)
    with open(cpu_non_empty_world_json, "r") as file:
      cpu_non_empty_world_data = json.load(file)
    with open(gpu_empty_world_json, "r") as file:
      gpu_empty_world_data = json.load(file)
    with open(gpu_non_empty_world_json, "r") as file:
      gpu_non_empty_world_data = json.load(file)

    cpu_empty_world_results = cpu_empty_world_data['benchmark_results']
    cpu_non_empty_world_results = cpu_non_empty_world_data['benchmark_results']
    gpu_empty_world_results = gpu_empty_world_data['benchmark_results']
    gpu_non_empty_world_results = gpu_non_empty_world_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    _, cpu_empty_world_metrics_list = list(cpu_empty_world_results.items())[0]
    _, cpu_non_empty_world_metrics_list = list(cpu_non_empty_world_results.items())[0]
    _, gpu_empty_world_metrics_list = list(gpu_empty_world_results.items())[0]
    scenario_name, gpu_non_empty_world_metrics_list = list(gpu_non_empty_world_results.items())[0]

    cpu_empty_world_batch_sizes = [m['batch_size'] for m in cpu_empty_world_metrics_list]
    cpu_non_empty_world_batch_sizes = [m['batch_size'] for m in cpu_non_empty_world_metrics_list]
    gpu_empty_world_batch_sizes = [m['batch_size'] for m in gpu_empty_world_metrics_list]
    gpu_non_empty_world_batch_sizes = [m['batch_size'] for m in gpu_non_empty_world_metrics_list]

    if y_axis == "time":
      cpu_empty_world_steps = [m['total_time'] for m in cpu_empty_world_metrics_list]
      cpu_non_empty_world_steps = [m['total_time'] for m in cpu_non_empty_world_metrics_list]
      gpu_empty_world_steps = [m['total_time'] for m in gpu_empty_world_metrics_list]
      gpu_non_empty_world_steps = [m['total_time'] for m in gpu_non_empty_world_metrics_list]
    elif y_axis == "steps":
      cpu_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in cpu_empty_world_metrics_list]
      cpu_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in cpu_non_empty_world_metrics_list]
      gpu_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_empty_world_metrics_list]
      gpu_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_non_empty_world_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    plt.plot(cpu_empty_world_batch_sizes, cpu_empty_world_steps, color="blue", marker="x", linestyle=':', label="CPU (empty world)")
    plt.plot(cpu_non_empty_world_batch_sizes, cpu_non_empty_world_steps, color="blue", marker="o", label="CPU (non-empty world)")
    plt.plot(gpu_empty_world_batch_sizes, gpu_empty_world_steps, color="green", marker="x", linestyle=':', label="GPU (empty world)")
    plt.plot(gpu_non_empty_world_batch_sizes, gpu_non_empty_world_steps, color="green", marker="o", label="GPU (non-empty world)")

    legend_elements = [
      mpatches.Patch(color='blue', label='CPU'),
      mpatches.Patch(color='green', label='GPU'),
      Line2D([0], [0], color='black', lw=1, marker='x', linestyle=':', label='empty world (baseline)'),
      Line2D([0], [0], color='black', lw=1, marker='o', label='non-empty (2x2) world'),
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    # special_points = [(1, "gpu"), (2**9, "cpu"), (2**9, "gpu"), (2048, "gpu"), (16384, "gpu")]
    special_points = [(1, "cpu"), (1, "gpu"), (2**10, "cpu"), (2**10, "gpu"), (16384, "gpu")]
    # Special points
    for x_val, pu in special_points:
        if pu == "cpu":
            y_val = cpu_non_empty_world_steps[cpu_non_empty_world_batch_sizes.index(x_val)]
            opp_y_val = gpu_non_empty_world_steps[gpu_non_empty_world_batch_sizes.index(x_val)]
            color = "blue"
        elif pu == "gpu":
            y_val = gpu_non_empty_world_steps[gpu_non_empty_world_batch_sizes.index(x_val)]
            if (x_val not in cpu_non_empty_world_batch_sizes):
               opp_y_val = 0
            else:
              opp_y_val = cpu_non_empty_world_steps[cpu_non_empty_world_batch_sizes.index(x_val)]
            color = "green"
        else:
            raise ValueError(f"Invalid value for pu: {pu}")

        plt.annotate(
            round_to_sig_figs(y_val), 
            (x_val, y_val), 
            textcoords="offset points",
            xytext=(0, 10) if y_val >= opp_y_val else (0, -15),
            ha="center", 
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
        )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_agents_empty_world_step_graph(output_dir: str, y_axis: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    gpu_1_agent_json = "step_gpu_1_agent_simple_100_steps.json"
    gpu_2_agents_json = "step_gpu_2_agents_simple_100_steps.json"
    gpu_4_agents_json = "step_gpu_4_agents_simple_100_steps.json"

    gpu_1_agent_json = os.path.join(input_dir, gpu_1_agent_json)
    gpu_2_agents_json = os.path.join(input_dir, gpu_2_agents_json)
    gpu_4_agents_json = os.path.join(input_dir, gpu_4_agents_json)

    with open(gpu_1_agent_json, "r") as file:
        gpu_1_agent_data = json.load(file)
    with open(gpu_2_agents_json, "r") as file:
        gpu_2_agents_data = json.load(file)
    with open(gpu_4_agents_json, "r") as file:
        gpu_4_agents_data = json.load(file)


    gpu_1_agent_results = gpu_1_agent_data['benchmark_results']
    gpu_2_agents_results = gpu_2_agents_data['benchmark_results']
    gpu_4_agents_results = gpu_4_agents_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    _, gpu_1_agent_metrics_list = list(gpu_1_agent_results.items())[0]
    _, gpu_2_agents_metrics_list = list(gpu_2_agents_results.items())[0]
    scenario_name, gpu_4_agents_metrics_list = list(gpu_4_agents_results.items())[0]

    gpu_1_agent_batch_sizes = [m['batch_size'] for m in gpu_1_agent_metrics_list]
    gpu_2_agents_batch_sizes = [m['batch_size'] for m in gpu_2_agents_metrics_list]
    gpu_4_agents_batch_sizes = [m['batch_size'] for m in gpu_4_agents_metrics_list]

    if y_axis == "time":
      gpu_1_agent_steps = [m['total_time'] for m in gpu_1_agent_metrics_list]
      gpu_2_agents_steps = [m['total_time'] for m in gpu_2_agents_metrics_list]
      gpu_4_agents_steps = [m['total_time'] for m in gpu_4_agents_metrics_list]
    elif y_axis == "steps":
      gpu_1_agent_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_1_agent_metrics_list]
      gpu_2_agents_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_2_agents_metrics_list]
      gpu_4_agents_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_4_agents_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    plt.plot(gpu_1_agent_batch_sizes, gpu_1_agent_steps, color="red", marker="o", label="1 agent")
    plt.plot(gpu_2_agents_batch_sizes, gpu_2_agents_steps, color="green", marker="s", label="2 agents")
    plt.plot(gpu_4_agents_batch_sizes, gpu_4_agents_steps, color="blue", marker="x", label="4 agents")
    plt.legend(loc="upper left")

    # # Special points
    # for x_val, pu in special_points:
    #     if pu == "cpu":
    #         y_val = cpu_steps[batch_sizes.index(x_val)]
    #         opp_y_val = gpu_steps[batch_sizes.index(x_val)]
    #         color = "blue"
    #     elif pu == "gpu":
    #         y_val = gpu_steps[batch_sizes.index(x_val)]
    #         opp_y_val = cpu_steps[batch_sizes.index(x_val)]
    #         color = "green"
    #     else:
    #         raise ValueError(f"Invalid value for pu: {pu}")

    #     plt.annotate(
    #         round_to_sig_figs(y_val), 
    #         (x_val, y_val), 
    #         textcoords="offset points",
    #         xytext=(0, 10) if y_val >= opp_y_val else (0, -20),
    #         ha="center", 
    #         fontsize=10,
    #         color=color,
    #         bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
    #     )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_agents_step_graph(output_dir: str, y_axis: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    gpu_1_agent_empty_world_json = "step_gpu_1_agent_simple_100_steps.json"
    gpu_2_agents_empty_world_json = "step_gpu_2_agents_simple_100_steps.json"
    gpu_4_agents_empty_world_json = "step_gpu_4_agents_simple_100_steps.json"
    gpu_1_agent_non_empty_world_json = "step_gpu_1_agent_simple_non_empty_world_100_steps.json"
    gpu_2_agents_non_empty_world_json = "step_gpu_2_agents_simple_non_empty_world_100_steps.json"
    gpu_4_agents_non_empty_world_json = "step_gpu_4_agents_simple_non_empty_world_100_steps.json"

    gpu_1_agent_empty_world_json = os.path.join(input_dir, gpu_1_agent_empty_world_json)
    gpu_2_agents_empty_world_json = os.path.join(input_dir, gpu_2_agents_empty_world_json)
    gpu_4_agents_empty_world_json = os.path.join(input_dir, gpu_4_agents_empty_world_json)
    gpu_1_agent_non_empty_world_json = os.path.join(input_dir, gpu_1_agent_non_empty_world_json)
    gpu_2_agents_non_empty_world_json = os.path.join(input_dir, gpu_2_agents_non_empty_world_json)
    gpu_4_agents_non_empty_world_json = os.path.join(input_dir, gpu_4_agents_non_empty_world_json)

    with open(gpu_1_agent_empty_world_json, "r") as file:
        gpu_1_agent_empty_world_data = json.load(file)
    with open(gpu_2_agents_empty_world_json, "r") as file:
        gpu_2_agents_empty_world_data = json.load(file)
    with open(gpu_4_agents_empty_world_json, "r") as file:
        gpu_4_agents_empty_world_data = json.load(file)
    with open(gpu_1_agent_non_empty_world_json, "r") as file:
        gpu_1_agent_non_empty_world_data = json.load(file)
    with open(gpu_2_agents_non_empty_world_json, "r") as file:
        gpu_2_agents_non_empty_world_data = json.load(file)
    with open(gpu_4_agents_non_empty_world_json, "r") as file:
        gpu_4_agents_non_empty_world_data = json.load(file)


    gpu_1_agent_empty_world_results = gpu_1_agent_empty_world_data['benchmark_results']
    gpu_2_agents_empty_world_results = gpu_2_agents_empty_world_data['benchmark_results']
    gpu_4_agents_empty_world_results = gpu_4_agents_empty_world_data['benchmark_results']
    gpu_1_agent_non_empty_world_results = gpu_1_agent_non_empty_world_data['benchmark_results']
    gpu_2_agents_non_empty_world_results = gpu_2_agents_non_empty_world_data['benchmark_results']
    gpu_4_agents_non_empty_world_results = gpu_4_agents_non_empty_world_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    _, gpu_1_agent_empty_world_metrics_list = list(gpu_1_agent_empty_world_results.items())[0]
    _, gpu_2_agents_empty_world_metrics_list = list(gpu_2_agents_empty_world_results.items())[0]
    _, gpu_4_agents_empty_world_metrics_list = list(gpu_4_agents_empty_world_results.items())[0]
    _, gpu_1_agent_non_empty_world_metrics_list = list(gpu_1_agent_non_empty_world_results.items())[0]
    _, gpu_2_agents_non_empty_world_metrics_list = list(gpu_2_agents_non_empty_world_results.items())[0]
    scenario_name, gpu_4_agents_non_empty_world_metrics_list = list(gpu_4_agents_non_empty_world_results.items())[0]

    gpu_1_agent_empty_world_batch_sizes = [m['batch_size'] for m in gpu_1_agent_empty_world_metrics_list]
    gpu_2_agents_empty_world_batch_sizes = [m['batch_size'] for m in gpu_2_agents_empty_world_metrics_list]
    gpu_4_agents_empty_world_batch_sizes = [m['batch_size'] for m in gpu_4_agents_empty_world_metrics_list]
    gpu_1_agent_non_empty_world_batch_sizes = [m['batch_size'] for m in gpu_1_agent_non_empty_world_metrics_list]
    gpu_2_agents_non_empty_world_batch_sizes = [m['batch_size'] for m in gpu_2_agents_non_empty_world_metrics_list]
    gpu_4_agents_non_empty_world_batch_sizes = [m['batch_size'] for m in gpu_4_agents_non_empty_world_metrics_list]

    if y_axis == "time":
      gpu_1_agent_empty_world_steps = [m['total_time'] for m in gpu_1_agent_empty_world_metrics_list]
      gpu_2_agents_empty_world_steps = [m['total_time'] for m in gpu_2_agents_empty_world_metrics_list]
      gpu_4_agents_empty_world_steps = [m['total_time'] for m in gpu_4_agents_empty_world_metrics_list]
      gpu_1_agent_non_empty_world_steps = [m['total_time'] for m in gpu_1_agent_non_empty_world_metrics_list]
      gpu_2_agents_non_empty_world_steps = [m['total_time'] for m in gpu_2_agents_non_empty_world_metrics_list]
      gpu_4_agents_non_empty_world_steps = [m['total_time'] for m in gpu_4_agents_non_empty_world_metrics_list]
    elif y_axis == "steps":
      gpu_1_agent_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_1_agent_empty_world_metrics_list]
      gpu_2_agents_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_2_agents_empty_world_metrics_list]
      gpu_4_agents_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_4_agents_empty_world_metrics_list]
      gpu_1_agent_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_1_agent_non_empty_world_metrics_list]
      gpu_2_agents_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_2_agents_non_empty_world_metrics_list]
      gpu_4_agents_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_4_agents_non_empty_world_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    plt.plot(gpu_1_agent_empty_world_batch_sizes, gpu_1_agent_empty_world_steps, color="red", marker="x", linestyle=":", label="1 agent (empty world)")
    plt.plot(gpu_2_agents_empty_world_batch_sizes, gpu_2_agents_empty_world_steps, color="green", marker="x", linestyle=":", label="2 agents (empty world)")
    plt.plot(gpu_4_agents_empty_world_batch_sizes, gpu_4_agents_empty_world_steps, color="blue", marker="x", linestyle=":", label="4 agents (empty world)")
    plt.plot(gpu_1_agent_non_empty_world_batch_sizes, gpu_1_agent_non_empty_world_steps, color="red", marker="o", label="1 agent (non-empty world)")
    plt.plot(gpu_2_agents_non_empty_world_batch_sizes, gpu_2_agents_non_empty_world_steps, color="green", marker="o", label="2 agents (non-empty world)")
    plt.plot(gpu_4_agents_non_empty_world_batch_sizes, gpu_4_agents_non_empty_world_steps, color="blue", marker="o", label="4 agents (non-empty world)")

    legend_elements = [
      mpatches.Patch(color='red', label='1 agent'),
      mpatches.Patch(color='green', label='2 agents'),
      mpatches.Patch(color='blue', label='4 agents'),
      Line2D([0], [0], color='black', lw=1, marker='x', linestyle=':', label='empty world (baseline)'),
      Line2D([0], [0], color='black', lw=1, marker='o', label='non-empty (2x2) world'),
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    # special_points = [(1, "1"), (1, "2"), (1, "4"), (2**11, "1"), (2**11, "2"), (2**8, "4"), (2**14, "1"), (2**13, "2"), (2**13, "4")]
    special_points = []

    # Special points
    for x_val, pu in special_points:
        if pu == "1":
            y_val = gpu_1_agent_non_empty_world_steps[gpu_1_agent_non_empty_world_batch_sizes.index(x_val)]
            color = "red"
        elif pu == "2":
            y_val = gpu_2_agents_non_empty_world_steps[gpu_2_agents_non_empty_world_batch_sizes.index(x_val)]
            color = "green"
        elif pu == "4":
            y_val = gpu_4_agents_non_empty_world_steps[gpu_4_agents_non_empty_world_batch_sizes.index(x_val)]
            color = "blue"
        else:
            raise ValueError(f"Invalid value for pu: {pu}")

        plt.annotate(
            round_to_sig_figs(y_val), 
            (x_val, y_val), 
            textcoords="offset points",
            xytext=(0, 10) if pu == "4" else (0, -15),
            ha="center", 
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
        )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_grid_sizes_step_graph(output_dir: str, y_axis: str, log: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    gpu_2x2_world_json = "step_gpu_1_agent_simple_non_empty_world_100_steps.json"
    gpu_4x4_world_json = "step_gpu_1_agent_simple_non_empty_world_4x4_100_steps.json"
    gpu_8x8_world_json = "step_gpu_1_agent_simple_non_empty_world_8x8_100_steps.json"

    gpu_2x2_world_json = os.path.join(input_dir, gpu_2x2_world_json)
    gpu_4x4_world_json = os.path.join(input_dir, gpu_4x4_world_json)
    gpu_8x8_world_json = os.path.join(input_dir, gpu_8x8_world_json)
    # gpu_1_agent_empty_world_json = os.path.join(input_dir, gpu_1_agent_empty_world_json)
    # gpu_2_agents_empty_world_json = os.path.join(input_dir, gpu_2_agents_empty_world_json)
    # gpu_4_agents_empty_world_json = os.path.join(input_dir, gpu_4_agents_empty_world_json)

    with open(gpu_2x2_world_json, "r") as file:
        gpu_2x2_world_data = json.load(file)
    with open(gpu_4x4_world_json, "r") as file:
        gpu_4x4_world_data = json.load(file)
    with open(gpu_8x8_world_json, "r") as file:
        gpu_8x8_world_data = json.load(file)

    gpu_2x2_world_results = gpu_2x2_world_data['benchmark_results']
    gpu_4x4_world_results = gpu_4x4_world_data['benchmark_results']
    gpu_8x8_world_results = gpu_8x8_world_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    _, gpu_2x2_world_metrics_list = list(gpu_2x2_world_results.items())[0]
    _, gpu_4x4_world_metrics_list = list(gpu_4x4_world_results.items())[0]
    scenario_name, gpu_8x8_world_metrics_list = list(gpu_8x8_world_results.items())[0]

    gpu_2x2_world_batch_sizes = [m['batch_size'] for m in gpu_2x2_world_metrics_list]
    gpu_4x4_world_batch_sizes = [m['batch_size'] for m in gpu_4x4_world_metrics_list]
    gpu_8x8_world_batch_sizes = [m['batch_size'] for m in gpu_8x8_world_metrics_list]

    if y_axis == "time":
      gpu_2x2_world_steps = [m['total_time'] for m in gpu_2x2_world_metrics_list]
      gpu_4x4_world_steps = [m['total_time'] for m in gpu_4x4_world_metrics_list]
      gpu_8x8_world_steps = [m['total_time'] for m in gpu_8x8_world_metrics_list]
    elif y_axis == "steps":
      gpu_2x2_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_2x2_world_metrics_list]
      gpu_4x4_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_4x4_world_metrics_list]
      gpu_8x8_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_8x8_world_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    plt.plot(gpu_2x2_world_batch_sizes, gpu_2x2_world_steps, color="red", marker="x", label="2x2 world")
    plt.plot(gpu_4x4_world_batch_sizes, gpu_4x4_world_steps, color="green", marker="o", label="4x4 world")
    plt.plot(gpu_8x8_world_batch_sizes, gpu_8x8_world_steps, color="blue", marker="s", label="8x8 world")

    plt.legend(loc='upper left')

    # special_points = [(1, "1"), (1, "2"), (1, "4"), (2**11, "1"), (2**11, "2"), (2**8, "4"), (2**14, "1"), (2**13, "2"), (2**13, "4")]
    special_points = []

    # # Special points
    # for x_val, pu in special_points:
    #     if pu == "1":
    #         y_val = gpu_1_agent_non_empty_world_steps[gpu_1_agent_non_empty_world_batch_sizes.index(x_val)]
    #         color = "red"
    #     elif pu == "2":
    #         y_val = gpu_2_agents_non_empty_world_steps[gpu_2_agents_non_empty_world_batch_sizes.index(x_val)]
    #         color = "green"
    #     elif pu == "4":
    #         y_val = gpu_4_agents_non_empty_world_steps[gpu_4_agents_non_empty_world_batch_sizes.index(x_val)]
    #         color = "blue"
    #     else:
    #         raise ValueError(f"Invalid value for pu: {pu}")

    #     plt.annotate(
    #         round_to_sig_figs(y_val), 
    #         (x_val, y_val), 
    #         textcoords="offset points",
    #         xytext=(0, 10) if pu == "4" else (0, -15),
    #         ha="center", 
    #         fontsize=8,
    #         color=color,
    #         bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
    #     )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
   
def plot_cpu_gpu_non_empty_world_render_frame(output_dir: str, y_axis: str, log: bool, agent_count: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "scripts/benchmarking/graph_gen/jsons"
    # gpu_empty_world_json = f"render_frame_gpu_{agent_count}_agents_empty_world_100_steps.json"
    # cpu_empty_world_json = f"render_frame_cpu_{agent_count}_agents_empty_world_100_steps.json"
    gpu_non_empty_world_json = f"render_frame_gpu_{agent_count}_agents_non_empty_world_100_steps.json"
    cpu_non_empty_world_json = f"render_frame_cpu_{agent_count}_agents_non_empty_world_100_steps.json"

    # cpu_empty_world_json = os.path.join(input_dir, cpu_empty_world_json)
    cpu_non_empty_world_json = os.path.join(input_dir, cpu_non_empty_world_json)
    cpu_non_empty_world_json = linux_to_windows_path(cpu_non_empty_world_json)
    # gpu_empty_world_json = os.path.join(input_dir, gpu_empty_world_json)
    gpu_non_empty_world_json = os.path.join(input_dir, gpu_non_empty_world_json)
    gpu_non_empty_world_json = linux_to_windows_path(gpu_non_empty_world_json)

    # with open(cpu_empty_world_json, "r") as file:
    #   cpu_empty_world_data = json.load(file)
    with open(cpu_non_empty_world_json, "r") as file:
      cpu_non_empty_world_data = json.load(file)
    # with open(gpu_empty_world_json, "r") as file:
    #   gpu_empty_world_data = json.load(file)
    with open(gpu_non_empty_world_json, "r") as file:
      gpu_non_empty_world_data = json.load(file)

    # cpu_empty_world_results = cpu_empty_world_data['benchmark_results']
    cpu_non_empty_world_results = cpu_non_empty_world_data['benchmark_results']
    # gpu_empty_world_results = gpu_empty_world_data['benchmark_results']
    gpu_non_empty_world_results = gpu_non_empty_world_data['benchmark_results']

    # Assume that there is only one scenario in both json files
    # _, cpu_empty_world_metrics_list = list(cpu_empty_world_results.items())[0]
    _, cpu_non_empty_world_metrics_list = list(cpu_non_empty_world_results.items())[0]
    # _, gpu_empty_world_metrics_list = list(gpu_empty_world_results.items())[0]
    scenario_name, gpu_non_empty_world_metrics_list = list(gpu_non_empty_world_results.items())[0]

    # cpu_empty_world_batch_sizes = [m['batch_size'] for m in cpu_empty_world_metrics_list]
    cpu_non_empty_world_batch_sizes = [m['batch_size'] for m in cpu_non_empty_world_metrics_list]
    # gpu_empty_world_batch_sizes = [m['batch_size'] for m in gpu_empty_world_metrics_list]
    gpu_non_empty_world_batch_sizes = [m['batch_size'] for m in gpu_non_empty_world_metrics_list]

    if y_axis == "time":
      # cpu_empty_world_steps = [m['total_time'] for m in cpu_empty_world_metrics_list]
      cpu_non_empty_world_steps = [m['total_time'] for m in cpu_non_empty_world_metrics_list]
      # gpu_empty_world_steps = [m['total_time'] for m in gpu_empty_world_metrics_list]
      gpu_non_empty_world_steps = [m['total_time'] for m in gpu_non_empty_world_metrics_list]
    elif y_axis == "steps":
      # cpu_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in cpu_empty_world_metrics_list]
      cpu_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in cpu_non_empty_world_metrics_list]
      # gpu_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_empty_world_metrics_list]
      gpu_non_empty_world_steps = [m['num_steps'] * m['batch_size'] / m['total_time'] for m in gpu_non_empty_world_metrics_list]
    else:
      raise ValueError(f"Invalid value for y_axis: {y_axis}")

    plt.figure()
    # plt.plot(cpu_empty_world_batch_sizes, cpu_empty_world_steps, color="blue", marker="x", linestyle=':', label="CPU (empty world)")
    plt.plot(cpu_non_empty_world_batch_sizes, cpu_non_empty_world_steps, color="blue", marker="o", label="CPU (non-empty world)")
    # plt.plot(gpu_empty_world_batch_sizes, gpu_empty_world_steps, color="green", marker="x", linestyle=':', label="GPU (empty world)")
    plt.plot(gpu_non_empty_world_batch_sizes, gpu_non_empty_world_steps, color="green", marker="o", label="GPU (non-empty world)")

    legend_elements = [
      mpatches.Patch(color='blue', label='CPU'),
      mpatches.Patch(color='green', label='GPU'),
      Line2D([0], [0], color='black', lw=1, marker='x', linestyle=':', label='empty world (baseline)'),
      Line2D([0], [0], color='black', lw=1, marker='o', label='non-empty (2x2) world'),
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    # special_points = [(1, "gpu"), (2**9, "cpu"), (2**9, "gpu"), (2048, "gpu"), (16384, "gpu")]
    special_points = [(1, "cpu"), (1, "gpu"), (2**10, "cpu"), (2**10, "gpu"), (16384, "gpu")]
    # Special points
    for x_val, pu in special_points:
        if pu == "cpu":
            y_val = cpu_non_empty_world_steps[cpu_non_empty_world_batch_sizes.index(x_val)]
            opp_y_val = gpu_non_empty_world_steps[gpu_non_empty_world_batch_sizes.index(x_val)]
            color = "blue"
        elif pu == "gpu":
            y_val = gpu_non_empty_world_steps[gpu_non_empty_world_batch_sizes.index(x_val)]
            if (x_val not in cpu_non_empty_world_batch_sizes):
               opp_y_val = 0
            else:
              opp_y_val = cpu_non_empty_world_steps[cpu_non_empty_world_batch_sizes.index(x_val)]
            color = "green"
        else:
            raise ValueError(f"Invalid value for pu: {pu}")

        plt.annotate(
            round_to_sig_figs(y_val), 
            (x_val, y_val), 
            textcoords="offset points",
            xytext=(0, 10) if y_val >= opp_y_val else (0, -15),
            ha="center", 
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="white")
        )

    if log:
        plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    if (y_axis == "time"):
      plt.ylabel("Total time (in s)")
      plt.title(f"Total time (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
          output_dir, f"{scenario_name}_total-time.png"
      )
    else:
      plt.ylabel(f"Total steps per second")
      plt.title(f"Total Steps per Second (100 steps) vs Batch Size ({scenario_name})")
      plot_path = os.path.join(
        output_dir, f"{scenario_name}_total-steps-per-sec.png"
      )

    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
  # plot_simple_step_graph(
  #   "scripts/benchmarking/graph_gen/graphs", 
  #   "scripts/benchmarking/graph_gen/jsons/step_cpu_1_agent_simple_100_steps.json", 
  #   "scripts/benchmarking/graph_gen/jsons/step_gpu_1_agent_simple_100_steps.json", 
  #   [(1, "cpu"), (1, "gpu"), (2**15, "cpu"), (2**15, "gpu"), (2**20, "gpu")],
  #   "steps",
  #   True
  # )
  # plot_simple_complex_step_graph(
  #   "scripts/benchmarking/graph_gen/graphs", 
  #   True
  # )
  plot_cpu_gpu_non_empty_world_render_frame(
    "scripts/benchmarking/graph_gen/graphs", 
    "Batched steps per second",
    True,
    1
  )
