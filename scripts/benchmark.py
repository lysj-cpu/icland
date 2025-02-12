"""Benchmarking script to generate a report with system information and benchmark results.

This module provides functions to get system information such as CPU, memory,
storage, OS, Python, and GPU, and generates a benchmark report with plots.
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import cpuinfo
import GPUtil
import matplotlib.pyplot as plt  # For plotting
import psutil

# Import the benchmark function (ensure benchmark_functions is in your PYTHONPATH)
from benchmark_functions import benchmark_batch_size
from pylatex import Document, NoEscape


@dataclass
class BenchmarkMetrics:
    """Dataclass for benchmark metrics."""

    batch_size: int
    batched_steps_per_second: float
    max_memory_usage_mb: float
    max_cpu_usage_percent: float
    max_gpu_usage_percent: list[float]
    max_gpu_memory_usage_mb: list[float]


# --------------------------------------------------------------------------------------
# Benchmarking scenarios
# --------------------------------------------------------------------------------------
# NOTE: The original parameters used "2^i" (bitwise XOR) so we use "2**i" for powers.
# Here we use powers of 2 from 2^1 to 2^12. (For testing purposes, the range can be adjusted.)
@dataclass
class BenchmarkScenario:
    """Dataclass for benchmarking scenarios."""

    description: str
    function: Any
    parameters: list[int]


BENCHMARKING_SCENARIOS: dict[str, BenchmarkScenario] = {
    "batched_step_performance": BenchmarkScenario(
        description="Batched step performance",
        function=benchmark_batch_size,
        parameters=[2**i for i in range(0, 13)],
    )
}


# --------------------------------------------------------------------------------------
# System Information Functions
# --------------------------------------------------------------------------------------
def get_cpu_info() -> dict[str, str]:
    """Get CPU information."""
    info = cpuinfo.get_cpu_info()
    return {
        "Model": info.get("brand_raw", "Unknown"),
        "Architecture": platform.machine(),
        "Cores (Physical/Logical)": f"{psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}",
        "Base Frequency": f"{info.get('hz_actual_friendly', 'Unknown')}",
        "L2 Cache": f"{info.get('l2_cache_size', 'Unknown')} bytes",
        "L3 Cache": f"{info.get('l3_cache_size', 'Unknown')} bytes",
    }


def get_memory_info() -> dict[str, str]:
    """Get memory information."""
    mem = psutil.virtual_memory()
    return {
        "Total RAM": f"{mem.total / (1024**3):.2f} GB",
        "Available RAM": f"{mem.available / (1024**3):.2f} GB",
    }


def get_storage_info() -> dict[str, str]:
    """Get storage information."""
    disk = shutil.disk_usage("/")
    return {
        "Total Storage": f"{disk.total / (1024**3):.2f} GB",
        "Used Storage": f"{disk.used / (1024**3):.2f} GB",
        "Free Storage": f"{disk.free / (1024**3):.2f} GB",
        "Filesystem Type": os.uname().sysname
        if hasattr(os, "uname")
        else platform.system(),
    }


def get_os_info() -> dict[str, str]:
    """Get OS information."""
    return {
        "OS": platform.system(),
        "Version": platform.version(),
        "Release": platform.release(),
        "Kernel": platform.uname().version,
        "Uptime": f"{time.time() - psutil.boot_time():.0f} seconds",
    }


def get_python_info() -> dict[str, str]:
    """Get Python information."""
    return {
        "Python Version": sys.version,
        "Interpreter": platform.python_implementation(),
        "Virtual Env": sys.prefix,
        "Installed Packages": subprocess.getoutput("pip freeze")[:500]
        + "...",  # Limiting output size
    }


def get_gpu_info() -> dict[str, dict[str, str]]:
    """Get GPU information."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return {"GPU": {"Model": "No dedicated GPU found"}}
    return {
        f"GPU {i + 1}": {
            "Model": gpu.name,
            "VRAM": f"{gpu.memoryTotal} MB",
            "Temperature": f"{gpu.temperature} °C",
            "Driver": gpu.driver,
        }
        for i, gpu in enumerate(gpus)
    }


@dataclass
class SystemInfo:
    """Dataclass to store system information."""

    cpu_information: dict[str, str]
    memory_information: dict[str, str]
    storage_information: dict[str, str]
    os_information: dict[str, str]
    gpu_information: dict[str, dict[str, str]]
    python_environment: dict[str, str]


def gather_system_info() -> SystemInfo:
    """Gather system information."""
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    storage_info = get_storage_info()
    os_info = get_os_info()
    gpu_info = get_gpu_info()
    python_info = get_python_info()

    system_info = SystemInfo(
        cpu_information=cpu_info,
        memory_information=memory_info,
        storage_information=storage_info,
        os_information=os_info,
        gpu_information={},
        python_environment={
            "Python Version": python_info["Python Version"],
            "Interpreter": python_info["Interpreter"],
            "Virtual Env": python_info["Virtual Env"],
            "Installed Packages": f"{len(python_info['Installed Packages'])} packages",
        },
    )

    for gpu_id, gpu_data in gpu_info.items():
        system_info.gpu_information[f"GPU {gpu_id}"] = {
            "Model": gpu_data.get("Model", "No dedicated GPU found"),
            "VRAM": gpu_data.get("VRAM", "N/A"),
            "Temperature": gpu_data.get("Temperature", "N/A"),
            "Driver": gpu_data.get("Driver", "N/A"),
        }

    return system_info


# --------------------------------------------------------------------------------------
# Helper Functions for LaTeX and Plotting
# --------------------------------------------------------------------------------------
def sanitize_for_latex(value: str) -> str:
    """Sanitize a string for LaTeX."""
    special_chars = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for char, replacement in special_chars.items():
        value = value.replace(char, replacement)
    return value


def generate_latex_table(title: str, data: dict[str, Any]) -> str:
    """Generate a LaTeX table from a dictionary."""
    table = f"\\subsection*{{{sanitize_for_latex(title)}}}\n"
    table += "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}p{0.25\\textwidth}p{0.75\\textwidth}@{}}\n"
    table += "    \\toprule\n"
    table += "    \\textbf{Attribute} & \\textbf{Value} \\\\\n"
    table += "    \\midrule\n"

    keys_list = list(data.keys())
    for i, key_ in enumerate(keys_list):
        val = data[key_]
        sanitized_key = sanitize_for_latex(str(key_))

        if isinstance(val, dict):
            # Nested dict (e.g., GPU 0 -> {Model, VRAM, ...})
            sub_keys = list(val.keys())
            for j, sub_key in enumerate(sub_keys):
                sanitized_sub_key = sanitize_for_latex(str(sub_key))
                sanitized_sub_val = sanitize_for_latex(str(val[sub_key]))
                table += (
                    f"    {sanitized_sub_key} & \\texttt{{{sanitized_sub_val}}} \\\\\n"
                )
            if i < len(keys_list) - 1:
                table += "    \\midrule\n"
        else:
            sanitized_val = sanitize_for_latex(str(val))
            table += f"    {sanitized_key} & \\texttt{{{sanitized_val}}} \\\\\n"

    table += "    \\bottomrule\n"
    table += "\\end{tabular*}\n"
    return table


def run_benchmarks() -> dict[str, list[BenchmarkMetrics]]:
    """The BenchmarkMetrics list contains the metrics for each batch size in the scenario.

    Run benchmark scenarios and return a dictionary where each key is the
    scenario name and the value is a list of BenchmarkMetrics.
    """
    results = {}
    for scenario_name, scenario_data in BENCHMARKING_SCENARIOS.items():
        benchmark_fn = scenario_data.function
        parameters = scenario_data.parameters
        scenario_results = []
        print(f"Running scenario: {scenario_name}")
        for param in parameters:
            print(f"  Benchmarking with batch size = {param} ...")
            metrics = benchmark_fn(param)
            scenario_results.append(metrics)
        results[scenario_name] = scenario_results
    return results


def plot_benchmark_results(
    scenario_name: str, metrics_list: list[BenchmarkMetrics], output_dir: str
) -> dict[str, str]:
    """Generate plots for each metric against batch size and save them to output_dir.

    Returns a dictionary mapping plot descriptions to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    # Prepare common x-axis data (batch sizes)
    batch_sizes = [m.batch_size for m in metrics_list]

    # 1. Batched Steps Per Second
    steps = [m.batched_steps_per_second for m in metrics_list]
    plt.figure()
    plt.plot(batch_sizes, steps, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Batched Steps per Second")
    plt.title(f"Batched Steps per Second vs Batch Size ({scenario_name})")
    plt.grid(True)
    plot_path = os.path.join(
        output_dir, f"{scenario_name}_batched_steps_per_second.png"
    )
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    plots["Batched Steps per Second"] = plot_path

    # 2. Max Memory Usage (MB)
    mem_usage = [m.max_memory_usage_mb for m in metrics_list]
    plt.figure()
    plt.plot(batch_sizes, mem_usage, marker="o", color="orange")
    plt.xlabel("Batch Size")
    plt.ylabel("Max Memory Usage (MB)")
    plt.title(f"Max Memory Usage vs Batch Size ({scenario_name})")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{scenario_name}_max_memory_usage_mb.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    plots["Max Memory Usage (MB)"] = plot_path

    # 3. Max CPU Usage Percent
    cpu_usage = [m.max_cpu_usage_percent for m in metrics_list]
    plt.figure()
    plt.plot(batch_sizes, cpu_usage, marker="o", color="green")
    plt.xlabel("Batch Size")
    plt.ylabel("Max CPU Usage (%)")
    plt.title(f"Max CPU Usage vs Batch Size ({scenario_name})")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{scenario_name}_max_cpu_usage_percent.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    plots["Max CPU Usage (%)"] = plot_path

    # 4. Max GPU Usage Percent (if available)
    if metrics_list and metrics_list[0].max_gpu_usage_percent:
        n_gpus = len(metrics_list[0].max_gpu_usage_percent)
        plt.figure()
        for i in range(n_gpus):
            gpu_usage = [m.max_gpu_usage_percent[i] for m in metrics_list]
            plt.plot(batch_sizes, gpu_usage, marker="o", label=f"GPU {i + 1}")
        plt.xlabel("Batch Size")
        plt.ylabel("Max GPU Usage (%)")
        plt.title(f"Max GPU Usage vs Batch Size ({scenario_name})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(
            output_dir, f"{scenario_name}_max_gpu_usage_percent.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        plots["Max GPU Usage (%)"] = plot_path

    # 5. Max GPU Memory Usage (MB) (if available)
    if metrics_list and metrics_list[0].max_gpu_memory_usage_mb:
        n_gpus = len(metrics_list[0].max_gpu_memory_usage_mb)
        plt.figure()
        for i in range(n_gpus):
            gpu_mem = [m.max_gpu_memory_usage_mb[i] for m in metrics_list]
            plt.plot(batch_sizes, gpu_mem, marker="o", label=f"GPU {i + 1}")
        plt.xlabel("Batch Size")
        plt.ylabel("Max GPU Memory Usage (MB)")
        plt.title(f"Max GPU Memory Usage vs Batch Size ({scenario_name})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(
            output_dir, f"{scenario_name}_max_gpu_memory_usage_mb.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        plots["Max GPU Memory Usage (MB)"] = plot_path

    return plots


# --------------------------------------------------------------------------------------
# Main Report Creation Function
# --------------------------------------------------------------------------------------
def create_report(output_pdf: str = "scripts/benchmark_output/report") -> None:
    """Create a PDF report with system information, benchmark results, and plots."""
    # 1) Gather System Info
    sys_info = gather_system_info()

    # 2) Run Benchmark Scenarios
    bench_results = run_benchmarks()

    # 3) Build LaTeX document
    doc = Document()
    doc.preamble.append(NoEscape(r"\usepackage[a4paper,margin=0.5in]{geometry}"))
    doc.preamble.append(NoEscape(r"\usepackage{graphicx}"))
    doc.preamble.append(NoEscape(r"\usepackage{array}"))
    doc.preamble.append(NoEscape(r"\usepackage{booktabs}"))
    doc.preamble.append(NoEscape(r"\usepackage{hyperref}"))
    doc.preamble.append(NoEscape(r"\usepackage{multirow}"))
    doc.preamble.append(NoEscape(r"\usepackage{makecell}"))
    doc.preamble.append(NoEscape(r"\usepackage{svg}"))
    doc.preamble.append(
        NoEscape(r"\title{\vspace{-2cm}ICLand Benchmark Report\vspace{-1cm}}")
    )
    doc.preamble.append(NoEscape(r"\date{}"))
    doc.append(NoEscape(r"\maketitle"))

    # 4) Device Information Section
    doc.append(NoEscape(r"\section*{Device Information}"))
    for section_title, section_data in sys_info.__dict__.items():
        table_tex = generate_latex_table(section_title, section_data)
        doc.append(NoEscape(table_tex))

    # 5) Benchmarking Results Section
    # Define a directory for saving plot images relative to the output PDF location.
    output_dir = os.path.join(os.path.dirname(output_pdf), "plots")
    os.makedirs(output_dir, exist_ok=True)

    for scenario_name, metrics_list in bench_results.items():
        # Insert a section header for the scenario.
        doc.append(
            NoEscape(
                r"\section*{Benchmarking Results: "
                + sanitize_for_latex(scenario_name)
                + "}"
            )
        )

        # Generate plots for the current scenario.
        plots_dict = plot_benchmark_results(scenario_name, metrics_list, output_dir)
        plot_items = list(plots_dict.items())

        # If there is at least one plot, print the first one full-width.
        if plot_items:
            first_plot_title, first_plot_path = plot_items[0]
            relative_plot_path = first_plot_path.replace("\\", "/").replace(
                "scripts/benchmark_output/", ""
            )
            doc.append(NoEscape(r"\begin{figure}[h]"))
            doc.append(NoEscape(r"\centering"))
            doc.append(
                NoEscape(
                    r"\includegraphics[width=\textwidth]{" + relative_plot_path + "}"
                )
            )
            doc.append(
                NoEscape(r"\caption*{" + sanitize_for_latex(first_plot_title) + "}")
            )
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\newpage"))

        # Remove the first plot from the list and group the rest in pairs.
        remaining_plots = plot_items[1:]
        for i in range(0, len(remaining_plots), 2):
            doc.append(NoEscape(r"\begin{figure}[h]"))
            doc.append(NoEscape(r"\centering"))
            # First plot of the pair.
            plot_title, plot_path = remaining_plots[i]
            relative_plot_path = plot_path.replace("\\", "/").replace(
                "scripts/benchmark_output/", ""
            )
            doc.append(NoEscape(r"\begin{minipage}[b]{0.48\textwidth}"))
            doc.append(NoEscape(r"\centering"))
            doc.append(
                NoEscape(
                    r"\includegraphics[width=\textwidth]{" + relative_plot_path + "}"
                )
            )
            doc.append(NoEscape(r"\caption*{" + sanitize_for_latex(plot_title) + "}"))
            doc.append(NoEscape(r"\end{minipage}"))
            # Second plot of the pair (if it exists).
            if i + 1 < len(remaining_plots):
                doc.append(NoEscape(r"\hfill"))
                plot_title, plot_path = remaining_plots[i + 1]
                relative_plot_path = plot_path.replace("\\", "/").replace(
                    "scripts/benchmark_output/", ""
                )
                doc.append(NoEscape(r"\begin{minipage}[b]{0.48\textwidth}"))
                doc.append(NoEscape(r"\centering"))
                doc.append(
                    NoEscape(
                        r"\includegraphics[width=\textwidth]{"
                        + relative_plot_path
                        + "}"
                    )
                )
                doc.append(
                    NoEscape(r"\caption*{" + sanitize_for_latex(plot_title) + "}")
                )
                doc.append(NoEscape(r"\end{minipage}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\newpage"))

    # 6) Generate final PDF
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc.generate_pdf(filepath=output_pdf, clean_tex=False)
    print(f"PDF generated at: {output_pdf}.pdf")


if __name__ == "__main__":
    create_report()
