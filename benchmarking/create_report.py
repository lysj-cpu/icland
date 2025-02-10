# type: ignore
"""Create a PDF report with system information and benchmark results."""

import os

from pylatex import Document, NoEscape
from run_benchmarking import run_all_scenarios
from sys_info import (
    get_cpu_info,
    get_gpu_info,
    get_memory_info,
    get_os_info,
    get_python_info,
    get_storage_info,
)


# --------------------------------------------------------------------------------------
# 1) Gather System Info
# --------------------------------------------------------------------------------------
def gather_system_info():
    """Gather system information."""
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    storage_info = get_storage_info()
    os_info = get_os_info()
    gpu_info = get_gpu_info()
    python_info = get_python_info()

    # Pack into nested dict for table generation
    values = {
        "CPU Information": cpu_info,
        "Memory Information": memory_info,
        "Storage Information": storage_info,
        "OS Information": os_info,
        "GPU Information": {},
        "Python Environment": {
            "Python Version": python_info["Python Version"],
            "Interpreter": python_info["Interpreter"],
            "Virtual Env": python_info["Virtual Env"],
            "Installed Packages": f"{len(python_info['Installed Packages'])} packages",
        },
    }
    # Fill GPU info
    for gpu_id, gpu_data in gpu_info.items():
        values["GPU Information"][f"GPU {gpu_id}"] = {
            "Model": gpu_data.get("Model", "No dedicated GPU found"),
            "VRAM": gpu_data.get("VRAM", "N/A"),
            "Temperature": gpu_data.get("Temperature", "N/A"),
            "Driver": gpu_data.get("Driver", "N/A"),
        }
    return values


# --------------------------------------------------------------------------------------
# 2) Helper: Sanitize for LaTeX
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


# --------------------------------------------------------------------------------------
# 3) Helper: Generate LaTeX Table for System Info
# --------------------------------------------------------------------------------------
def generate_latex_table(title: str, data: dict) -> str:
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


# --------------------------------------------------------------------------------------
# 4) Main create_report function
# --------------------------------------------------------------------------------------
def create_report(output_pdf="benchmarking/output/benchmark_report"):
    """Create a PDF report with system information and benchmark results."""
    # 4.1) Gather System Info
    sys_info = gather_system_info()

    # 4.2) Run all scenario benchmarks
    scenario_results = run_all_scenarios()

    # 4.3) Build LaTeX content
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

    # 4.4) Device Information Section
    doc.append(NoEscape(r"\section*{Device Information}"))
    for section_title, section_data in sys_info.items():
        table_tex = generate_latex_table(section_title, section_data)
        doc.append(NoEscape(table_tex))

    # 4.5) Now for each scenario, add a new section to the PDF
    for scenario_name, results_dict in scenario_results.items():
        desc = results_dict.get("description", "")
        compile_time_s = results_dict.get("compile_time_s", 0.0)
        graphics = results_dict.get("graphics", {})

        doc.append(NoEscape(r"\clearpage"))
        doc.append(
            NoEscape(f"\\section*{{Scenario: {sanitize_for_latex(scenario_name)}}}")
        )
        doc.append(NoEscape(f"\\textit{{{sanitize_for_latex(desc)}}}\\\\"))
        doc.append(NoEscape(r"\\[6pt]"))

        # 4.5.1) Compile Time
        doc.append(NoEscape(f"\\textbf{{Compile Time}}: {compile_time_s:.3f} seconds."))
        doc.append(NoEscape(r"\\[12pt]"))

        # 4.5.2) Insert images if they exist
        # We can create small figure blocks or a single figure. Example:
        # We'll show each graphic from 'graphics' in a new minipage
        for graph_key, graph_path in graphics.items():
            if graph_path is None:
                continue
            # Clean up path for LaTeX
            abs_path = os.path.abspath(graph_path)
            doc.append(NoEscape(r"\begin{figure}[h!]"))
            doc.append(NoEscape(r"\centering"))
            doc.append(
                NoEscape(r"\includegraphics[width=0.8\textwidth]{%s}" % abs_path)
            )
            caption = f"{graph_key.replace('_', ' ').title()} for {scenario_name}"
            doc.append(NoEscape(f"\\caption{{{sanitize_for_latex(caption)}}}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\\[6pt]"))

    # 4.6) Generate final PDF
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc.generate_pdf(filepath=output_pdf, clean_tex=False)
    print(f"PDF generated at: {output_pdf}.pdf")


# --------------------------------------------------------------------------------------
# If you'd like to run from CLI (optional):
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    create_report()
