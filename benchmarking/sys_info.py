"""This module provides functions to get system information such as CPU, memory, storage, OS, Python, and GPU."""

import os
import platform
import shutil
import subprocess
import sys
import time

import cpuinfo
import GPUtil
import psutil


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
