"""Hardware / runtime inspection. Never assumes a GPU exists; every probe degrades gracefully."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

GPU_QUERY = "name,memory.total,memory.used,driver_version,utilization.gpu,temperature.gpu"


def _ttl_cache(seconds: float) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Cache a zero-argument probe for a few seconds so live stats stay fresh but cheap."""

    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        state: dict[str, Any] = {}

        @wraps(fn)
        def wrapper() -> Any:
            if "value" not in state or time.monotonic() - state["at"] > seconds:
                state.update(value=fn(), at=time.monotonic())
            return state["value"]

        wrapper.cache_clear = state.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _number(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:  # "[N/A]" on some laptop GPUs
        return None


@_ttl_cache(seconds=3.0)
def detect_nvidia_gpus() -> list[dict[str, object]]:
    """Return NVIDIA GPUs reported by ``nvidia-smi`` (empty list if none or no driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            name, total, used, driver, util, temp = parts
            gpus.append(
                {
                    "name": name,
                    "memory_total_mib": _number(total),
                    "memory_used_mib": _number(used),
                    "driver": driver,
                    "utilization_pct": _number(util),
                    "temperature_c": _number(temp),
                }
            )
    return gpus


def nvidia_smi_diagnostic() -> str:
    """Human-readable reason when no GPU is reported (e.g. driver/library version mismatch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found (no NVIDIA driver installed)"
    try:
        proc = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"nvidia-smi failed: {type(exc).__name__}"
    output = (proc.stdout or proc.stderr).strip().splitlines()
    return output[0] if output else f"nvidia-smi exited with {proc.returncode}"


def summarize_environment(project_root: Path) -> dict[str, object]:
    usage = shutil.disk_usage(project_root)
    mem_total = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith(("MemTotal", "MemAvailable")):
                key, value = line.split(":")
                mem_total = mem_total or {}
                mem_total[key] = f"{int(value.split()[0]) // 1024} MiB"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": mem_total,
        "disk_free_gib": round(usage.free / 2**30, 1),
        "nvidia_gpus": detect_nvidia_gpus(),
        "nvidia_smi": nvidia_smi_diagnostic(),
        "uv": shutil.which("uv"),
        "ollama": shutil.which("ollama"),
    }
