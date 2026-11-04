"""NVIDIA requirements of the pinned runtime stack, without importing Torch.

The installer preflight runs before Torch is downloaded; Help > Check System
runs with Torch loaded. Both judge a PC through this module, so the hardware
rules cannot drift apart from the wheels in installer/runtime-lock.json.

The pinned torch 2.7.1+cu128 and 2.8.0+cu128 wheels carry kernels for the
architectures in WHEEL_ARCHES. A device runs them when its compute capability
matches a compiled architecture or is a later minor revision of the same major
version (CUDA binary compatibility). That covers RTX 20 (7.5), 30 (8.6),
40 (8.9, running the 8.6 kernels) and 50 (12.0) series cards.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CUDA_RUNTIME = "12.8"
# CUDA 12.x minor-version compatibility on Windows, required by every GPU.
WINDOWS_DRIVER_MIN = (527, 41)
# Blackwell (RTX 50) support starts with the 570 driver branch.
BLACKWELL_MAJOR = 12
BLACKWELL_DRIVER_MIN = (570, 0)
BLACKWELL_DRIVER_TEXT = "570 or newer"
WINDOWS_DRIVER_TEXT = "527.41"
# Kernels compiled into the pinned wheels.
WHEEL_ARCHES = ((5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0))
# Measured peak allocated VRAM is 11.3 GiB (ExMR), 14.9 GiB (FlowHMR) and
# 5.1 GiB (GVHMR); low-memory mode lowers batch sizes, not model weights.
VRAM_RECOMMENDED_GIB = 12.0
VRAM_MINIMUM_GIB = 6.0


def parse_version(text):
    """Parse '617.42' or '12.0' into a comparable tuple, or None."""
    try:
        parts = str(text).strip().split(".")
        return int(parts[0]), int(parts[1])
    except (AttributeError, ValueError, IndexError):
        return None


def format_driver(value):
    """Render a driver version the way NVIDIA writes it (546 -> 546.01)."""
    return None if not value else f"{value[0]}.{value[1]:02d}"


def driver_required(compute):
    """Minimum Windows driver for the pinned runtime on this device."""
    if compute and compute[0] >= BLACKWELL_MAJOR:
        return BLACKWELL_DRIVER_MIN, BLACKWELL_DRIVER_TEXT
    return WINDOWS_DRIVER_MIN, WINDOWS_DRIVER_TEXT


def arch_supported(compute, arches=WHEEL_ARCHES):
    """Whether a compiled kernel can run on a device of this capability."""
    if not compute:
        return False
    major, minor = compute
    return any(compiled[0] == major and compiled[1] <= minor for compiled in arches)


def describe_arches(arches=WHEEL_ARCHES):
    return ", ".join(f"{major}.{minor}" for major, minor in arches)


def _nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    return str(candidate) if candidate.is_file() else None


def query(timeout=10.0):
    """Read installed NVIDIA GPUs, newest driver first. Empty when unavailable.

    ``compute_cap`` is missing from older drivers, so the query is retried
    without it; the caller then treats the capability as unknown.
    """
    executable = _nvidia_smi()
    if not executable:
        return []
    fields = ("name", "driver_version", "memory.total", "compute_cap")
    for columns in (fields, fields[:-1]):
        command = [executable, "--query-gpu=" + ",".join(columns), "--format=csv,noheader,nounits"]
        try:
            finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                      check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            continue
        if finished.returncode != 0 or not finished.stdout.strip():
            continue
        rows = []
        for line in finished.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < len(columns):
                continue
            fields_of_line = parts[-(len(columns) - 1):]
            try:
                memory = float(fields_of_line[1])
            except ValueError:
                memory = None
            rows.append(dict(
                name=parts[0],
                driver=parse_version(fields_of_line[0]),
                vram_gib=round(memory / 1024, 1) if memory else None,
                compute=parse_version(fields_of_line[2]) if len(columns) == 4 else None,
            ))
        return rows
    return []


def evaluate(*, gpu=None, driver=None, compute=None, vram_gib=None, arches=WHEEL_ARCHES):
    """Judge one device against the pinned wheels.

    ``status`` is ``ok``, ``warning`` (usable with limits) or ``unsupported``
    (the pinned kernels or driver cannot run on this device).
    """
    problems, warnings = [], []
    required, required_text = driver_required(compute)
    if not gpu:
        problems.append("No NVIDIA CUDA GPU was detected. RGB MoCap needs an NVIDIA GPU and its current driver.")
    else:
        if not compute:
            warnings.append("The driver did not report a compute capability. Update the NVIDIA driver to the current release.")
        elif not arch_supported(compute, arches):
            problems.append("Compute capability %d.%d is not covered by the installed PyTorch kernels (%s). "
                            "Use an RTX 20 series or newer GPU." % (compute[0], compute[1], describe_arches(arches)))
        if not driver:
            warnings.append("The NVIDIA driver version could not be read. Install the current driver from nvidia.com.")
        elif driver < required:
            if required == BLACKWELL_DRIVER_MIN:
                problems.append("NVIDIA driver %s is too old for this RTX 50 series GPU. Install driver %s, then retry."
                                % (format_driver(driver), BLACKWELL_DRIVER_TEXT))
            else:
                problems.append("NVIDIA driver %s is older than %s, required by the pinned CUDA %s kernels. "
                                "Update the driver from nvidia.com, then retry."
                                % (format_driver(driver), WINDOWS_DRIVER_TEXT, CUDA_RUNTIME))
        if vram_gib is not None:
            if vram_gib < VRAM_MINIMUM_GIB:
                warnings.append("Only %.1f GiB of VRAM is available. Short clips in Low Memory mode may run, "
                                "but the supported models need about 6 GiB or more." % vram_gib)
            elif vram_gib < VRAM_RECOMMENDED_GIB:
                warnings.append("%.1f GiB of VRAM is below the %.0f GiB recommended for every model. "
                                "Use Low Memory mode and shorter clips." % (vram_gib, VRAM_RECOMMENDED_GIB))
    status = "unsupported" if problems else ("warning" if warnings else "ok")
    return dict(
        status=status, problems=problems, warnings=warnings, gpu=gpu,
        driver=format_driver(driver) if driver else None,
        driver_required=required_text,
        compute_capability=list(compute) if compute else None,
        vram_gib=vram_gib, cuda_runtime=CUDA_RUNTIME,
        supported_arches=list(describe_arches(arches).split(", ")),
    )


def torch_arches(torch):
    """Compiled architectures of the loaded Torch build, as comparable tuples."""
    try:
        names = torch.cuda.get_arch_list()
    except Exception:
        return WHEEL_ARCHES
    parsed = [parse_version(name.replace("sm_", "").replace("compute_", "")) for name in names]
    found = tuple(value for value in parsed if value)
    return found or WHEEL_ARCHES
