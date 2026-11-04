"""Check this PC against the pinned runtime stack before install.cmd downloads it.

Standard library only: it runs from the private bootstrap Python, before Torch
or any runtime exists, so a machine that cannot run the app is reported in one
minute instead of after a multi-gigabyte install. ``mocap.gpu_support`` holds
the same rules Help > Check System applies once Torch is installed.

Exit codes: 0 = supported (warnings allowed), 2 = warnings only, 1 = unsupported.
Run ``tools/preflight.py --json`` for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.gpu_support import describe_arches, evaluate, query  # noqa: E402

# Longest installed path under a staged runtime (measured from runtime-lock
# staging layout: ".setup/staging/<id>/runtime/Lib/site-packages/...").
STAGING_PATH_BUDGET = 216
PATH_LIMIT = 260
PATH_WARN_AT = 250
# Model files are downloaded by Model Manager, not by install.cmd.
MODELS_GIB = 29


def windows_report():
    if sys.platform != "win32":
        return dict(ok=False, text=f"{platform.system()} is not supported; RGB MoCap runs on 64-bit Windows only.")
    bits = struct.calcsize("P") * 8
    machine = platform.machine() or "unknown"
    text = f"{platform.platform()} ({bits}-bit {machine})"
    if bits != 64:
        return dict(ok=False, text=text + " - a 64-bit Windows installation is required.")
    if machine.lower() not in ("amd64", "x86_64"):
        return dict(ok=False, text=text + " - x64 Windows is required.")
    return dict(ok=True, text=text)


def path_report(root):
    length = len(str(root))
    projected = length + STAGING_PATH_BUDGET
    if projected > PATH_LIMIT:
        return dict(ok=False, level="unsupported", text=(
            f"{root} ({length} characters). Installation paths reach about {projected} characters, over the "
            f"{PATH_LIMIT}-character Windows limit. Move the checkout to a short folder such as D:\\RGBMoCap and retry."))
    if projected > PATH_WARN_AT:
        return dict(ok=True, level="warning", text=(
            f"{root} ({length} characters). Installation paths reach about {projected} characters, close to the "
            f"{PATH_LIMIT}-character Windows limit. A shorter folder such as D:\\RGBMoCap is safer."))
    return dict(ok=True, level="ok", text=f"{root} ({length} characters)")


def disk_report(root, lock_path):
    try:
        lock = json.loads(lock_path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        lock = {}
    required = lock.get("fresh_free_bytes_required")
    probe = root if root.exists() else next((parent for parent in root.parents if parent.exists()), root)
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return dict(ok=True, level="warning", text="Free space could not be read.")
    if not required:
        return dict(ok=True, level="ok", text=f"{free / 1024 ** 3:.1f} GiB free")
    needed_gib = required / 1024 ** 3
    text = f"{free / 1024 ** 3:.1f} GiB free, {needed_gib:.1f} GiB needed for the runtimes"
    if free < required:
        return dict(ok=False, level="unsupported", text=text + f". Free at least {needed_gib:.0f} GiB and retry.")
    level = "ok" if free > required * 1.5 else "warning"
    return dict(ok=True, level=level, text=text + f"; model files add about {MODELS_GIB} GiB")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check this PC against the pinned RGB MoCap runtimes.")
    parser.add_argument("--root", type=Path, default=ROOT, help="installation folder to check")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    devices = query()
    device = devices[0] if devices else {}
    support = evaluate(gpu=device.get("name"), driver=device.get("driver"),
                       compute=device.get("compute"), vram_gib=device.get("vram_gib"))
    windows = windows_report()
    folder = path_report(root)
    disk = disk_report(root, ROOT / "installer/runtime-lock.json")

    blocking = [problem for problem in support["problems"]]
    if not windows["ok"]:
        blocking.append(windows["text"])
    if not folder["ok"]:
        blocking.append(folder["text"])
    if not disk["ok"]:
        blocking.append(disk["text"])
    warnings = list(support["warnings"])
    for row in (folder, disk):
        if row["ok"] and row["level"] == "warning":
            warnings.append(row["text"])
    status = "unsupported" if blocking else ("warning" if warnings else "ok")

    report = dict(status=status, windows=windows["text"], gpu=support, install_folder=folder["text"],
                  disk=disk["text"], blocking=blocking, warnings=warnings,
                  extra_gpus=[row.get("name") for row in devices[1:]])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        compute = support["compute_capability"]
        capability = f" (compute {compute[0]}.{compute[1]})" if compute else ""
        print("RGB MoCap installation check")
        print("  Windows       " + windows["text"])
        print("  GPU           " + (support["gpu"] or "none detected") + capability)
        if support["gpu"]:
            print(f"  Driver        {support['driver'] or 'unknown'} (needs {support['driver_required']})")
            vram = support["vram_gib"]
            print(f"  VRAM          {vram:.1f} GiB" if vram else "  VRAM          unknown")
        print("  PyTorch       kernels for compute " + describe_arches())
        print("  Install folder " + folder["text"])
        print("  Disk          " + disk["text"])
        if report["extra_gpus"]:
            print("  Other GPUs    " + ", ".join(report["extra_gpus"]) + " (the first GPU is used)")
        shown = {windows["text"], folder["text"], disk["text"]}
        for warning in warnings:
            print("  Warning       " + warning)
        for problem in blocking:
            if problem not in shown:
                print("  Problem       " + problem)
        print("  Result        " + {"ok": "OK", "warning": "OK with warnings", "unsupported": "NOT SUPPORTED"}[status])
        if blocking:
            print("  To install anyway for inspection, set RGBMOCAP_ALLOW_UNSUPPORTED_GPU=1 and rerun install.cmd.")
    return {"ok": 0, "warning": 2, "unsupported": 1}[status]


if __name__ == "__main__":
    raise SystemExit(main())
