from __future__ import annotations

import platform


def collect() -> dict:
    import psutil

    from .gpu_support import evaluate, query, torch_arches

    result = {"platform": platform.platform(), "python": platform.python_version(),
              "ram_gib": round(psutil.virtual_memory().total / 1024**3, 1), "cuda": False}
    installed = query()
    device = installed[0] if installed else {}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cuda"] = torch.cuda.is_available()
        if result["cuda"]:
            properties = torch.cuda.get_device_properties(0)
            result["gpu"] = torch.cuda.get_device_name(0)
            result["vram_gib"] = round(properties.total_memory / 1024**3, 1)
            result["compute_capability"] = list(torch.cuda.get_device_capability(0))
            result["driver"] = device.get("driver") and f"{device['driver'][0]}.{device['driver'][1]:02d}"
            # Judge the device against the architectures of the installed build,
            # so a replaced or downgraded wheel is reported instead of failing later.
            result["support"] = evaluate(
                gpu=result["gpu"], driver=device.get("driver"),
                compute=tuple(result["compute_capability"]), vram_gib=result["vram_gib"],
                arches=torch_arches(torch))
            x = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
            value = (x @ x.T).sum().item()
            result["cuda_matmul_verified"] = value > 0
            torch.cuda.synchronize()
        else:
            result["support"] = evaluate(gpu=device.get("name"), driver=device.get("driver"),
                                         compute=device.get("compute"), vram_gib=device.get("vram_gib"))
    except Exception as error:
        result["error"] = str(error)
    return result
