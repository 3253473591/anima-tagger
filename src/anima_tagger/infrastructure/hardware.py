from __future__ import annotations

import psutil


def sample_hardware() -> dict[str, object]:
    memory = psutil.virtual_memory()
    result = {
        "memory": f"{memory.used / 2**30:.1f}/{memory.total / 2**30:.1f} GB",
        "memory_percent": int(memory.percent),
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        result.update(
            gpu=f"{utilization.gpu}%",
            gpu_percent=int(utilization.gpu),
            vram=f"{gpu_memory.used / 2**30:.1f}/{gpu_memory.total / 2**30:.1f} GB",
            vram_percent=int(gpu_memory.used * 100 / gpu_memory.total),
            temperature=f"{temperature} C",
            temperature_value=temperature,
        )
    except Exception:
        result.update(gpu="不可用", vram="不可用", temperature="不可用")
        result.update(gpu_percent=0, vram_percent=0, temperature_value=0)
    return result
