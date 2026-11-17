import platform
import psutil
import torch
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/telemetry", tags=["Telemetry"])

def _get_cpu_name() -> str:
    """Dynamically get CPU brand model name."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
    except Exception:
        pass
    proc = platform.processor() or "CPU"
    cores = psutil.cpu_count(logical=False) or 0
    threads = psutil.cpu_count(logical=True) or 0
    return f"{proc} ({cores}C/{threads}T)"

@router.get("")
def get_system_telemetry():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()

    gpu_available = torch.cuda.is_available()
    gpu_name = "None"
    gpu_vram_total = 0.0
    gpu_vram_alloc = 0.0
    device_count = 0

    if gpu_available:
        try:
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            gpu_vram_alloc = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        except Exception as e:
            gpu_name = f"Detected ({e})"

    return {
        "cpu": {
            "name": _get_cpu_name(),
            "usage_percent": cpu_percent
        },
        "ram": {
            "total_gb": round(ram.total / (1024**3), 1),
            "used_gb": round(ram.used / (1024**3), 1),
            "percent": ram.percent
        },
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "device_count": device_count,
            "vram_total_gb": gpu_vram_total,
            "vram_allocated_gb": gpu_vram_alloc
        }
    }
