#!/usr/bin/env python3
"""
Shared hardware-detection helpers used by the STT benchmark scripts (and
mirrored inline in llm/llm_benchmark.py, which was validated independently
and is kept as-is rather than refactored to import this mid-flight).

Import as: from hardware_info import get_hardware_info, classify_result_path
"""

import os
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Optional


def _read_proc_cpuinfo_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _cpu_core_counts() -> tuple:
    logical = os.cpu_count() or 1
    physical = logical
    try:
        with open("/proc/cpuinfo") as f:
            ids = set()
            phys_id = core_id = None
            for line in f:
                line = line.strip()
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if phys_id is not None and core_id is not None:
                        ids.add((phys_id, core_id))
            if ids:
                physical = len(ids)
    except Exception:
        pass
    return physical, logical


def _mem_info_gb() -> dict:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "swap_total_gb": round(sw.total / (1024 ** 3), 2),
            "swap_used_gb": round(sw.used / (1024 ** 3), 2),
        }
    except ImportError:
        pass
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip().split()[0])
        total = vals.get("MemTotal", 0)
        avail = vals.get("MemAvailable", 0)
        swap_total = vals.get("SwapTotal", 0)
        swap_free = vals.get("SwapFree", 0)
        return {
            "ram_total_gb": round(total / (1024 ** 2), 2),
            "ram_available_gb": round(avail / (1024 ** 2), 2),
            "swap_total_gb": round(swap_total / (1024 ** 2), 2),
            "swap_used_gb": round((swap_total - swap_free) / (1024 ** 2), 2),
        }
    except Exception:
        return {"ram_total_gb": None, "ram_available_gb": None,
                "swap_total_gb": None, "swap_used_gb": None}


def _detect_gpu() -> tuple:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().split(",", 1)
            mem_mb = float(re.sub(r"[^\d.]", "", mem) or 0)
            return name.strip(), round(mem_mb / 1024, 2) if mem_mb else None
    except Exception:
        pass
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=15)
        for line in out.stdout.splitlines():
            if re.search(r"VGA|3D controller|Display controller", line):
                return line.split(":", 2)[-1].strip(), None
    except Exception:
        pass
    return None, None


def _detect_device_class() -> tuple:
    dt_model_path = Path("/proc/device-tree/model")
    if dt_model_path.exists():
        try:
            model = dt_model_path.read_bytes().split(b"\x00")[0].decode(errors="ignore").strip()
            if model:
                return "edge_device", model
        except Exception:
            pass
    if platform.machine().lower() in ("aarch64", "armv7l", "arm64"):
        return "edge_device", f"unknown ARM board ({platform.machine()})"
    return "laptop_desktop", None


def _detect_power_source() -> str:
    supply_dir = Path("/sys/class/power_supply")
    if not supply_dir.exists():
        return "unknown"
    try:
        has_battery = False
        ac_online = False
        for entry in supply_dir.iterdir():
            type_file = entry / "type"
            online_file = entry / "online"
            if type_file.exists() and "battery" in type_file.read_text().strip().lower():
                has_battery = True
            if online_file.exists() and online_file.read_text().strip() == "1":
                ac_online = True
        if not has_battery:
            return "ac"
        return "ac" if ac_online else "battery"
    except Exception:
        return "unknown"


def get_hardware_info(power_source_override: str = "auto", device_class_override: str = "auto") -> dict:
    cpu_model = _read_proc_cpuinfo_model() or platform.processor() or platform.machine()
    physical, logical = _cpu_core_counts()
    mem = _mem_info_gb()
    gpu_model, gpu_vram_gb = _detect_gpu()
    device_class, device_type = _detect_device_class()
    if device_class_override != "auto":
        device_class = device_class_override.replace("-", "_")
    power_source = _detect_power_source() if power_source_override == "auto" else power_source_override

    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_cores_physical": physical,
        "cpu_threads_logical": logical,
        **mem,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine_arch": platform.machine(),
        "device_class": device_class,
        "device_type": device_type,
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "power_source": power_source,
    }


def classify_result_path(hw: dict, nickname: str, out_root: Path) -> Path:
    """Same laptop-desktop/tier-N vs edge-device/<board> split used by the
    LLM benchmark, so STT results land in a consistent place."""
    slug = re.sub(r"[^a-z0-9]+", "-", nickname.lower()).strip("-") or "device"
    if hw.get("device_class") == "edge_device":
        dt = (hw.get("device_type") or "").lower()
        if "raspberry" in dt:
            sub = "raspberry-pi"
        elif "jetson" in dt or "tegra" in dt:
            sub = "jetson-nano"
        else:
            sub = "other-sbc"
        return out_root / "edge-device" / sub / slug

    ram = hw.get("ram_total_gb") or 0
    if ram <= 4:
        tier = "tier-1-ultra-low-ram"
    elif ram <= 8:
        tier = "tier-2-low-ram"
    else:
        tier = "tier-3-mid-ram"
    return out_root / "laptop-desktop" / tier / slug
