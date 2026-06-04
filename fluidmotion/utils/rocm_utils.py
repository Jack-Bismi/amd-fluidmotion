"""AMD GPU / ROCm utility functions."""

import torch
import subprocess
import json
from typing import Optional


def get_amd_gpu_info() -> dict:
    """Get detailed AMD GPU information via ROCm."""
    info = {"detected": False, "name": "Unknown", "vram_gb": 0}

    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()

    is_amd = any(x in name for x in ["amd", "radeon", "instinct", "gfx", "mi"])

    if not is_amd:
        info["error"] = "NVIDIA GPU detected. AMD-FluidMotion requires an AMD GPU."
        return info

    info.update({
        "detected": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "compute_units": props.multi_processor_count,
        "arch": _detect_arch(name, props),
    })

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["rocm_smi"] = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return info


def _detect_arch(name: str, props) -> str:
    """Detect AMD GPU architecture generation."""
    if "gfx942" in name or "mi300" in name:
        return "CDNA3"
    elif "gfx110" in name or "7900" in name or "7800" in name:
        return "RDNA3"
    elif "gfx103" in name or "6900" in name or "6800" in name or "6700" in name or "6600" in name:
        return "RDNA2"
    elif "gfx90a" in name or "mi200" in name:
        return "CDNA2"
    elif "gfx908" in name or "mi100" in name:
        return "CDNA1"
    elif "gfx101" in name or "5700" in name or "5600" in name or "5500" in name:
        return "RDNA1"
    return "Unknown"


def optimize_for_amd_gpu(model) -> None:
    """Apply AMD GPU-specific optimizations."""
    if not torch.cuda.is_available():
        return

    name = torch.cuda.get_device_properties(0).name.lower()
    is_amd = any(x in name for x in ["amd", "radeon", "instinct", "gfx"])

    if not is_amd:
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 16:
        torch.cuda.set_per_process_memory_fraction(0.90)

    gpu_arch = _detect_arch(name, torch.cuda.get_device_properties(0))
    print(f"[rocm_utils] Optimized for {gpu_arch} GPU ({vram_gb:.0f} GB VRAM)")


def print_amd_banner():
    """Print fancy AMD GPU banner."""
    info = get_amd_gpu_info()

    if info.get("error"):
        print(f"\n{info['error']}\n")
        return

    banner = f"""
============================================================
      AMD-FluidMotion -- ROCm Edition
============================================================
  GPU      : {info['name']}
  Arch     : {info.get('arch', 'Unknown')}
  VRAM     : {info['vram_gb']:.1f} GB
  Backend  : ROCm + HIP
============================================================
"""
    print(banner)
