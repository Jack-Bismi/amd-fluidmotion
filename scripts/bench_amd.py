#!/usr/bin/env python3
"""AMD GPU Benchmark Suite for FluidMotion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from fluidmotion import FluidInterpolator, FluidUpscaler
from fluidmotion.utils import (
    print_amd_banner, get_amd_gpu_info,
    benchmark_interpolator, print_bench_table,
)


def main():
    print_amd_banner()

    info = get_amd_gpu_info()
    if info.get("error"):
        print(f"Error: {info['error']}")
        sys.exit(1)

    print(f"GPU: {info['name']} | {info['vram_gb']} GB | {info.get('arch', 'Unknown')}")
    print("Starting comprehensive benchmark...\n")

    print("[1/2] Frame Interpolation Benchmark")
    print("-" * 50)
    interpolator = FluidInterpolator(fp16=True)
    interp_results = benchmark_interpolator(
        interpolator,
        resolutions=[(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)],
        warmup=30,
        iterations=200,
    )
    print_bench_table(interp_results)

    print("[2/2] Super-Resolution Benchmark")
    print("-" * 50)
    for scale in [2, 4]:
        upscaler = FluidUpscaler(scale=scale, fp16=True)
        for res in [(1280, 720), (1920, 1080)]:
            h, w = res
            frame = torch.randn(1, 3, h, w, device=upscaler.device)

            for _ in range(30):
                _ = upscaler.upscale(frame)
            torch.cuda.synchronize()

            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = upscaler.upscale(frame)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            avg_lat = sum(times) / len(times)
            fps = 1000.0 / avg_lat
            vram = torch.cuda.memory_allocated() / 1024**3

            print(f"  {scale}x Upscale {w}x{h:<12} {fps:>7.1f} FPS  {avg_lat:>6.2f}ms  {vram:>4.1f}GB")

    print(f"\nBenchmark complete on {info['name']}")


if __name__ == "__main__":
    main()
