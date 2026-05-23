"""Benchmarking utilities for AMD GPUs."""

import time
import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class BenchResult:
    test_name: str
    fps: float
    latency_ms: float
    vram_gb: float
    gpu_name: str


def benchmark_interpolator(
    interpolator,
    resolutions: List[Tuple[int, int]] = None,
    warmup: int = 50,
    iterations: int = 300,
) -> List[BenchResult]:
    """Benchmark frame interpolation across resolutions."""
    if resolutions is None:
        resolutions = [(1280, 720), (1920, 1080), (3840, 2160)]

    results = []
    gpu_name = torch.cuda.get_device_properties(0).name

    for w, h in resolutions:
        duration = 0

        for _ in range(warmup):
            fa = torch.randn(1, 3, h, w, device=interpolator.device)
            fb = torch.randn(1, 3, h, w, device=interpolator.device)
            _ = interpolator.interpolate(fa, fb, num_frames=4)

        if interpolator.device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(iterations):
            fa = torch.randn(1, 3, h, w, device=interpolator.device)
            fb = torch.randn(1, 3, h, w, device=interpolator.device)
            if interpolator.device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = interpolator.interpolate(fa, fb, num_frames=4)
            if interpolator.device.type == "cuda":
                torch.cuda.synchronize()
            duration += (time.perf_counter() - t0) * 1000

        avg_latency = duration / iterations
        fps = 1000.0 / avg_latency * 5
        vram = torch.cuda.memory_allocated() / 1024**3

        results.append(BenchResult(
            test_name=f"Interpolation {w}x{h}",
            fps=round(fps, 1),
            latency_ms=round(avg_latency, 2),
            vram_gb=round(vram, 2),
            gpu_name=gpu_name,
        ))

    return results


def print_bench_table(results: List[BenchResult]) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"  AMD-FluidMotion Benchmark -- {results[0].gpu_name}")
    print(f"{'='*70}")
    print(f"  {'Test':<28} {'FPS':>8} {'Latency':>10} {'VRAM':>8}")
    print(f"  {'-'*56}")
    for r in results:
        print(f"  {r.test_name:<28} {r.fps:>7.1f} {r.latency_ms:>7.2f}ms {r.vram_gb:>5.1f}GB")
    print(f"{'='*70}\n")
