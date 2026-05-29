#!/usr/bin/env python3
"""AMD-FluidMotion -- Interactive Demo"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from fluidmotion import FluidInterpolator, FluidUpscaler, FluidPipeline
from fluidmotion.app import VideoPlayer, FluidGUI
from fluidmotion.utils import print_amd_banner, get_amd_gpu_info


def main():
    parser = argparse.ArgumentParser(
        description="AMD-FluidMotion: AI Frame Interpolation & Super-Resolution for AMD GPUs"
    )
    parser.add_argument("--input", type=str, help="Input video path")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--fps-target", type=int, default=120, help="Target FPS")
    parser.add_argument("--upscale", type=str, default=None, help="2x or 4x upscaling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output", type=str, default="./output/", help="Output dir")
    parser.add_argument("--headless", action="store_true", help="No GUI")
    args = parser.parse_args()

    print_amd_banner()

    info = get_amd_gpu_info()
    if info.get("error"):
        print(f"Error: {info['error']}")
        sys.exit(1)

    print(f"AMD GPU detected: {info['name']} ({info['vram_gb']} GB, {info.get('arch', 'Unknown')})")

    print("[demo] Building interpolation engine...")
    interpolator = FluidInterpolator(fp16=True, use_wmma=True)

    upscaler = None
    if args.upscale:
        scale = int(args.upscale.replace("x", ""))
        print(f"[demo] Building {scale}x upscaler...")
        upscaler = FluidUpscaler(scale=scale, fp16=True)

    pipeline = FluidPipeline(
        interpolator=interpolator,
        upscaler=upscaler,
        fps_target=args.fps_target,
    )

    if args.benchmark:
        from fluidmotion.utils import benchmark_interpolator, print_bench_table
        results = benchmark_interpolator(interpolator)
        print_bench_table(results)
        return

    if args.webcam:
        cap = cv2.VideoCapture(0)
        source_fps = 30
        print(f"[demo] Using webcam @ {source_fps}fps")
    elif args.input:
        player = VideoPlayer(Path(args.input))
        player.open()
        source_fps = player.fps
        print(f"[demo] Input: {args.input} ({player.resolution[0]}x{player.resolution[1]} @ {source_fps:.1f}fps)")
    else:
        print("[demo] No input specified. Use --input or --webcam")
        sys.exit(1)

    gui = None
    if not args.headless:
        gui = FluidGUI()
        gui.start()

    print("[demo] Processing... Press Ctrl+C to stop")
    print(f"[demo] Source: {source_fps:.0f}fps -> Target: {args.fps_target}fps")

    prev_frame = None
    frame_count = 0

    try:
        frame_gen = player.frames() if args.input else None
        while True:
            if args.webcam:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                try:
                    frame = next(frame_gen)
                except StopIteration:
                    break

            results = pipeline.process_frame(frame, prev_frame)
            prev_frame = frame

            if gui and results:
                gui.update(frame, results[0], pipeline.current_fps)

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  [demo] Processed {frame_count} frames -> {pipeline.total_frames_output} output | "
                      f"{pipeline.current_fps:.1f} FPS | {pipeline.avg_latency_ms:.1f}ms latency")

    except KeyboardInterrupt:
        print("\n[demo] Interrupted!")
    finally:
        if gui:
            gui.stop()
        if args.webcam:
            cap.release()
        elif args.input:
            player.close()

        print(f"\nDone! {frame_count} input -> {pipeline.total_frames_output} output frames")
        print(f"   Avg: {pipeline.current_fps:.0f} FPS | {pipeline.avg_latency_ms:.1f}ms latency")


if __name__ == "__main__":
    main()
