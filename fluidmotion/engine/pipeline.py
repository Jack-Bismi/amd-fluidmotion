"""End-to-End FluidMotion Pipeline.

Handles video I/O, frame scheduling, and coordinates
interpolation + upscaling in real-time.
"""

import time
import threading
from queue import Queue
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
import torch
from collections import deque

from .interpolator import FluidInterpolator
from .upscaler import FluidUpscaler


class FluidPipeline:
    """Complete FluidMotion processing pipeline.

    Reads video -> interpolates frames -> upscales -> writes output.
    All running on AMD GPU with ROCm.
    """

    def __init__(
        self,
        interpolator: FluidInterpolator,
        upscaler: Optional[FluidUpscaler] = None,
        fps_target: int = 120,
        real_time: bool = True,
        prefetch: int = 8,
    ):
        self.interpolator = interpolator
        self.upscaler = upscaler
        self.fps_target = fps_target
        self.real_time = real_time
        self.prefetch = prefetch

        self.frame_times = deque(maxlen=200)
        self.total_frames_processed = 0
        self.total_frames_output = 0

        self.input_queue: Queue = Queue(maxsize=prefetch)
        self.output_queue: Queue = Queue(maxsize=prefetch)
        self._stop_event = threading.Event()

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Process a frame pair — interpolate + upscale."""
        t_start = time.perf_counter()

        curr_t = self._numpy_to_tensor(frame).to(self.interpolator.device)

        results = []

        if prev_frame is not None:
            prev_t = self._numpy_to_tensor(prev_frame).to(self.interpolator.device)

            num_inter = self._calc_intermediate_frames()
            interpolated = self.interpolator.interpolate(prev_t, curr_t, num_inter)

            for interp_frame in interpolated:
                if self.upscaler:
                    interp_frame = self.upscaler.upscale(interp_frame)
                results.append(self._tensor_to_numpy(interp_frame))

        curr_output = curr_t
        if self.upscaler:
            curr_output = self.upscaler.upscale(curr_output)
        results.append(self._tensor_to_numpy(curr_output))

        elapsed = (time.perf_counter() - t_start) * 1000
        self.frame_times.append(elapsed)

        self.total_frames_output += len(results)
        self.total_frames_processed += 1

        return results

    def _calc_intermediate_frames(self) -> int:
        source_fps = 24
        multiplier = self.fps_target / source_fps
        return max(0, int(multiplier) - 1)

    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
        )

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )

    @property
    def current_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        avg_ms = sum(self.frame_times) / len(self.frame_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)
