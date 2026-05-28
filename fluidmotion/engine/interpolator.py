"""Frame Interpolation Engine — AMD GPU Optimized.

Uses PWC-Net for optical flow + U-Net attention for frame synthesis.
All kernels optimized for AMD RDNA/CDNA wavefront execution.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from collections import deque

from ..models.flow_net import build_flow_net
from ..models.fusion_net import build_fusion_net
from ..utils.rocm_utils import optimize_for_amd_gpu, get_amd_gpu_info


class FluidInterpolator(nn.Module):
    """Real-time AI frame interpolator for AMD GPUs.

    Takes two consecutive frames and generates N intermediate frames
    using bidirectional optical flow with adaptive fusion.
    """

    def __init__(
        self,
        flow_model: str = "pwc_net",
        fp16: bool = True,
        use_wmma: bool = True,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.use_wmma = use_wmma

        self.flow_net = build_flow_net(flow_model, pretrained=True)
        self.fusion_net = build_fusion_net()

        self.flow_net.to(self.device)
        self.fusion_net.to(self.device)

        if self.fp16:
            self.flow_net = self.flow_net.half()
            self.fusion_net = self.fusion_net.half()

        for param in self.flow_net.parameters():
            param.requires_grad = False

        self.prev_flows: deque = deque(maxlen=5)
        self.prev_frames: deque = deque(maxlen=3)

        optimize_for_amd_gpu(self)
        self.gpu_info = get_amd_gpu_info()

        self.num_streams = 3
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)] if self.device.type == "cuda" else []

    @torch.no_grad()
    def interpolate(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
        num_frames: int = 4,
        temporal_weight: float = 0.82,
    ) -> List[torch.Tensor]:
        """Generate interpolated frames between frame_a and frame_b."""
        if self.fp16:
            frame_a = frame_a.half()
            frame_b = frame_b.half()

        flow_ab = self.flow_net(frame_a, frame_b)
        flow_ba = self.flow_net(frame_b, frame_a)

        if len(self.prev_flows) > 0 and temporal_weight > 0:
            flow_ab = self._smooth_flow(flow_ab, 0, temporal_weight)
            flow_ba = self._smooth_flow(flow_ba, 1, temporal_weight)

        self.prev_flows.append((flow_ab.detach(), flow_ba.detach()))
        self.prev_frames.append(frame_a.detach())

        outputs = []
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)

            warped_a = self._warp_frame(frame_a, flow_ab, t)
            warped_b = self._warp_frame(frame_b, flow_ba, 1 - t)

            intermediate = self.fusion_net(warped_a, warped_b, t)
            intermediate = torch.clamp(intermediate, 0.0, 1.0)

            if self.fp16:
                intermediate = intermediate.float()

            outputs.append(intermediate)

        return outputs

    def _warp_frame(self, frame: torch.Tensor, flow: torch.Tensor, t: float) -> torch.Tensor:
        """Warp frame using optical flow at timestep t."""
        scaled_flow = flow * t
        B, C, H, W = frame.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=torch.float32),
            torch.arange(W, device=frame.device, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = grid_x + scaled_flow[:, 0, :, :]
        grid_y = grid_y + scaled_flow[:, 1, :, :]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        warped = torch.nn.functional.grid_sample(
            frame, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped

    def _smooth_flow(self, flow: torch.Tensor, direction: int, weight: float) -> torch.Tensor:
        """Apply temporal smoothing to reduce flicker."""
        if len(self.prev_flows) == 0:
            return flow
        prev_flow = self.prev_flows[-1][direction]
        return weight * prev_flow.detach() + (1 - weight) * flow

    @property
    def vram_used_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated() / 1024**3

    @property
    def vram_total_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024**3


def build_interpolator(preset: str = "quality") -> FluidInterpolator:
    """Factory with quality presets."""
    presets = {
        "quality": {"fp16": True, "use_wmma": True},
        "balanced": {"fp16": True, "use_wmma": False},
        "performance": {"fp16": False, "use_wmma": False},
    }
    config = presets.get(preset, presets["balanced"])
    return FluidInterpolator(**config)
