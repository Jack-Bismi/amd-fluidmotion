"""Unit tests for FluidMotion interpolator."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluidmotion.models import build_flow_net, build_fusion_net


class TestFlowNet:
    def test_output_shape(self):
        net = build_flow_net("pwc_net")
        a = torch.randn(1, 3, 256, 256)
        b = torch.randn(1, 3, 256, 256)
        flow = net(a, b)
        assert flow.shape == (1, 2, 256, 256)

    def test_zero_motion(self):
        net = build_flow_net("pwc_net")
        img = torch.randn(1, 3, 256, 256)
        flow = net(img, img)
        assert flow.abs().mean() < 0.5


class TestFusionNet:
    def test_output_shape(self):
        net = build_fusion_net()
        a = torch.randn(1, 3, 256, 256)
        b = torch.randn(1, 3, 256, 256)
        out = net(a, b, 0.5)
        assert out.shape == (1, 3, 256, 256)

    def test_range(self):
        net = build_fusion_net()
        a = torch.rand(1, 3, 64, 64)
        b = torch.rand(1, 3, 64, 64)
        out = net(a, b, 0.5)
        assert out.min() >= 0
        assert out.max() <= 1
