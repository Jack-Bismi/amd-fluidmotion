# -*- coding: utf-8 -*-
__version__ = "0.1.1"

"""AMD-FluidMotion — Real-Time AI Frame Interpolation & Super-Resolution
Built exclusively for AMD GPUs with ROCm.
"""

version = "0.2.1"
author = "Indie Developer"
license = "MIT"

from .engine.interpolator import FluidInterpolator
from .engine.upscaler import FluidUpscaler
from .engine.pipeline import FluidPipeline

__all__ = ["FluidInterpolator", "FluidUpscaler", "FluidPipeline"]
