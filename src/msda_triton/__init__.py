"""
Written by Abhinav Atrishi <abhinav@mach9.io>, January 2026.

MSDA-Triton package exports for Multi-Scale Deformable Attention kernels.

Copyright (c) 2026 Mach9 Robotics, Inc.
Licensed under the MIT License. See the LICENSE file for details.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Abhinav Atrishi"
__email__ = "abhinav@mach9.io"

from msda_triton.triton.msda_2d import msda2d, msda2d_fwd
from msda_triton.triton.msda_3d import msda3d, msda3d_fwd

__all__ = [
    "__version__",
    # 2D (image)
    "msda2d",
    "msda2d_fwd",
    # 3D (volumetric)
    "msda3d",
    "msda3d_fwd",
]
