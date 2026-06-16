"""
Written by Abhinav Atrishi <abhinav@mach9.io>, January 2026.

Shared pytest fixtures for MSDA-Triton test suite.

Copyright (c) 2026 Mach9 Robotics, Inc.
Licensed under the MIT License. See the LICENSE file for details.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def seed_rng() -> None:
    """Ensure reproducible tests by seeding RNG."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
