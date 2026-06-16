# msda-triton

Fast 2D **and 3D** Multi-Scale Deformable Attention (MSDA) for PyTorch, written in [Triton](https://github.com/triton-lang/triton).

- **3D MSDA.** The original Deformable-DETR CUDA op is 2D-only; this library provides a native 3D (volumetric) variant with full autograd.
- **No compilation step.** Pure Python — installs with `pip`, no `nvcc`, no CUDA toolkit, no C++ build. The kernels JIT-compile on first use via Triton, which ships with PyTorch.
- **Fast.** 2–10x faster forward and ~2x faster training step than the pure-PyTorch fallback (see [benchmarks](#benchmarks)).
- **Drop-in.** Same tensor contract as the standard MSDA op, with forward + backward via `torch.autograd`. Supports fp32, fp16, and bf16.
- **Validated.** Every kernel is tested against a pure-PyTorch reference implementation (also included, for CPU use or double-checking).

## Installation

Install directly from GitHub:

```bash
# HTTPS
pip install git+https://github.com/mach9robotics/msda-triton.git

# or SSH
pip install git+ssh://git@github.com/mach9robotics/msda-triton.git
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, Triton ≥ 2.1, and a CUDA GPU.

## Quick start

```python
import torch
from msda_triton import msda3d

device = "cuda"
B, H, C = 2, 8, 32          # batch, heads, channels per head
Q, P = 100, 4               # queries, sampling points per level

# Two volumetric pyramid levels: 16x32x32 and 8x16x16  (D, H, W)
spatial_shapes = torch.tensor([[16, 32, 32], [8, 16, 16]], dtype=torch.int32)
K = int((spatial_shapes[:, 0] * spatial_shapes[:, 1] * spatial_shapes[:, 2]).sum())  # total keys
L = spatial_shapes.size(0)

value = torch.randn(B, K, H, C, device=device, requires_grad=True)
sampling_locations = torch.rand(B, Q, H, L, P, 3, device=device, requires_grad=True)
attention_weights = torch.rand(B, Q, H, L, P, device=device).softmax(dim=-1)

out = msda3d(value, spatial_shapes, sampling_locations, attention_weights)
out.sum().backward()        # gradients flow to value, locations, and weights

print(out.shape)            # (B, Q, H * C)
```

The 2D variant `msda2d` is identical except spatial shapes are `(L, 2)` as `(H, W)` and sampling locations have a trailing dimension of 2.

## API

| Function | Description |
|---|---|
| `msda2d` / `msda3d` | MSDA with full autograd — use these for training |
| `msda2d_fwd` / `msda3d_fwd` | Forward-only kernels (inference) |
| `msda_triton.pytorch.msda2d_pytorch` / `msda3d_pytorch` | Pure-PyTorch references |

Tensor contract (`d = 2` or `3`):

| Tensor | Shape | Notes |
|---|---|---|
| `value` | `(B, K, H, C)` | `K = Σ` of per-level sizes, flattened |
| `value_spatial_shapes` | `(L, d)` | per-level `(H, W)` or `(D, H, W)` |
| `sampling_locations` | `(B, Q, H, L, P, d)` | normalized to `[0, 1]` |
| `attention_weights` | `(B, Q, H, L, P)` | softmax over `(L, P)` |
| returns | `(B, Q, H*C)` | |

Both entry points accept `clamp_grid: bool = True` and `clamp_eps: float = 1e-4`, which clamp sampling locations away from the exact boundary for AMP stability (matching the PyTorch reference's grid-space clamp).

> **Tip:** pass `value_spatial_shapes` as a CPU tensor. The kernel uploads it asynchronously either way, and CPU-resident shapes additionally let the wrapper verify `K` against the level sizes without stalling the GPU.

## Benchmarks

Triton kernels vs. the included pure-PyTorch reference (`grid_sample`-based), fp32, on an RTX 4070 Ti (12 GB, driver 580.97, CUDA 13.0 / PyTorch CUDA 12.8, WSL2). Reproduce with:

```bash
uv run python -m benchmarks.run_benchmarks
```

### 2D MSDA

| Config | PyTorch fwd (ms) | Triton fwd (ms) | Speedup | PyTorch fwd+bwd (ms) | Triton fwd+bwd (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 0.41 | **0.16** | **2.6x** | 2.50 | **0.58** | **4.3x** |
| small | 0.48 | **0.22** | **2.1x** | 2.24 | **0.71** | **3.2x** |
| medium | 0.87 | **0.23** | **3.8x** | 3.27 | **2.23** | **1.5x** |
| large | 5.97 | **0.62** | **9.6x** | 14.99 | **6.20** | **2.4x** |
| xl | 17.82 | **1.79** | **10.0x** | 52.47 | **22.28** | **2.4x** |

### 3D MSDA

| Config | PyTorch fwd (ms) | Triton fwd (ms) | Speedup | PyTorch fwd+bwd (ms) | Triton fwd+bwd (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.32 | **0.16** | **2.1x** | 1.25 | **0.62** | **2.0x** |
| medium | 1.85 | **0.22** | **8.3x** | 6.24 | **3.34** | **1.9x** |
| large | 15.57 | **1.79** | **8.7x** | 45.44 | **21.87** | **2.1x** |
| xl | 69.06 | **8.92** | **7.7x** | 223.28 | **89.65** | **2.5x** |

## Development

```bash
uv sync --extra dev
uv run pytest tests/        # requires a CUDA GPU
uv run ruff check .
```

## License

Released under the MIT License. Copyright (c) 2026 Mach9 Robotics, Inc. See the [LICENSE](LICENSE) file for details.
