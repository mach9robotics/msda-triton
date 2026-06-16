"""
Written by Abhinav Atrishi <abhinav@mach9.io>, January 2026.

Triton implementation of 2D Multi-Scale Deformable Attention (MSDA2D).

Copyright (c) 2026 Mach9 Robotics, Inc.
Licensed under the MIT License. See the LICENSE file for details.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _prepare_spatial_2d(
    value: torch.Tensor, value_spatial_shapes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate spatial shapes against `value` and build per-level start indices.

    Converts per-level 2D shapes (H,W) to flat start indices for efficient indexing:

        Level 0: H₀xW₀ pixels  →  start = 0
        Level 1: H₁xW₁ pixels  →  start = H₀xW₀
        Level 2: H₂xW₂ pixels  →  start = H₀xW₀ + H₁xW₁

    Example:
        spatial_shapes = [[64,64], [32,32], [16,16]]  # L=3 levels
        →  sizes = [4096, 1024, 256]
        →  starts = [0, 4096, 5120]

    When `value_spatial_shapes` lives on the CPU, the per-level sizes are also
    checked against value's K dimension for free. When it lives on the GPU, the
    start indices are computed on-device without forcing a host-device sync, and
    the K check is skipped — pass spatial shapes on CPU to get full validation
    at no cost.

    Args:
        value: (B, K, H, C) feature tensor; only K is inspected.
        value_spatial_shapes: (L, 2) tensor with (H, W) per level.

    Returns:
        Tuple of (L, 2) and (L,) int32 tensors on value's device:
        spatial shapes and cumulative start indices.

    Raises:
        ValueError: If shapes are CPU-resident and K does not equal Σ(H_l x W_l),
            or K exceeds int32 range.
    """
    device = value.device

    if value_spatial_shapes.device.type == "cpu":
        shapes_cpu = value_spatial_shapes.detach().to(dtype=torch.int64)
        lvl_sizes = shapes_cpu[:, 0] * shapes_cpu[:, 1]
        starts = torch.zeros_like(lvl_sizes)
        if lvl_sizes.numel() > 1:
            starts[1:] = torch.cumsum(lvl_sizes[:-1], dim=0)

        total_keys = int(lvl_sizes.sum())
        if total_keys > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"Total number of keys ({total_keys}) exceeds int32 range. "
                "Reduce spatial dimensions or number of levels."
            )
        if value.size(1) != total_keys:
            raise ValueError(
                f"value dim 1 ({value.size(1)}) does not match sum of spatial shapes "
                f"({total_keys}). Ensure K = Σ(H_l × W_l) across all levels."
            )
        return (
            shapes_cpu.to(device=device, dtype=torch.int32),
            starts.to(device=device, dtype=torch.int32),
        )

    # GPU-resident shapes: stay asynchronous (reading values would stall the stream).
    spatial_i32 = value_spatial_shapes.to(dtype=torch.int32)
    lvl_sizes = (spatial_i32[:, 0] * spatial_i32[:, 1]).to(torch.int64)
    starts = torch.zeros_like(lvl_sizes)
    if lvl_sizes.numel() > 1:
        starts[1:] = torch.cumsum(lvl_sizes[:-1], dim=0)
    return spatial_i32, starts.to(torch.int32)


@triton.jit
def _fwd_kernel_2d(
    value_ptr,
    sampling_ptr,
    attn_ptr,
    spatial_ptr,
    lvl_start_ptr,
    out_ptr,
    Q: tl.constexpr,
    H_: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    sv_b: tl.constexpr,
    sv_k: tl.constexpr,
    sv_h: tl.constexpr,
    sv_c: tl.constexpr,
    ss_b: tl.constexpr,
    ss_q: tl.constexpr,
    ss_h: tl.constexpr,
    ss_l: tl.constexpr,
    ss_p: tl.constexpr,
    ss_c2: tl.constexpr,
    sa_b: tl.constexpr,
    sa_q: tl.constexpr,
    sa_h: tl.constexpr,
    sa_l: tl.constexpr,
    sa_p: tl.constexpr,
    so_bh: tl.constexpr,
    so_q: tl.constexpr,
    so_c: tl.constexpr,
    clamp_grid: tl.constexpr,
    clamp_eps: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """MSDA2D forward kernel: fused bilinear interpolation + attention weighting.

    Computation Flow (per thread block):
    ════════════════════════════════════════════════════════════════════════

        Grid: (BxH, ⌈Q/BLOCK_Q⌉, ⌈C/BLOCK_C⌉)  →  Each block processes:
                                                  • BLOCK_Q queries
                                                  • BLOCK_C channels

        ┌─────────────────────────────────────────────────────────────┐
        │  For each (level, point) pair:                              │
        │                                                             │
        │  1. Load sampling coords (x,y) ∈ [0,1]                      │
        │     ↓                                                       │
        │  2. Map to pixel space: x_img = x×W - 0.5                   │
        │     ↓                                                       │
        │  3. Find 4 neighbors:                                       │
        │                                                             │
        │      (x0,y0)────(x1,y0)                                     │
        │         │    ●P    │       • Load 4 pixel values            │
        │         │          │       • Compute bilinear weights       │
        │      (x0,y1)────(x1,y1)   • sample = Σ w_ij x value_ij      │
        │     ↓                                                       │
        │  4. Weight by attention: acc += sample x attn_weight[l,p]   │
        │                                                             │
        │  5. Loop over all L levels, P points per level              │
        │                                                             │
        │  6. Store: out[b,h,q,c] = acc (fp32→input_dtype)            │
        └─────────────────────────────────────────────────────────────┘

    Optimization Notes:
        • FP32 accumulation over LxP iterations
        • Fully inlined 4-neighbor loads
        • Optional grid clamping: x ∈ [ε, 1-ε] for AMP stability
        • Coalesced memory access via (BxH, Q, C) layout
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_c = tl.program_id(2)

    q_off = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    q_mask = q_off < Q
    c_mask = c_off < C

    b = pid_bh // H_
    h = pid_bh % H_

    acc = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
    base_val_bh = b * sv_b + h * sv_h

    if clamp_grid != 0:
        # PyTorch reference clamps in grid space [-1, 1] before converting to image coords:
        # grid = 2 * x - 1, clamp(grid, -1 + eps, 1 - eps)
        # Equivalent location-space clamp is x in [eps/2, 1 - eps/2].
        clamp_lo = 0.5 * clamp_eps
        clamp_hi = 1.0 - 0.5 * clamp_eps

    for lvl_idx in range(L):
        h_i = tl.load(spatial_ptr + lvl_idx * 2 + 0).to(tl.int32)
        w_i = tl.load(spatial_ptr + lvl_idx * 2 + 1).to(tl.int32)
        h_f = tl.cast(h_i, tl.float32)
        w_f = tl.cast(w_i, tl.float32)
        lvl_start = tl.load(lvl_start_ptr + lvl_idx).to(tl.int32)

        for p in range(P):
            base_s = b * ss_b + h * ss_h + lvl_idx * ss_l + p * ss_p
            x_raw = tl.load(
                sampling_ptr + base_s + q_off * ss_q + 0 * ss_c2, mask=q_mask, other=0.0
            )
            y_raw = tl.load(
                sampling_ptr + base_s + q_off * ss_q + 1 * ss_c2, mask=q_mask, other=0.0
            )
            x = x_raw
            y = y_raw

            if clamp_grid != 0:
                x = tl.maximum(clamp_lo, tl.minimum(clamp_hi, x))
                y = tl.maximum(clamp_lo, tl.minimum(clamp_hi, y))

            # Map to image coordinates (align_corners=False)
            x_img = x * w_f - 0.5
            y_img = y * h_f - 0.5

            x0 = tl.floor(x_img).to(tl.int32)
            y0 = tl.floor(y_img).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            # Bilinear weights
            dx = x_img - tl.cast(x0, tl.float32)
            dy = y_img - tl.cast(y0, tl.float32)
            dx1 = 1.0 - dx
            dy1 = 1.0 - dy

            w00 = dx1 * dy1
            w01 = dx1 * dy
            w10 = dx * dy1
            w11 = dx * dy

            # Load 4 neighbors (fully inlined)
            # Neighbor 00 (x0, y0)
            valid_00 = (x0 >= 0) & (x0 < w_i) & (y0 >= 0) & (y0 < h_i)
            k_idx_00 = lvl_start + y0 * w_i + x0
            ptr_00 = value_ptr + base_val_bh + k_idx_00[:, None] * sv_k + c_off[None, :] * sv_c
            mask_00 = q_mask[:, None] & c_mask[None, :] & valid_00[:, None]
            v00 = tl.load(ptr_00, mask=mask_00, other=0.0).to(tl.float32) * w00[:, None]

            # Neighbor 01 (x0, y1)
            valid_01 = (x0 >= 0) & (x0 < w_i) & (y1 >= 0) & (y1 < h_i)
            k_idx_01 = lvl_start + y1 * w_i + x0
            ptr_01 = value_ptr + base_val_bh + k_idx_01[:, None] * sv_k + c_off[None, :] * sv_c
            mask_01 = q_mask[:, None] & c_mask[None, :] & valid_01[:, None]
            v01 = tl.load(ptr_01, mask=mask_01, other=0.0).to(tl.float32) * w01[:, None]

            # Neighbor 10 (x1, y0)
            valid_10 = (x1 >= 0) & (x1 < w_i) & (y0 >= 0) & (y0 < h_i)
            k_idx_10 = lvl_start + y0 * w_i + x1
            ptr_10 = value_ptr + base_val_bh + k_idx_10[:, None] * sv_k + c_off[None, :] * sv_c
            mask_10 = q_mask[:, None] & c_mask[None, :] & valid_10[:, None]
            v10 = tl.load(ptr_10, mask=mask_10, other=0.0).to(tl.float32) * w10[:, None]

            # Neighbor 11 (x1, y1)
            valid_11 = (x1 >= 0) & (x1 < w_i) & (y1 >= 0) & (y1 < h_i)
            k_idx_11 = lvl_start + y1 * w_i + x1
            ptr_11 = value_ptr + base_val_bh + k_idx_11[:, None] * sv_k + c_off[None, :] * sv_c
            mask_11 = q_mask[:, None] & c_mask[None, :] & valid_11[:, None]
            v11 = tl.load(ptr_11, mask=mask_11, other=0.0).to(tl.float32) * w11[:, None]

            sample = v00 + v01 + v10 + v11

            aw = tl.load(
                attn_ptr + b * sa_b + h * sa_h + lvl_idx * sa_l + p * sa_p + q_off * sa_q,
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
            acc += sample * aw[:, None]

    out_ptrs = out_ptr + pid_bh * so_bh + q_off[:, None] * so_q + c_off[None, :] * so_c
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & c_mask[None, :])


@triton.jit
def _bwd_kernel_2d(
    value_ptr,
    sampling_ptr,
    attn_ptr,
    grad_out_ptr,
    spatial_ptr,
    lvl_start_ptr,
    grad_value_ptr,
    grad_sampling_ptr,
    grad_attn_ptr,
    Q: tl.constexpr,
    H_: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    sv_b: tl.constexpr,
    sv_k: tl.constexpr,
    sv_h: tl.constexpr,
    sv_c: tl.constexpr,
    ss_b: tl.constexpr,
    ss_q: tl.constexpr,
    ss_h: tl.constexpr,
    ss_l: tl.constexpr,
    ss_p: tl.constexpr,
    ss_c2: tl.constexpr,
    sa_b: tl.constexpr,
    sa_q: tl.constexpr,
    sa_h: tl.constexpr,
    sa_l: tl.constexpr,
    sa_p: tl.constexpr,
    sgo_bh: tl.constexpr,
    sgo_q: tl.constexpr,
    sgo_c: tl.constexpr,
    clamp_grid: tl.constexpr,
    clamp_eps: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """MSDA2D backward kernel: compute gradients via chain rule + bilinear derivatives.

    Gradient Flow Architecture:
    ═══════════════════════════════════════════════════════════════════════════

                           grad_out (dL/dO)
                                 │
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
             dL/d(value)   dL/d(attn)   dL/d(sampling)


    1. Gradient w.r.t. Value (scatter to 4 neighbors):
       ─────────────────────────────────────────────
       grad_value[x_i,y_j] += grad_out x attn_weight[l,p] x w_ij

    2. Gradient w.r.t. Attention Weight (dot product):
       ───────────────────────────────────────────────
        grad_attn[l,p] = grad_out · sampled_value

    3. Gradient w.r.t. Sampling Locations (chain through bilinear):
       ─────────────────────────────────────────────────────────────

       Derivative Table (4 neighbors):
       ────────────────────────────────
           Corner    |  ∂w/∂x      |  ∂w/∂y
           ──────────┼─────────────┼─────────────
           (x0,y0)   │ -dy1        │ -dx1
           (x0,y1)   │ -dy         │ +dx1
           (x1,y0)   │ +dy1        │ -dx
           (x1,y1)   │ +dy         │ +dx

           where: dx1=(1-dx), dy1=(1-dy)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_c = tl.program_id(2)

    q_off = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    q_mask = q_off < Q
    c_mask = c_off < C

    b = pid_bh // H_
    h = pid_bh % H_

    base_val_bh = b * sv_b + h * sv_h

    if clamp_grid != 0:
        # Match PyTorch grid-space clamping semantics (see forward kernel note).
        clamp_lo = 0.5 * clamp_eps
        clamp_hi = 1.0 - 0.5 * clamp_eps

    for lvl_idx in range(L):
        h_i = tl.load(spatial_ptr + lvl_idx * 2 + 0).to(tl.int32)
        w_i = tl.load(spatial_ptr + lvl_idx * 2 + 1).to(tl.int32)
        h_f = tl.cast(h_i, tl.float32)
        w_f = tl.cast(w_i, tl.float32)
        lvl_start = tl.load(lvl_start_ptr + lvl_idx).to(tl.int32)

        for p in range(P):
            base_s = b * ss_b + h * ss_h + lvl_idx * ss_l + p * ss_p
            x_raw = tl.load(
                sampling_ptr + base_s + q_off * ss_q + 0 * ss_c2,
                mask=q_mask,
                other=0.0,
            )
            y_raw = tl.load(
                sampling_ptr + base_s + q_off * ss_q + 1 * ss_c2,
                mask=q_mask,
                other=0.0,
            )
            x = x_raw
            y = y_raw

            if clamp_grid != 0:
                x = tl.maximum(clamp_lo, tl.minimum(clamp_hi, x))
                y = tl.maximum(clamp_lo, tl.minimum(clamp_hi, y))

            x_img = x * w_f - 0.5
            y_img = y * h_f - 0.5

            x0 = tl.floor(x_img).to(tl.int32)
            y0 = tl.floor(y_img).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            dx = x_img - tl.cast(x0, tl.float32)
            dy = y_img - tl.cast(y0, tl.float32)
            dx1 = 1.0 - dx
            dy1 = 1.0 - dy

            w00 = dx1 * dy1
            w01 = dx1 * dy
            w10 = dx * dy1
            w11 = dx * dy

            # Load attention weight and grad_out for this query/point
            aw = tl.load(
                attn_ptr + b * sa_b + h * sa_h + lvl_idx * sa_l + p * sa_p + q_off * sa_q,
                mask=q_mask,
                other=0.0,
            )

            grad_out_tile = tl.load(
                grad_out_ptr + pid_bh * sgo_bh + q_off[:, None] * sgo_q + c_off[None, :] * sgo_c,
                mask=q_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            grad_out_weighted = grad_out_tile * aw[:, None]

            grad_x_accum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
            grad_y_accum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
            grad_attn_accum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

            # Process 4 neighbors
            # Neighbor 00 (x0, y0)
            valid_00 = (x0 >= 0) & (x0 < w_i) & (y0 >= 0) & (y0 < h_i)
            k_idx_00 = lvl_start + y0 * w_i + x0
            val_ptr_00 = value_ptr + base_val_bh + k_idx_00[:, None] * sv_k + c_off[None, :] * sv_c
            val_mask_00 = q_mask[:, None] & c_mask[None, :] & valid_00[:, None]
            val_00 = tl.load(val_ptr_00, mask=val_mask_00, other=0.0).to(tl.float32)
            grad_val_00 = grad_out_weighted * w00[:, None]
            grad_val_ptr_00 = (
                grad_value_ptr + base_val_bh + k_idx_00[:, None] * sv_k + c_off[None, :] * sv_c
            )
            tl.atomic_add(grad_val_ptr_00, grad_val_00, mask=val_mask_00)
            grad_out_val_00 = (grad_out_tile * val_00).sum(axis=1) * aw
            grad_x_accum += grad_out_val_00 * (-dy1)
            grad_y_accum += grad_out_val_00 * (-dx1)
            grad_attn_accum += (grad_out_tile * val_00 * w00[:, None]).sum(axis=1)

            # Neighbor 01 (x0, y1)
            valid_01 = (x0 >= 0) & (x0 < w_i) & (y1 >= 0) & (y1 < h_i)
            k_idx_01 = lvl_start + y1 * w_i + x0
            val_ptr_01 = value_ptr + base_val_bh + k_idx_01[:, None] * sv_k + c_off[None, :] * sv_c
            val_mask_01 = q_mask[:, None] & c_mask[None, :] & valid_01[:, None]
            val_01 = tl.load(val_ptr_01, mask=val_mask_01, other=0.0).to(tl.float32)
            grad_val_01 = grad_out_weighted * w01[:, None]
            grad_val_ptr_01 = (
                grad_value_ptr + base_val_bh + k_idx_01[:, None] * sv_k + c_off[None, :] * sv_c
            )
            tl.atomic_add(grad_val_ptr_01, grad_val_01, mask=val_mask_01)
            grad_out_val_01 = (grad_out_tile * val_01).sum(axis=1) * aw
            grad_x_accum += grad_out_val_01 * (-dy)
            grad_y_accum += grad_out_val_01 * (dx1)
            grad_attn_accum += (grad_out_tile * val_01 * w01[:, None]).sum(axis=1)

            # Neighbor 10 (x1, y0)
            valid_10 = (x1 >= 0) & (x1 < w_i) & (y0 >= 0) & (y0 < h_i)
            k_idx_10 = lvl_start + y0 * w_i + x1
            val_ptr_10 = value_ptr + base_val_bh + k_idx_10[:, None] * sv_k + c_off[None, :] * sv_c
            val_mask_10 = q_mask[:, None] & c_mask[None, :] & valid_10[:, None]
            val_10 = tl.load(val_ptr_10, mask=val_mask_10, other=0.0).to(tl.float32)
            grad_val_10 = grad_out_weighted * w10[:, None]
            grad_val_ptr_10 = (
                grad_value_ptr + base_val_bh + k_idx_10[:, None] * sv_k + c_off[None, :] * sv_c
            )
            tl.atomic_add(grad_val_ptr_10, grad_val_10, mask=val_mask_10)
            grad_out_val_10 = (grad_out_tile * val_10).sum(axis=1) * aw
            grad_x_accum += grad_out_val_10 * (dy1)
            grad_y_accum += grad_out_val_10 * (-dx)
            grad_attn_accum += (grad_out_tile * val_10 * w10[:, None]).sum(axis=1)

            # Neighbor 11 (x1, y1)
            valid_11 = (x1 >= 0) & (x1 < w_i) & (y1 >= 0) & (y1 < h_i)
            k_idx_11 = lvl_start + y1 * w_i + x1
            val_ptr_11 = value_ptr + base_val_bh + k_idx_11[:, None] * sv_k + c_off[None, :] * sv_c
            val_mask_11 = q_mask[:, None] & c_mask[None, :] & valid_11[:, None]
            val_11 = tl.load(val_ptr_11, mask=val_mask_11, other=0.0).to(tl.float32)
            grad_val_11 = grad_out_weighted * w11[:, None]
            grad_val_ptr_11 = (
                grad_value_ptr + base_val_bh + k_idx_11[:, None] * sv_k + c_off[None, :] * sv_c
            )
            tl.atomic_add(grad_val_ptr_11, grad_val_11, mask=val_mask_11)
            grad_out_val_11 = (grad_out_tile * val_11).sum(axis=1) * aw
            grad_x_accum += grad_out_val_11 * (dy)
            grad_y_accum += grad_out_val_11 * (dx)
            grad_attn_accum += (grad_out_tile * val_11 * w11[:, None]).sum(axis=1)

            # Scale by spatial dimensions (chain rule: d_img/d_norm = spatial_size)
            grad_x_final = grad_x_accum * tl.cast(w_i, tl.float32)
            grad_y_final = grad_y_accum * tl.cast(h_i, tl.float32)
            if clamp_grid != 0:
                # Match clamp backward semantics: gradients are zero for clamped values.
                grad_x_final = tl.where(
                    (x_raw >= clamp_lo) & (x_raw <= clamp_hi),
                    grad_x_final,
                    0.0,
                )
                grad_y_final = tl.where(
                    (y_raw >= clamp_lo) & (y_raw <= clamp_hi),
                    grad_y_final,
                    0.0,
                )

            # Atomic adds: every BLOCK_C tile of this (b,h,q) accumulates into the
            # same (l,p) slots, so contention grows with ceil(C / BLOCK_C).
            grad_s_ptr_x = grad_sampling_ptr + base_s + q_off * ss_q + 0 * ss_c2
            grad_s_ptr_y = grad_sampling_ptr + base_s + q_off * ss_q + 1 * ss_c2
            tl.atomic_add(grad_s_ptr_x, grad_x_final, mask=q_mask)
            tl.atomic_add(grad_s_ptr_y, grad_y_final, mask=q_mask)

            # Write gradient for attention weight (atomic add)
            grad_a_ptr = (
                grad_attn_ptr + b * sa_b + h * sa_h + lvl_idx * sa_l + p * sa_p + q_off * sa_q
            )
            tl.atomic_add(grad_a_ptr, grad_attn_accum, mask=q_mask)


def _block_sizes_2d(q: int, c: int) -> tuple[int, int]:
    """Select optimal Triton block sizes based on problem dimensions.

    Balances occupancy, shared memory, and register pressure:

        Dimension Space          Block Choice     Rationale
        ──────────────────────────────────────────────────────
        Q≥64,  C≥64         →   (64, 64)         Max occupancy
        Q≥32,  C≥64         →   (32, 64)         Channel-heavy
        Q≥64,  C≥32         →   (64, 32)         Query-heavy
        Otherwise           →   (32, 32)         Conservative

    Args:
        q: Number of query tokens.
        c: Channel dimension per head.

    Returns:
        (BLOCK_Q, BLOCK_C) tuple.
    """
    if q >= 64 and c >= 64:
        return 64, 64
    if q >= 32 and c >= 64:
        return 32, 64
    if q >= 64 and c >= 32:
        return 64, 32
    return 32, 32


def _validate_inputs_2d(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> None:
    """Validate device, dtype, and tensor shapes. Raises on the first violation."""
    if not value.is_cuda:
        raise ValueError("MSDA2D Triton kernel requires CUDA tensors. Got CPU tensors instead.")

    if value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"MSDA2D Triton kernel only supports fp32, fp16, and bf16. Got {value.dtype}. "
            "The kernel accumulates in fp32 internally."
        )

    if value.dim() != 4:
        raise ValueError(f"value must be (B, K, H, C), got {tuple(value.shape)}")

    if sampling_locations.dim() != 6:
        raise ValueError(
            f"sampling_locations must be (B, Q, H, L, P, 2), got {tuple(sampling_locations.shape)}"
        )

    bsz, _, num_heads, _ = value.shape
    bsz_s, num_queries, num_heads_s, num_levels, num_points, last2 = sampling_locations.shape

    if not (bsz == bsz_s and num_heads == num_heads_s and last2 == 2):
        raise ValueError(
            "Shape mismatch: value=(B,K,H,C), sampling_locations=(B,Q,H,L,P,2). "
            f"Got value={tuple(value.shape)}, sampling_locations={tuple(sampling_locations.shape)}"
        )

    if attention_weights.shape != (bsz, num_queries, num_heads, num_levels, num_points):
        raise ValueError(
            f"attention_weights must be (B,Q,H,L,P), got {tuple(attention_weights.shape)}"
        )

    if value_spatial_shapes.shape != (num_levels, 2):
        raise ValueError(
            f"value_spatial_shapes must be (L,2), got {tuple(value_spatial_shapes.shape)}. "
            f"num_levels={num_levels}, expected shape=({num_levels}, 2)"
        )


@torch.compiler.disable
def _msda2d_fwd_prepared(
    value: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    spatial_i32: torch.Tensor,
    lvl_start_i32: torch.Tensor,
    clamp_grid: bool,
    clamp_eps: float,
) -> torch.Tensor:
    """Launch the forward kernel with pre-validated, device-resident metadata."""
    bsz, _, num_heads, dim_per_head = value.shape
    num_queries = sampling_locations.size(1)
    num_levels = sampling_locations.size(3)
    num_points = sampling_locations.size(4)

    value_c = value.contiguous()
    sampling_c = sampling_locations.contiguous()
    attn_c = attention_weights.contiguous()

    out_bhqc = torch.empty(
        (bsz * num_heads, num_queries, dim_per_head), device=value.device, dtype=torch.float32
    )

    BLOCK_Q, BLOCK_C = _block_sizes_2d(num_queries, dim_per_head)
    grid = (bsz * num_heads, triton.cdiv(num_queries, BLOCK_Q), triton.cdiv(dim_per_head, BLOCK_C))

    _fwd_kernel_2d[grid](
        value_c,
        sampling_c,
        attn_c,
        spatial_i32,
        lvl_start_i32,
        out_bhqc,
        num_queries,
        num_heads,
        dim_per_head,
        num_levels,
        num_points,
        value_c.stride(0),
        value_c.stride(1),
        value_c.stride(2),
        value_c.stride(3),
        sampling_c.stride(0),
        sampling_c.stride(1),
        sampling_c.stride(2),
        sampling_c.stride(3),
        sampling_c.stride(4),
        sampling_c.stride(5),
        attn_c.stride(0),
        attn_c.stride(1),
        attn_c.stride(2),
        attn_c.stride(3),
        attn_c.stride(4),
        out_bhqc.stride(0),
        out_bhqc.stride(1),
        out_bhqc.stride(2),
        1 if clamp_grid else 0,
        float(clamp_eps),
        BLOCK_Q,
        BLOCK_C,
    )

    out = out_bhqc.view(bsz, num_heads, num_queries, dim_per_head).permute(0, 2, 1, 3).contiguous()
    out = out.view(bsz, num_queries, num_heads * dim_per_head)
    return out.to(dtype=value.dtype)


@torch.compiler.disable
def msda2d_fwd(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    clamp_grid: bool = True,
    clamp_eps: float = 1e-4,
) -> torch.Tensor:
    """Forward-only MSDA2D (no autograd). Use `msda2d()` for training.

    Performs multi-scale deformable attention over 2D feature pyramids:

        ┌─────────────────────────────────────────────────────────────┐
        │  Input: Multi-scale 2D features (coarse → fine resolution)  │
        │                                                             │
        │  Level 0: [B, H₀xW₀, H, C]  ◄─┐                             │
        │  Level 1: [B, H₁xW₁, H, C]  ◄─┼─ Flatten to [B,K,H,C]       │
        │  Level 2: [B, H₂xW₂, H, C]  ◄─┘                             │
        │                                                             │
        │  ↓  (for each query q, head h)                              │
        │                                                             │
        │  For each (level l, sampling point p):                      │
        │    1. Get learned offset: (x,y) ∈ [0,1]²                    │
        │    2. Bilinear sample from level l at position (x,y)        │
        │    3. Weight by attention: attn[l,p] x sampled_value        │
        │                                                             │
        │  ↓  (accumulate across L levels x P points)                 │
        │                                                             │
        │  Output: [B, Q, HxC]  ← Reshape from [BxH, Q, C]            │
        └─────────────────────────────────────────────────────────────┘

    Mathematical Formulation:
        out[b,q,h] = Σ_{l=0}^{L-1} Σ_{p=0}^{P-1} attn[b,q,h,l,p] x
                     BilinearSample(value[b,:,h,:], loc[b,q,h,l,p])

    Args:
        value: (B, K, H, C_per_head) - Flattened multi-scale features
            where K = Σ(H_l x W_l) across all levels.
        value_spatial_shapes: (L, 2) - Per-level 2D shapes (H, W).
        sampling_locations: (B, Q, H, L, P, 2) - Normalized coords ∈ [0,1]².
        attention_weights: (B, Q, H, L, P) - Softmax weights (sum to 1 over LxP).
        clamp_grid: Clamp coords to [ε, 1-ε] for AMP numerical stability.
        clamp_eps: Epsilon for grid clamping (default: 1e-4).

    Returns:
        (B, Q, HxC_per_head) attention output in input dtype.

    Raises:
        ValueError: If inputs not on CUDA or shape mismatch.

    Note:
        Accumulates in fp32 internally, then casts to input dtype.
    """
    _validate_inputs_2d(value, value_spatial_shapes, sampling_locations, attention_weights)
    spatial_i32, lvl_start_i32 = _prepare_spatial_2d(value, value_spatial_shapes)
    return _msda2d_fwd_prepared(
        value,
        sampling_locations,
        attention_weights,
        spatial_i32,
        lvl_start_i32,
        clamp_grid,
        clamp_eps,
    )


class MSDA2DFn(torch.autograd.Function):
    """Autograd wrapper for MSDA2D with Triton forward/backward kernels.

    Torch Function Lifecycle:
    ═════════════════════════════════════════════════════════════════

        Training Forward:
            User calls msda2d() → MSDA2DFn.apply()
                ↓
            Forward kernel computes output + saves tensors for backward
                ↓
            Returns output with grad_fn attached

        Backward Pass:
            PyTorch calls MSDA2DFn.backward(grad_output)
                ↓
            Backward kernel computes ∂L/∂value, ∂L/∂sampling, ∂L/∂attn
                ↓
            Returns gradients (matching forward input order)

    Saved Tensors:
        • value (B,K,H,C) - needed for grad_sampling, grad_attn
        • sampling_locations (B,Q,H,L,P,2) - needed for bilinear weights
        • attention_weights (B,Q,H,L,P) - needed for grad_value, grad_sampling
        • spatial_i32 (L,2), lvl_start_i32 (L,) - prepared in forward so the
          backward pass launches without extra host-device synchronization
    """

    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        clamp_grid: bool = True,
        clamp_eps: float = 1e-4,
    ) -> torch.Tensor:
        _validate_inputs_2d(value, value_spatial_shapes, sampling_locations, attention_weights)
        spatial_i32, lvl_start_i32 = _prepare_spatial_2d(value, value_spatial_shapes)
        out = _msda2d_fwd_prepared(
            value,
            sampling_locations,
            attention_weights,
            spatial_i32,
            lvl_start_i32,
            clamp_grid,
            clamp_eps,
        )
        ctx.save_for_backward(
            value, sampling_locations, attention_weights, spatial_i32, lvl_start_i32
        )
        ctx.clamp_grid = clamp_grid
        ctx.clamp_eps = clamp_eps
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        value, sampling_locations, attention_weights, spatial_i32, lvl_start_i32 = ctx.saved_tensors

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        bsz, num_queries, embed_dims = grad_output.shape
        num_heads = value.size(2)
        dim_per_head = value.size(3)
        num_levels = spatial_i32.size(0)
        num_points = attention_weights.size(-1)

        # Reshape grad_output to (B*H, Q, C)
        grad_out_bhqc = (
            grad_output.view(bsz, num_queries, num_heads, dim_per_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        grad_out_bhqc = grad_out_bhqc.view(bsz * num_heads, num_queries, dim_per_head)

        # Allocate gradient tensors
        grad_value = torch.zeros_like(value)
        grad_sampling = torch.zeros_like(sampling_locations)
        grad_attn = torch.zeros_like(attention_weights)

        value_c = value.contiguous()
        sampling_c = sampling_locations.contiguous()
        attn_c = attention_weights.contiguous()

        BLOCK_Q, BLOCK_C = _block_sizes_2d(num_queries, dim_per_head)
        grid = (
            bsz * num_heads,
            triton.cdiv(num_queries, BLOCK_Q),
            triton.cdiv(dim_per_head, BLOCK_C),
        )

        _bwd_kernel_2d[grid](
            value_c,
            sampling_c,
            attn_c,
            grad_out_bhqc,
            spatial_i32,
            lvl_start_i32,
            grad_value,
            grad_sampling,
            grad_attn,
            num_queries,
            num_heads,
            dim_per_head,
            num_levels,
            num_points,
            value_c.stride(0),
            value_c.stride(1),
            value_c.stride(2),
            value_c.stride(3),
            sampling_c.stride(0),
            sampling_c.stride(1),
            sampling_c.stride(2),
            sampling_c.stride(3),
            sampling_c.stride(4),
            sampling_c.stride(5),
            attn_c.stride(0),
            attn_c.stride(1),
            attn_c.stride(2),
            attn_c.stride(3),
            attn_c.stride(4),
            grad_out_bhqc.stride(0),
            grad_out_bhqc.stride(1),
            grad_out_bhqc.stride(2),
            1 if ctx.clamp_grid else 0,
            float(ctx.clamp_eps),
            BLOCK_Q,
            BLOCK_C,
        )

        return grad_value, None, grad_sampling, grad_attn, None, None


def msda2d(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    clamp_grid: bool = True,
    clamp_eps: float = 1e-4,
) -> torch.Tensor:
    """Multi-Scale Deformable Attention 2D - Main entry point with full autograd.

    This is the PRIMARY interface for MSDA2D. Use this in training loops.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Swappable Backend API (drop-in replacement for other MSDA2D ops)  │
    └────────────────────────────────────────────────────────────────────┘

        Standard API:                    This Backend:
        ─────────────                    ─────────────
        msda2d_pytorch(...)      →       msda2d(...)
        msda2d_cuda(...)         →       msda2d(...)

        ✓ Same signature, same semantics
        ✓ Triton-optimized for better memory efficiency
        ✓ Full backward pass via autograd

    Tensor Contracts:
        value: (B, K, H, C)  where K = Σ(H_l×W_l) flattened across levels
        spatial_shapes: (L, 2)  with (H, W) per level
        sampling_locations: (B, Q, H, L, P, 2)  normalized coords ∈ [0,1]²
        attention_weights: (B, Q, H, L, P)  must sum to 1 over (L,P) dimension

        Returns: (B, Q, HxC)

    Args:
        value: Multi-scale feature values (flattened).
        value_spatial_shapes: Per-level spatial dimensions.
        sampling_locations: Learned deformable offsets (normalized).
        attention_weights: Softmax attention weights.
        clamp_grid: Clamp coordinates for AMP stability (recommended: True).
        clamp_eps: Epsilon for clamping (default: 1e-4).

    Returns:
        Attention-weighted output features.

    Raises:
        ValueError: If inputs are not CUDA tensors.
    """
    if not value.is_cuda:
        raise ValueError("MSDA2D Triton kernel requires CUDA tensors. Got CPU tensors instead.")
    return MSDA2DFn.apply(
        value, value_spatial_shapes, sampling_locations, attention_weights, clamp_grid, clamp_eps
    )


__all__ = [
    # Primary API (use this!)
    "msda2d",
    # Forward-only (for inference)
    "msda2d_fwd",
    # Autograd function (advanced users)
    "MSDA2DFn",
]
