"""
Written by Praveen Venkatesh <praveen@mach9.io>, June 2025.

PyTorch reference implementation of 3D Multi-Scale Deformable Attention (MSDA3D).

Copyright (C) Mach9 Robotics, Inc - All Rights Reserved
Proprietary and confidential
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def msda3d_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    clamp_grid: bool = True,
    clamp_eps: float = 1e-4,
) -> torch.Tensor:
    """Compute 3D Multi-Scale Deformable Attention in PyTorch.

    This implementation uses streaming accumulation to avoid materializing
    large (num_levels x num_points) stacked tensors.

    ┌─────────────────────────────────────────────────────────────┐
    │  Computation Flow (per level l):                            │
    │                                                             │
    │  1. Reshape value[l] → (B*H, C, D, H, W)                    │
    │  2. Prepare grid → (B*H, Q, P, 1, 3)                        │
    │  3. grid_sample (trilinear) → (B*H, C, Q, P)                │
    │  4. Multiply by attn_weights[l] → (B*H, C, Q, P)            │
    │  5. Sum over P → accumulate to output                       │
    └─────────────────────────────────────────────────────────────┘

    Args:
        value: (B, K, H, C_per_head) - Flattened multi-scale features
            where K = Σ(D_l x H_l x W_l) across all levels.
        value_spatial_shapes: (L, 3) - Per-level 3D shapes (D, H, W).
        sampling_locations: (B, Q, H, L, P, 3) - Normalized coords ∈ [0,1]³.
        attention_weights: (B, Q, H, L, P) - Softmax weights.
        clamp_grid: If True, clamp the [-1, 1] grid to [-1+eps, 1-eps].
        clamp_eps: Epsilon margin used when clamping the grid.

    Returns:
        (B, Q, H*C_per_head) - Attention output.
    """
    bs, _, num_heads, dims_per_head = value.shape
    _, num_queries, _, _num_levels, num_points, _ = sampling_locations.shape

    # Split value tensor by spatial shapes (D*H*W for each level)
    spatial_sections = torch.prod(value_spatial_shapes, dim=1).tolist()
    value_list = value.split(spatial_sections, dim=1)

    # Convert sampling locations to grid_sample format [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    if clamp_grid:
        sampling_grids = sampling_grids.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

    # Prepare output accumulator: (bs*num_heads, C_per_head, Q)
    output_accum = value.new_zeros(bs * num_heads, dims_per_head, num_queries)

    # Extract integer spatial shapes once to avoid repeated .item() device syncs
    spatial_shapes_cpu = value_spatial_shapes.cpu()
    spatial_shapes_int = [
        (int(d.item()), int(h.item()), int(w.item())) for d, h, w in spatial_shapes_cpu
    ]

    for level, (d_i, h_i, w_i) in enumerate(spatial_shapes_int):
        # Reshape value for 3D grid sampling
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, dims_per_head, d_i, h_i, w_i)
        )

        # Prepare sampling grid for this level
        # -> (bs*num_heads, num_queries, num_points, 1, 3)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1).unsqueeze(-2)
        )

        # 3D grid sampling with trilinear interpolation
        # output: (bs*num_heads, C, Q, P, 1) -> (bs*num_heads, C, Q, P)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",  # 5D input uses trilinear internally
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1)

        # Gather attention weights for this level and accumulate over points
        # attn: (bs, Q, H, P) -> (bs*H, 1, Q, P)
        attn_w_l = (
            attention_weights[:, :, :, level, :]
            .transpose(1, 2)
            .reshape(bs * num_heads, 1, num_queries, num_points)
        )

        # Accumulate per-level contribution
        # (bs*H, C, Q, P) * (bs*H, 1, Q, P) -> (bs*H, C, Q)
        output_accum = output_accum + (sampling_value_l_ * attn_w_l).sum(-1)

    # Reshape back to (bs, Q, H*C)
    output = output_accum.view(bs, num_heads * dims_per_head, num_queries)
    return output.transpose(1, 2).contiguous()


class MSDA3D(nn.Module):
    """3D Multi-Scale Deformable Attention module.

    ┌─────────────────────────────────────────────────────────────┐
    │  Module Architecture                                        │
    │                                                             │
    │  query ──┬──► sampling_offsets ──┐                          │
    │          │                       ├──► sampling_locations    │
    │          │    reference_points ──┘                          │
    │          │                                                  │
    │          └──► attention_weights ──┐                         │
    │                                   ├──► MSDA3D ──► output    │
    │  value ──► value_proj ────────────┘                         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        embed_dims: The embedding dimension of Attention. Default: 256.
        num_heads: Parallel attention heads. Default: 8.
        num_levels: The number of feature map used in Attention. Default: 4.
        num_points: The number of sampling points per query per head. Default: 4.
        dropout: A Dropout layer on `inp_identity`. Default: 0.1.
        batch_first: Key, Query and Value are shape of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        query_chunk_size: If set, process queries in chunks to reduce memory.
        head_chunk_size: If set, split heads across multiple passes.
        use_checkpoint: If True, use gradient checkpointing.
        clamp_grid: Clamp sampling grid for AMP stability.
        clamp_eps: Epsilon margin for grid clamping.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float = 0.1,
        batch_first: bool = False,
        query_chunk_size: int | None = None,
        head_chunk_size: int | None = None,
        use_checkpoint: bool = False,
        clamp_grid: bool = True,
        clamp_eps: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # Memory/throughput controls
        self.query_chunk_size = query_chunk_size
        self.head_chunk_size = head_chunk_size
        self.use_checkpoint = use_checkpoint
        self.clamp_grid = clamp_grid
        self.clamp_eps = clamp_eps

        # 3D sampling offsets (x, y, z)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def init_weights(self) -> None:
        """Initialize module parameters with optimized 3D sampling patterns.

        Uses Xavier uniform initialization for projection layers and creates
        spatially diverse sampling patterns for deformable attention offsets.
        """
        # Initialize sampling offsets weights and biases
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)

        device = next(self.parameters()).device

        # Initialize 3D sampling patterns with spatial diversity
        grid_init = torch.zeros(self.num_heads, self.num_levels, self.num_points, 3, device=device)

        # Create spatially diverse sampling patterns per head
        for head_idx in range(self.num_heads):
            for point_idx in range(self.num_points):
                angle = (
                    2.0
                    * math.pi
                    * (head_idx * self.num_points + point_idx)
                    / (self.num_heads * self.num_points)
                )
                radius = 0.02 * (point_idx + 1)

                # 3D offset initialization (x, y, z)
                grid_init[head_idx, :, point_idx, 0] = radius * 0.5 * math.cos(angle)
                grid_init[head_idx, :, point_idx, 1] = radius * 0.5 * math.sin(angle)
                grid_init[head_idx, :, point_idx, 2] = radius * 2.0 * math.sin(angle * 2)

        self.sampling_offsets.bias.data = grid_init.view(-1)

        # Initialize attention weights to zero
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # Xavier uniform initialization for value and output projections
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        self._is_init = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for 3D Multi-Scale Deformable Attention.

        Note: Unlike standard attention (Q×K^T×V), deformable attention uses
        queries to predict sampling locations and weights, then samples from
        values. The key parameter is unused.

        Args:
            query: Query tensor. Shape depends on batch_first setting.
            key: Unused. Kept for API compatibility.
            value: Value tensor to sample from. If None, uses query.
            identity: Residual tensor for skip connection. If None, uses query.
            query_pos: Positional encoding for query.
            key_padding_mask: Mask for padding with shape (bs, num_key).
            reference_points: Normalized reference points (bs, num_query, num_levels, 3).
            spatial_shapes: Spatial dimensions per level (num_levels, 3).
            level_start_index: Unused. Kept for API compatibility.

        Returns:
            Output tensor with same shape as query input.

        Raises:
            ValueError: If reference_points last dimension is not 3.
        """
        # Initialize value and identity with defaults if None
        value_tensor: torch.Tensor = query if value is None else value
        identity_tensor: torch.Tensor = query if identity is None else identity

        # Add positional encoding if provided
        query_tensor = query + query_pos if query_pos is not None else query

        # Convert to batch-first format if needed
        if not self.batch_first:
            query_tensor = query_tensor.permute(1, 0, 2)
            value_tensor = value_tensor.permute(1, 0, 2)

        bs, num_query, _ = query_tensor.shape
        _, num_value, _ = value_tensor.shape

        # Type guards for required tensors
        assert spatial_shapes is not None, "spatial_shapes is required"
        assert reference_points is not None, "reference_points is required"

        # Validate 3D spatial shapes match value size
        total_spatial_size = (
            spatial_shapes[:, 0] * spatial_shapes[:, 1] * spatial_shapes[:, 2]
        ).sum()
        assert total_spatial_size == num_value, (
            f"Spatial shapes product {total_spatial_size} doesn't match value size {num_value}"
        )

        value_proj = self.value_proj(value_tensor)
        if key_padding_mask is not None:
            value_proj = value_proj.masked_fill(key_padding_mask[..., None], 0.0)
        value_proj = value_proj.view(bs, num_value, self.num_heads, -1)

        if reference_points.shape[-1] != 3:
            raise ValueError(
                f"Last dim of reference_points must be 3, "
                f"but got {reference_points.shape[-1]} instead."
            )

        # Normalizer for offsets: (L, 3) -> (w, h, d)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 2], spatial_shapes[..., 1], spatial_shapes[..., 0]],
            -1,
        )

        # Determine chunk sizes
        q_chunk_size = int(self.query_chunk_size) if self.query_chunk_size else num_query
        h_chunk_size = (
            int(self.head_chunk_size)
            if (self.head_chunk_size and self.head_chunk_size < self.num_heads)
            else self.num_heads
        )

        chunk_outputs: list[torch.Tensor] = []
        dim_per_head = self.embed_dims // self.num_heads

        for q_start in range(0, num_query, q_chunk_size):
            q_end = min(q_start + q_chunk_size, num_query)
            q_len = q_end - q_start

            # Compute per-chunk offsets and attention weights
            query_chunk = query_tensor[:, q_start:q_end, :]
            sampling_offsets = self.sampling_offsets(query_chunk).view(
                bs, q_len, self.num_heads, self.num_levels, self.num_points, 3
            )
            attention_weights = self.attention_weights(query_chunk).view(
                bs, q_len, self.num_heads, self.num_levels * self.num_points
            )
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(
                bs, q_len, self.num_heads, self.num_levels, self.num_points
            )

            sampling_locations = (
                reference_points[:, q_start:q_end, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

            # Output buffer for this query chunk (bs, q_len, embed_dims)
            out_chunk = query_tensor.new_zeros(bs, q_len, self.embed_dims)

            # Optional head chunking to further bound memory
            for h_start in range(0, self.num_heads, h_chunk_size):
                h_end = min(h_start + h_chunk_size, self.num_heads)
                out_dim_start = h_start * dim_per_head
                out_dim_end = h_end * dim_per_head

                value_h = value_proj[:, :, h_start:h_end, :]
                attn_h = attention_weights[:, :, h_start:h_end, :, :]
                samp_h = sampling_locations[:, :, h_start:h_end, :, :, :]

                def _forward_chunk(
                    v: torch.Tensor, sp: torch.Tensor, sl: torch.Tensor, aw: torch.Tensor
                ) -> torch.Tensor:
                    return msda3d_pytorch(
                        v, sp, sl, aw, clamp_grid=self.clamp_grid, clamp_eps=self.clamp_eps
                    )

                if self.use_checkpoint and self.training:
                    out_h = checkpoint(
                        _forward_chunk,
                        value_h,
                        spatial_shapes,
                        samp_h,
                        attn_h,
                        use_reentrant=False,
                    )
                else:
                    out_h = _forward_chunk(value_h, spatial_shapes, samp_h, attn_h)

                out_chunk[:, :, out_dim_start:out_dim_end] = out_h

            chunk_outputs.append(out_chunk)

        output = torch.cat(chunk_outputs, dim=1)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity_tensor


__all__ = [
    "msda3d_pytorch",
    "MSDA3D",
]
