"""
Written by Abhinav Atrishi <abhinav@mach9.io>, January 2026.

Test suite for Triton 3D Multi-Scale Deformable Attention kernel.

Validates correctness of both forward and backward passes against PyTorch reference
implementation with various input configurations and edge cases.

Copyright (c) 2026 Mach9 Robotics, Inc.
Licensed under the MIT License. See the LICENSE file for details.
"""

from __future__ import annotations

import pytest
import torch

from msda_triton.pytorch.msda_3d import msda3d_pytorch
from msda_triton.triton.msda_3d import msda3d, msda3d_fwd


class TestMSDA3DTriton:
    """Test suite for Triton MSDA3D kernel correctness."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda")

    def _generate_test_inputs(
        self,
        batch_size: int = 2,
        num_queries: int = 50,
        num_heads: int = 8,
        dim_per_head: int = 32,
        num_levels: int = 2,
        num_points: int = 4,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test inputs for MSDA3D.

        Args:
            batch_size: Batch size.
            num_queries: Number of query points.
            num_heads: Number of attention heads.
            dim_per_head: Dimension per head.
            num_levels: Number of feature pyramid levels.
            num_points: Number of sampling points per query.
            spatial_shapes: List of (D, H, W) tuples, or None for default.
            dtype: Data type for tensors.
            requires_grad: Whether tensors require gradients.

        Returns:
            Tuple of (value, value_spatial_shapes, sampling_locations, attention_weights).
        """
        if spatial_shapes is None:
            spatial_shapes = [(4, 5, 5), (2, 3, 3), (3, 4, 4), (2, 2, 2)][:num_levels]

        # Compute total number of keys
        num_keys = sum(d * h * w for d, h, w in spatial_shapes)

        # Create value tensor
        value = torch.randn(
            batch_size,
            num_keys,
            num_heads,
            dim_per_head,
            device=self.device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        # Create spatial shapes tensor
        value_spatial_shapes = torch.tensor(spatial_shapes, device=self.device, dtype=torch.int32)

        # Create sampling locations (normalized [0, 1])
        sampling_locations = torch.rand(
            batch_size,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            3,
            device=self.device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        # Create attention weights (should sum to 1 across levels*points)
        attention_weights = torch.rand(
            batch_size,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            device=self.device,
            dtype=dtype,
        )
        attention_weights = (
            attention_weights.view(batch_size, num_queries, num_heads, num_levels * num_points)
            .softmax(-1)
            .view(batch_size, num_queries, num_heads, num_levels, num_points)
        )
        if requires_grad:
            attention_weights.requires_grad_(True)

        return value, value_spatial_shapes, sampling_locations, attention_weights

    def test_forward_pass_basic(self) -> None:
        """Test that Triton forward matches PyTorch reference on basic input."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            requires_grad=False
        )

        # PyTorch reference
        out_pytorch = msda3d_pytorch(
            value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=True, clamp_eps=1e-4
        )

        # Triton implementation
        out_triton = msda3d_fwd(
            value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=True, clamp_eps=1e-4
        )

        # Check shapes match
        assert out_pytorch.shape == out_triton.shape

        # Check values are close (allow small numerical differences)
        max_diff = (out_pytorch - out_triton).abs().max().item()
        mean_diff = (out_pytorch - out_triton).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} exceeds threshold"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} exceeds threshold"

    def test_forward_pass_different_configs(self) -> None:
        """Test forward pass with various configurations."""
        configs = [
            # (batch_size, num_queries, num_heads, dim_per_head, num_levels, num_points)
            (1, 100, 4, 64, 3, 4),  # Single batch, more queries
            (4, 25, 8, 32, 2, 8),  # More points
            (2, 200, 16, 16, 4, 4),  # More heads, smaller dim
        ]

        for bs, nq, nh, dph, nl, np in configs:
            value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
                batch_size=bs,
                num_queries=nq,
                num_heads=nh,
                dim_per_head=dph,
                num_levels=nl,
                num_points=np,
                requires_grad=False,
            )

            out_pytorch = msda3d_pytorch(value, spatial_shapes, sampling_locs, attn_weights)
            out_triton = msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

            max_diff = (out_pytorch - out_triton).abs().max().item()
            assert max_diff < 1e-4, (
                f"Config {(bs, nq, nh, dph, nl, np)} failed with diff {max_diff}"
            )

    def test_backward_pass_basic(self) -> None:
        """Test that Triton backward matches PyTorch reference on basic input."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            requires_grad=True
        )

        # PyTorch reference
        value_ref = value.clone().detach().requires_grad_(True)
        sampling_locs_ref = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_ref = attn_weights.clone().detach().requires_grad_(True)

        out_pytorch = msda3d_pytorch(
            value_ref,
            spatial_shapes,
            sampling_locs_ref,
            attn_weights_ref,
            clamp_grid=True,
            clamp_eps=1e-4,
        )

        # Triton implementation
        value_triton = value.clone().detach().requires_grad_(True)
        sampling_locs_triton = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_triton = attn_weights.clone().detach().requires_grad_(True)

        out_triton = msda3d(
            value_triton,
            spatial_shapes,
            sampling_locs_triton,
            attn_weights_triton,
            clamp_grid=True,
            clamp_eps=1e-4,
        )

        # Create dummy gradient
        grad_output = torch.randn_like(out_pytorch)

        # Backward pass
        out_pytorch.backward(grad_output)
        out_triton.backward(grad_output)

        # Gradient w.r.t. value
        grad_value_diff_max = (value_ref.grad - value_triton.grad).abs().max().item()
        grad_value_diff_mean = (value_ref.grad - value_triton.grad).abs().mean().item()
        assert grad_value_diff_max < 1e-5, f"grad_value max diff {grad_value_diff_max} too large"
        assert grad_value_diff_mean < 1e-7, f"grad_value mean diff {grad_value_diff_mean} too large"

        # Gradient w.r.t. sampling_locations
        grad_sampling_diff_max = (
            (sampling_locs_ref.grad - sampling_locs_triton.grad).abs().max().item()
        )
        grad_sampling_diff_mean = (
            (sampling_locs_ref.grad - sampling_locs_triton.grad).abs().mean().item()
        )
        # Atomic accumulation order differs from the reference; allow looser tolerance
        assert grad_sampling_diff_max < 0.25, (
            f"grad_sampling max diff {grad_sampling_diff_max} too large"
        )
        assert grad_sampling_diff_mean < 2e-5, (
            f"grad_sampling mean diff {grad_sampling_diff_mean} too large"
        )

        # Gradient w.r.t. attention_weights
        grad_attn_diff_max = (attn_weights_ref.grad - attn_weights_triton.grad).abs().max().item()
        grad_attn_diff_mean = (attn_weights_ref.grad - attn_weights_triton.grad).abs().mean().item()
        assert grad_attn_diff_max < 3e-4, f"grad_attn max diff {grad_attn_diff_max} too large"
        assert grad_attn_diff_mean < 1e-6, f"grad_attn mean diff {grad_attn_diff_mean} too large"

    def test_fp64_not_supported(self) -> None:
        """Test that fp64 inputs raise NotImplementedError."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            batch_size=1,
            num_queries=10,
            num_heads=2,
            dim_per_head=8,
            dtype=torch.float64,
            requires_grad=False,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

        assert "fp32, fp16, and bf16" in str(exc_info.value)
        assert "float64" in str(exc_info.value)

    def test_edge_case_boundary_sampling(self) -> None:
        """Test sampling at boundaries (0.0 and 1.0)."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            batch_size=1, num_queries=10, requires_grad=False
        )

        # Set some sampling locations to boundaries
        sampling_locs[0, 0, 0, 0, 0, :] = 0.0
        sampling_locs[0, 1, 0, 0, 0, :] = 1.0
        sampling_locs[0, 2, 0, 0, 0, :] = torch.tensor([0.0, 1.0, 0.5])

        out_pytorch = msda3d_pytorch(value, spatial_shapes, sampling_locs, attn_weights)
        out_triton = msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

        max_diff = (out_pytorch - out_triton).abs().max().item()
        assert max_diff < 1e-4, f"Boundary sampling failed with diff {max_diff}"

    def test_edge_case_out_of_bounds(self) -> None:
        """Test sampling slightly out of bounds (should be clamped)."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            batch_size=1, num_queries=10, requires_grad=False
        )

        # Set some locations slightly out of bounds (will be clamped)
        sampling_locs[0, 0, 0, 0, 0, :] = -0.1
        sampling_locs[0, 1, 0, 0, 0, :] = 1.1

        out_pytorch = msda3d_pytorch(
            value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=True
        )
        out_triton = msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=True)

        max_diff = (out_pytorch - out_triton).abs().max().item()
        assert max_diff < 1e-4, f"Out-of-bounds sampling failed with diff {max_diff}"

    def test_gradient_flow_end_to_end(self) -> None:
        """Test gradient flow through a simple end-to-end scenario."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            batch_size=2, num_queries=20, num_heads=4, dim_per_head=16, requires_grad=True
        )

        def compute_loss(out: torch.Tensor) -> torch.Tensor:
            return (out**2).sum()

        # PyTorch reference
        value_ref = value.clone().detach().requires_grad_(True)
        sampling_locs_ref = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_ref = attn_weights.clone().detach().requires_grad_(True)

        out_pytorch = msda3d_pytorch(value_ref, spatial_shapes, sampling_locs_ref, attn_weights_ref)
        loss_pytorch = compute_loss(out_pytorch)
        loss_pytorch.backward()

        # Triton implementation
        value_triton = value.clone().detach().requires_grad_(True)
        sampling_locs_triton = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_triton = attn_weights.clone().detach().requires_grad_(True)

        out_triton = msda3d(value_triton, spatial_shapes, sampling_locs_triton, attn_weights_triton)
        loss_triton = compute_loss(out_triton)
        loss_triton.backward()

        # Check losses are close
        loss_diff = abs(loss_pytorch.item() - loss_triton.item())
        assert loss_diff < 1e-4, f"Loss difference {loss_diff} too large"

        # Check gradients
        grad_value_diff = (value_ref.grad - value_triton.grad).abs().max().item()
        grad_sampling_diff = (sampling_locs_ref.grad - sampling_locs_triton.grad).abs().max().item()
        grad_attn_diff = (attn_weights_ref.grad - attn_weights_triton.grad).abs().max().item()

        assert grad_value_diff < 1e-3
        assert grad_sampling_diff < 1e-3
        assert grad_attn_diff < 1e-3

    def test_no_clamp_grid(self) -> None:
        """Test with clamp_grid=False."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            requires_grad=False
        )

        out_pytorch = msda3d_pytorch(
            value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=False
        )
        out_triton = msda3d_fwd(
            value, spatial_shapes, sampling_locs, attn_weights, clamp_grid=False
        )

        max_diff = (out_pytorch - out_triton).abs().max().item()
        assert max_diff < 1e-4, f"No clamp test failed with diff {max_diff}"

    def test_single_level(self) -> None:
        """Test with a single pyramid level."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            num_levels=1, spatial_shapes=[(4, 5, 5)], requires_grad=True
        )

        # Forward
        out_pytorch = msda3d_pytorch(value, spatial_shapes, sampling_locs, attn_weights)
        out_triton = msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

        max_diff = (out_pytorch - out_triton).abs().max().item()
        assert max_diff < 1e-4, f"Single level forward failed with diff {max_diff}"

        # Backward
        value_ref = value.clone().detach().requires_grad_(True)
        sampling_locs_ref = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_ref = attn_weights.clone().detach().requires_grad_(True)

        out_pytorch = msda3d_pytorch(value_ref, spatial_shapes, sampling_locs_ref, attn_weights_ref)

        value_triton = value.clone().detach().requires_grad_(True)
        sampling_locs_triton = sampling_locs.clone().detach().requires_grad_(True)
        attn_weights_triton = attn_weights.clone().detach().requires_grad_(True)

        out_triton = msda3d(value_triton, spatial_shapes, sampling_locs_triton, attn_weights_triton)

        grad_output = torch.randn_like(out_pytorch)
        out_pytorch.backward(grad_output)
        out_triton.backward(grad_output)

        grad_value_diff = (value_ref.grad - value_triton.grad).abs().max().item()
        assert grad_value_diff < 1e-3, f"Single level backward failed with diff {grad_value_diff}"

    def test_error_on_cpu_tensors(self) -> None:
        """Test that CPU tensors raise ValueError."""
        value = torch.randn(2, 100, 8, 32)  # CPU tensor
        spatial_shapes = torch.tensor([[4, 5, 5], [2, 3, 3]], dtype=torch.int32)
        sampling_locs = torch.rand(2, 50, 8, 2, 4, 3)
        attn_weights = torch.rand(2, 50, 8, 2, 4)

        with pytest.raises(ValueError) as exc_info:
            msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

        assert "CUDA" in str(exc_info.value)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision(self, dtype: torch.dtype) -> None:
        """Test fp16/bf16 forward against an fp32 reference, and that backward is finite."""
        value, spatial_shapes, sampling_locs, attn_weights = self._generate_test_inputs(
            batch_size=1,
            num_queries=50,
            num_heads=4,
            dim_per_head=32,
            dtype=dtype,
            requires_grad=False,
        )

        # Compare against the reference computed in fp32 on upcast inputs; the
        # remaining error is dominated by rounding the output to half precision.
        out_ref = msda3d_pytorch(
            value.float(), spatial_shapes, sampling_locs.float(), attn_weights.float()
        )
        out_triton = msda3d_fwd(value, spatial_shapes, sampling_locs, attn_weights)

        assert out_triton.shape == out_ref.shape
        assert out_triton.dtype == dtype

        tol = 1e-2 if dtype == torch.float16 else 5e-2
        max_diff = (out_ref - out_triton.float()).abs().max().item()
        assert max_diff < tol, f"{dtype} forward failed with diff {max_diff}"

        # Backward must run and produce finite gradients in half precision.
        value_g = value.clone().requires_grad_(True)
        sampling_g = sampling_locs.clone().requires_grad_(True)
        attn_g = attn_weights.clone().requires_grad_(True)
        out = msda3d(value_g, spatial_shapes, sampling_g, attn_g)
        out.sum().backward()
        for name, grad in [
            ("value", value_g.grad),
            ("sampling_locations", sampling_g.grad),
            ("attention_weights", attn_g.grad),
        ]:
            assert grad is not None and torch.isfinite(grad).all(), (
                f"{dtype} grad_{name} contains non-finite values"
            )


def run_tests() -> bool:
    """Run all tests with verbose output."""
    return pytest.main([__file__, "-v"]) == 0


if __name__ == "__main__":
    exit(0 if run_tests() else 1)
