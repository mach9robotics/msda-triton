"""Benchmark 2D Multi-Scale Deformable Attention (PyTorch vs Triton)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from msda_triton.pytorch.msda_2d import msda2d_pytorch
from msda_triton.triton.msda_2d import msda2d, msda2d_fwd


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark scenario."""

    name: str
    batch_size: int
    num_queries: int
    num_heads: int
    dim_per_head: int
    num_levels: int
    num_points: int
    spatial_shapes: list[tuple[int, int]]

    @property
    def embed_dims(self) -> int:
        return self.num_heads * self.dim_per_head


BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="tiny",
        batch_size=1,
        num_queries=100,
        num_heads=8,
        dim_per_head=32,
        num_levels=4,
        num_points=4,
        spatial_shapes=[(64, 64), (32, 32), (16, 16), (8, 8)],
    ),
    BenchmarkConfig(
        name="small",
        batch_size=1,
        num_queries=900,
        num_heads=8,
        dim_per_head=32,
        num_levels=4,
        num_points=4,
        spatial_shapes=[(80, 80), (40, 40), (20, 20), (10, 10)],
    ),
    BenchmarkConfig(
        name="medium",
        batch_size=2,
        num_queries=900,
        num_heads=8,
        dim_per_head=32,
        num_levels=4,
        num_points=4,
        spatial_shapes=[(100, 134), (50, 67), (25, 34), (13, 17)],
    ),
    BenchmarkConfig(
        name="large",
        batch_size=1,
        num_queries=8100,
        num_heads=8,
        dim_per_head=32,
        num_levels=4,
        num_points=4,
        spatial_shapes=[(200, 200), (100, 100), (50, 50), (25, 25)],
    ),
    BenchmarkConfig(
        name="xl",
        batch_size=1,
        num_queries=32400,
        num_heads=8,
        dim_per_head=32,
        num_levels=4,
        num_points=4,
        spatial_shapes=[(256, 256), (128, 128), (64, 64), (32, 32)],
    ),
]


def generate_inputs(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate benchmark inputs."""
    num_keys = sum(h * w for h, w in config.spatial_shapes)

    value = torch.randn(
        config.batch_size,
        num_keys,
        config.num_heads,
        config.dim_per_head,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    spatial_shapes = torch.tensor(config.spatial_shapes, device=device, dtype=torch.int32)

    sampling_locations = torch.rand(
        config.batch_size,
        config.num_queries,
        config.num_heads,
        config.num_levels,
        config.num_points,
        2,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    attention_weights = torch.rand(
        config.batch_size,
        config.num_queries,
        config.num_heads,
        config.num_levels,
        config.num_points,
        device=device,
        dtype=dtype,
    )
    attention_weights = (
        attention_weights.view(
            config.batch_size,
            config.num_queries,
            config.num_heads,
            config.num_levels * config.num_points,
        )
        .softmax(-1)
        .view(
            config.batch_size,
            config.num_queries,
            config.num_heads,
            config.num_levels,
            config.num_points,
        )
    )
    if requires_grad:
        attention_weights.requires_grad_(True)

    return {
        "value": value,
        "spatial_shapes": spatial_shapes,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
    }


def benchmark_forward(
    fn: Callable,
    inputs: dict[str, torch.Tensor],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> dict[str, float]:
    """Benchmark forward pass."""
    device = inputs["value"].device

    for _ in range(warmup_iters):
        _ = fn(
            inputs["value"],
            inputs["spatial_shapes"],
            inputs["sampling_locations"],
            inputs["attention_weights"],
        )
    torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mem = torch.cuda.memory_allocated(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_iters):
        _ = fn(
            inputs["value"],
            inputs["spatial_shapes"],
            inputs["sampling_locations"],
            inputs["attention_weights"],
        )
    end_event.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start_event.elapsed_time(end_event) / benchmark_iters
    peak_mem = torch.cuda.max_memory_allocated(device)
    extra_mem_mb = (peak_mem - baseline_mem) / (1024**2)

    return {"time_ms": elapsed_ms, "memory_mb": extra_mem_mb}


def benchmark_backward(
    fn: Callable,
    inputs: dict[str, torch.Tensor],
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> dict[str, float]:
    """Benchmark backward pass."""
    device = inputs["value"].device

    def run_forward_backward():
        value = inputs["value"].clone().detach().requires_grad_(True)
        sampling = inputs["sampling_locations"].clone().detach().requires_grad_(True)
        attn = inputs["attention_weights"].clone().detach().requires_grad_(True)

        out = fn(value, inputs["spatial_shapes"], sampling, attn)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        return value.grad, sampling.grad, attn.grad

    for _ in range(warmup_iters):
        _ = run_forward_backward()
    torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mem = torch.cuda.memory_allocated(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_iters):
        _ = run_forward_backward()
    end_event.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start_event.elapsed_time(end_event) / benchmark_iters
    peak_mem = torch.cuda.max_memory_allocated(device)
    extra_mem_mb = (peak_mem - baseline_mem) / (1024**2)

    return {"time_ms": elapsed_ms, "memory_mb": extra_mem_mb}


def run_benchmark(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype) -> dict:
    """Run all benchmarks for one configuration."""
    print(f"\n{'=' * 70}")
    print(f"Config: {config.name}")
    print(
        f"  Queries: {config.num_queries}, Heads: {config.num_heads}, Levels: {config.num_levels}"
    )
    print(f"  Spatial shapes: {config.spatial_shapes}")
    print(f"  Dtype: {dtype}")
    print(f"{'=' * 70}")

    results: dict[str, Any] = {"config": config.name}

    inputs = generate_inputs(config, device, dtype)
    inputs_grad = generate_inputs(config, device, dtype, requires_grad=True)

    print("\nForward pass:")

    pytorch_fwd = benchmark_forward(msda2d_pytorch, inputs)
    print(f"  PyTorch: {pytorch_fwd['time_ms']:.3f} ms, {pytorch_fwd['memory_mb']:.1f} MB")
    results["pytorch_fwd_ms"] = pytorch_fwd["time_ms"]
    results["pytorch_fwd_mb"] = pytorch_fwd["memory_mb"]

    triton_fwd = benchmark_forward(msda2d_fwd, inputs)
    print(f"  Triton:  {triton_fwd['time_ms']:.3f} ms, {triton_fwd['memory_mb']:.1f} MB")
    results["triton_fwd_ms"] = triton_fwd["time_ms"]
    results["triton_fwd_mb"] = triton_fwd["memory_mb"]

    speedup = pytorch_fwd["time_ms"] / max(triton_fwd["time_ms"], 1e-6)
    print(f"  Speedup: {speedup:.2f}x")
    results["fwd_speedup"] = speedup

    print("\nForward + Backward pass:")

    pytorch_bwd = benchmark_backward(msda2d_pytorch, inputs_grad)
    print(f"  PyTorch: {pytorch_bwd['time_ms']:.3f} ms, {pytorch_bwd['memory_mb']:.1f} MB")
    results["pytorch_bwd_ms"] = pytorch_bwd["time_ms"]
    results["pytorch_bwd_mb"] = pytorch_bwd["memory_mb"]

    triton_bwd = benchmark_backward(msda2d, inputs_grad)
    print(f"  Triton:  {triton_bwd['time_ms']:.3f} ms, {triton_bwd['memory_mb']:.1f} MB")
    results["triton_bwd_ms"] = triton_bwd["time_ms"]
    results["triton_bwd_mb"] = triton_bwd["memory_mb"]

    bwd_speedup = pytorch_bwd["time_ms"] / max(triton_bwd["time_ms"], 1e-6)
    print(f"  Speedup: {bwd_speedup:.2f}x")
    results["bwd_speedup"] = bwd_speedup

    return results


def print_summary_table(all_results: list[dict]) -> None:
    """Print summary table of all benchmark results."""
    print("\n" + "=" * 80)
    print("SUMMARY: 2D MSDA Benchmarks (PyTorch vs Triton)")
    print("=" * 80)

    print(f"{'Config':<10} | {'Forward (ms)':<24} | {'Fwd+Bwd (ms)':<24} | {'Speedup':<12}")
    print(
        f"{'':10} | {'PyTorch':<11} {'Triton':<11} | {'PyTorch':<11} {'Triton':<11} | {'Fwd':<5} {'Bwd':<5}"
    )
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['config']:<10} | "
            f"{r['pytorch_fwd_ms']:>10.2f} {r['triton_fwd_ms']:>10.2f} | "
            f"{r['pytorch_bwd_ms']:>10.2f} {r['triton_bwd_ms']:>10.2f} | "
            f"{r['fwd_speedup']:>4.1f}x {r['bwd_speedup']:>4.1f}x"
        )

    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MSDA2D implementations")
    parser.add_argument(
        "--config",
        type=str,
        choices=[c.name for c in BENCHMARK_CONFIGS] + ["all"],
        default="all",
        help="Benchmark configuration to run",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type for benchmarking",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    device = torch.device("cuda")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")

    if args.config == "all":
        configs = BENCHMARK_CONFIGS
    else:
        configs = [c for c in BENCHMARK_CONFIGS if c.name == args.config]

    all_results = []
    for config in configs:
        try:
            results = run_benchmark(config, device, dtype)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR running {config.name}: {e}")
            continue

    if all_results:
        print_summary_table(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
