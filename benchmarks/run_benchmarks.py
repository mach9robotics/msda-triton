#!/usr/bin/env python
"""
Main benchmark runner for MSDA-Triton.

Runs all benchmarks and produces a comprehensive comparison report.

Usage:
    uv run python -m benchmarks.run_benchmarks
    uv run python benchmarks/run_benchmarks.py

Options:
    --2d-only     Run only 2D benchmarks
    --3d-only     Run only 3D benchmarks
    --dtype       Data type (fp32, fp16, bf16)
"""

from __future__ import annotations

import argparse
import sys

import torch


def run_3d_benchmarks(dtype: str) -> int:
    """Run 3D MSDA benchmarks."""
    from benchmarks.benchmark_msda3d import BENCHMARK_CONFIGS, print_summary_table, run_benchmark

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    print("\n" + "=" * 80)
    print("3D MULTI-SCALE DEFORMABLE ATTENTION BENCHMARKS")
    print("=" * 80)

    device = torch.device("cuda")
    all_results = []

    for config in BENCHMARK_CONFIGS:
        try:
            results = run_benchmark(config, device, dtype_map[dtype])
            all_results.append(results)
        except Exception as e:
            print(f"ERROR running {config.name}: {e}")
            continue

    if all_results:
        print_summary_table(all_results)

    return 0


def run_2d_benchmarks(dtype: str) -> int:
    """Run 2D MSDA benchmarks."""
    from benchmarks.benchmark_msda2d import BENCHMARK_CONFIGS, print_summary_table, run_benchmark

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    print("\n" + "=" * 80)
    print("2D MULTI-SCALE DEFORMABLE ATTENTION BENCHMARKS")
    print("=" * 80)

    device = torch.device("cuda")
    all_results = []

    for config in BENCHMARK_CONFIGS:
        try:
            results = run_benchmark(config, device, dtype_map[dtype])
            all_results.append(results)
        except Exception as e:
            print(f"ERROR running {config.name}: {e}")
            continue

    if all_results:
        print_summary_table(all_results)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MSDA-Triton benchmarks")
    parser.add_argument(
        "--2d-only", dest="only_2d", action="store_true", help="Run only 2D benchmarks"
    )
    parser.add_argument(
        "--3d-only", dest="only_3d", action="store_true", help="Run only 3D benchmarks"
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

    print("=" * 80)
    print("MSDA-TRITON BENCHMARK SUITE")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Data type: {args.dtype}")

    ret = 0

    if args.only_2d:
        ret = run_2d_benchmarks(args.dtype)
    elif args.only_3d:
        ret = run_3d_benchmarks(args.dtype)
    else:
        # Run both
        ret = run_3d_benchmarks(args.dtype)
        ret = max(ret, run_2d_benchmarks(args.dtype))

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return ret


if __name__ == "__main__":
    sys.exit(main())
