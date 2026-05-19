#!/usr/bin/env python3
"""Benchmark harness for dispatch-to-combine MoE experiments.

The script owns the experiment contract and timing window. Real kernels are
plugged in through an importable Python entrypoint, because this repository is
mostly a design and reference workspace.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULTS = {
    "hidden_size": 7168,
    "moe_intermediate_size": 3072,
    "num_experts": 384,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 61,
    "num_hash_layers": 3,
    "max_position_embeddings": 1048576,
}


CASE_ENV = {
    "A": {
        "OMNI_NPU_FUSED_DECODE_MOE": "0",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
    },
    "B": {
        "OMNI_NPU_FUSED_DECODE_MOE": "1",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "2",
    },
    "C": {},
}


@dataclass(frozen=True)
class BenchConfig:
    case: str
    backend: str
    world_size: int
    m: int
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    warmup: int
    repeat: int
    output: str
    entrypoint: str
    deepgemm_test: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MoE dispatch-to-combine closed interval."
    )
    parser.add_argument("--case", choices=("A", "B", "C"), required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "npu", "cuda"),
        default="auto",
        help="A and B default to npu, C defaults to cuda.",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--m", type=int, default=24, help="Local M dimension.")
    parser.add_argument("--hidden-size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=DEFAULTS["moe_intermediate_size"],
    )
    parser.add_argument("--num-experts", type=int, default=DEFAULTS["num_experts"])
    parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=DEFAULTS["num_experts_per_tok"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=("plan", "callable", "deepgemm"),
        default="plan",
        help="plan prints config only, callable imports a Python op wrapper, deepgemm calls DeepGEMM test_mega_moe.py.",
    )
    parser.add_argument(
        "--entrypoint",
        default="",
        help="Import path in module:function form. The function receives BenchConfig and returns a zero-arg callable.",
    )
    parser.add_argument(
        "--deepgemm-test",
        default="external_repos/DeepGEMM/tests/test_mega_moe.py",
        help="Path used by --mode deepgemm for case C.",
    )
    parser.add_argument("--output", default="moe_dispatch_combine_results.csv")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def resolve_backend(case: str, backend: str) -> str:
    if backend != "auto":
        return backend
    return "cuda" if case == "C" else "npu"


def build_config(args: argparse.Namespace) -> BenchConfig:
    backend = resolve_backend(args.case, args.backend)
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if args.m < 1:
        raise ValueError("--m must be >= 1")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    return BenchConfig(
        case=args.case,
        backend=backend,
        world_size=args.world_size,
        m=args.m,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
        entrypoint=args.entrypoint,
        deepgemm_test=args.deepgemm_test,
    )


def apply_case_env(case: str) -> None:
    for key, value in CASE_ENV[case].items():
        os.environ[key] = value


def import_entrypoint(entrypoint: str) -> Callable[[BenchConfig], Callable[[], Any]]:
    if ":" not in entrypoint:
        raise ValueError("--entrypoint must use module:function format")
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func


def get_timer(backend: str):
    import torch

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but torch.cuda is not available")
        return torch.cuda.Event, torch.cuda.synchronize
    if backend == "npu":
        import torch_npu  # noqa: F401

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU backend requested, but torch.npu is not available")
        return torch.npu.Event, torch.npu.synchronize
    raise ValueError(f"Unsupported backend: {backend}")


def measure_callable(config: BenchConfig) -> list[float]:
    if not config.entrypoint:
        raise ValueError("--mode callable requires --entrypoint")
    apply_case_env(config.case)
    builder = import_entrypoint(config.entrypoint)
    operation = builder(config)
    if not callable(operation):
        raise TypeError("entrypoint must return a zero-arg callable")

    event_cls, synchronize = get_timer(config.backend)
    for _ in range(config.warmup):
        operation()
    synchronize()

    samples_ms: list[float] = []
    for _ in range(config.repeat):
        start = event_cls(enable_timing=True)
        end = event_cls(enable_timing=True)
        start.record()
        operation()
        end.record()
        synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return samples_ms


def run_deepgemm(config: BenchConfig) -> list[float]:
    if config.case != "C":
        raise ValueError("--mode deepgemm is only valid for case C")
    script = Path(config.deepgemm_test)
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--num-processes",
        str(config.world_size),
        "--num-max-tokens-per-rank",
        str(config.m),
        "--num-tokens",
        str(config.m),
        "--hidden",
        str(config.hidden_size),
        "--intermediate-hidden",
        str(config.moe_intermediate_size),
        "--num-experts",
        str(config.num_experts),
        "--num-topk",
        str(config.num_experts_per_tok),
    ]
    started = time.perf_counter()
    completed = subprocess.run(cmd, check=False, text=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(f"DeepGEMM test failed with exit code {completed.returncode}")
    return [elapsed_ms]


def summarize(samples_ms: Iterable[float]) -> dict[str, float]:
    samples = list(samples_ms)
    samples_sorted = sorted(samples)
    idx_p50 = int((len(samples_sorted) - 1) * 0.50)
    idx_p90 = int((len(samples_sorted) - 1) * 0.90)
    idx_p99 = int((len(samples_sorted) - 1) * 0.99)
    return {
        "count": float(len(samples_sorted)),
        "mean_ms": statistics.fmean(samples_sorted),
        "median_ms": samples_sorted[idx_p50],
        "min_ms": samples_sorted[0],
        "max_ms": samples_sorted[-1],
        "p90_ms": samples_sorted[idx_p90],
        "p99_ms": samples_sorted[idx_p99],
    }


def write_csv(config: BenchConfig, samples_ms: list[float], summary: dict[str, float]) -> None:
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "backend",
                "world_size",
                "m",
                "hidden_size",
                "moe_intermediate_size",
                "num_experts",
                "num_experts_per_tok",
                "sample_ms",
                "count",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "p90_ms",
                "p99_ms",
            ],
        )
        if not exists:
            writer.writeheader()
        for sample in samples_ms:
            row = asdict(config)
            row.pop("warmup")
            row.pop("repeat")
            row.pop("output")
            row.pop("entrypoint")
            row.pop("deepgemm_test")
            row["sample_ms"] = sample
            row.update(summary)
            writer.writerow(row)


def print_plan(config: BenchConfig) -> None:
    payload = {
        "config": asdict(config),
        "case_env": CASE_ENV[config.case],
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "contract": {
            "timing_window": "dispatch-to-combine closed interval",
            "metric": "elapsed milliseconds only",
            "default_world_size": 8,
            "default_m": 24,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.mode == "plan":
        print_plan(config)
        return 0

    samples_ms = (
        run_deepgemm(config)
        if args.mode == "deepgemm"
        else measure_callable(config)
    )
    summary = summarize(samples_ms)
    write_csv(config, samples_ms, summary)
    payload = {"config": asdict(config), "summary": summary, "samples_ms": samples_ms}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"case={config.case} backend={config.backend} "
            f"mean_ms={summary['mean_ms']:.6f} median_ms={summary['median_ms']:.6f} "
            f"output={config.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
