#!/usr/bin/env python3
"""Run the vLLM-Ascend fused MoE A3 benchmark through torchrun."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from benchmark_moe_dispatch_combine import (
    DEFAULTS,
    BenchConfig,
    is_primary_rank,
    measure_callable,
    summarize,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM-Ascend dispatch_gmm_combine_decode on Ascend A3."
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--m", type=int, default=24)
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="results/vllm_ascend_fused_moe_a3.csv")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--no-torchrun",
        action="store_true",
        help="Internal flag used after this script re-execs under torchrun.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if _should_launch_torchrun(args):
        return _launch_torchrun(args)

    os.environ["VLLM_ASCEND_ENABLE_FUSED_MC2"] = "2"
    config = BenchConfig(
        case="B",
        backend="npu",
        world_size=args.world_size,
        m=args.m,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
        entrypoint="vllm_ascend_fused_moe_adapter:build_case",
        deepgemm_test="",
    )
    samples_ms = measure_callable(config)
    summary = summarize(samples_ms)
    if is_primary_rank():
        write_csv(config, samples_ms, summary)
        payload = {"config": asdict(config), "summary": summary, "samples_ms": samples_ms}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"case=B backend=npu op=dispatch_gmm_combine_decode "
                f"mean_ms={summary['mean_ms']:.6f} "
                f"median_ms={summary['median_ms']:.6f} output={config.output}"
            )
    return 0


def _should_launch_torchrun(args: argparse.Namespace) -> bool:
    return (
        not args.no_torchrun
        and args.world_size > 1
        and "RANK" not in os.environ
        and "WORLD_SIZE" not in os.environ
    )


def _launch_torchrun(args: argparse.Namespace) -> int:
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is not found in PATH")
    script = Path(__file__).resolve()
    forwarded = [
        arg
        for arg in sys.argv[1:]
        if arg != "--no-torchrun"
    ]
    cmd = [
        torchrun,
        "--no-python",
        "--nproc-per-node",
        str(args.world_size),
        "--",
        sys.executable,
        str(script),
        "--no-torchrun",
        *forwarded,
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
