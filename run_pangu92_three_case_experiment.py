#!/usr/bin/env python3
"""Three-case Pangu 92B MoE dispatch-combine experiment runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import run_pangu92_decode_dispatch_combine_benchmark as bench


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "pangu92_moe_weights_sync"
INPUT_DIR = ARTIFACT_DIR / "inputs"
OUTPUT_DIR = ARTIFACT_DIR / "outputs"
RESULT_DIR = ARTIFACT_DIR / "results"
PLOT_DIR = ARTIFACT_DIR / "plots"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

TENSOR_NAMES = [
    "hidden_states",
    "topk_ids",
    "topk_weights",
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "quant_scale",
]
CASE_NAMES = ["pangu_chain", "vllm_base", "vllm_modified"]
DETERMINISM_RTOL = 1e-4
DETERMINISM_ATOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pangu 92B three-case MoE decode experiment."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Overwrite and generate fixed artifacts.")
    add_config_args(prepare)
    prepare.add_argument("--model-path", default="")
    prepare.add_argument("--synthetic-fallback", action="store_true")
    prepare.add_argument("--layer-index", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=20260521)

    sub.add_parser("verify-artifacts", help="Verify artifact metadata and tensor hashes.")

    run_case = sub.add_parser("run-case", help="Run one case with fixed artifacts.")
    run_case.add_argument("--case-name", choices=CASE_NAMES, required=True)
    run_case.add_argument("--op-path", choices=["pangu", "vllm"], required=True)
    run_case.add_argument("--determinism-repeat", type=int, default=3)
    run_case.add_argument("--dump-output", action="store_true")
    run_case.add_argument("--warmup", type=int, default=20)
    run_case.add_argument("--repeat", type=int, default=100)
    run_case.add_argument(
        "--output",
        default=str(RESULT_DIR / "timing.csv"),
        help="CSV path for timing samples.",
    )
    run_case.add_argument("--no-torchrun", action="store_true")

    check = sub.add_parser("check-outputs", help="Compare case outputs with golden.")
    check.add_argument("--golden-case", default="pangu_chain")
    check.add_argument("--case", action="append", required=True)
    check.add_argument("--rtol", type=float, default=1e-2)
    check.add_argument("--atol", type=float, default=1e-2)

    plot = sub.add_parser("plot", help="Generate PNG plots and summary CSV.")
    plot.add_argument(
        "--input",
        default=str(RESULT_DIR / "timing.csv"),
        help="Timing CSV path.",
    )
    return parser.parse_args()


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=bench.PANGU92_DEFAULTS["hidden_size"])
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=bench.PANGU92_DEFAULTS["moe_intermediate_size"],
    )
    parser.add_argument("--num-experts", type=int, default=bench.PANGU92_DEFAULTS["num_experts"])
    parser.add_argument("--top-k", type=int, default=bench.PANGU92_DEFAULTS["top_k"])
    parser.add_argument("--quant-mode", type=int, default=bench.PANGU92_DEFAULTS["quant_mode"])


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "verify-artifacts":
        return cmd_verify_artifacts()
    if args.command == "run-case":
        if should_launch_torchrun(args):
            return launch_torchrun(args)
        return cmd_run_case(args)
    if args.command == "check-outputs":
        return cmd_check_outputs(args)
    if args.command == "plot":
        return cmd_plot(args)
    raise ValueError(f"unknown command {args.command}")


def cmd_prepare(args: argparse.Namespace) -> int:
    import torch

    cfg = config_from_prepare_args(args)
    validate_artifact_config(cfg)
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    INPUT_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir()
    RESULT_DIR.mkdir()
    PLOT_DIR.mkdir()

    cfg["weight_source"] = resolve_weight_source(torch, cfg, args)
    try:
        rank_hashes = write_rank_artifacts(torch, cfg, args)
    except Exception as exc:
        if cfg["weight_source"] != "real" or not args.synthetic_fallback:
            raise
        print(f"warning: failed to build real artifacts, fallback to synthetic: {exc}")
        shutil.rmtree(INPUT_DIR)
        INPUT_DIR.mkdir()
        cfg["weight_source"] = "synthetic"
        rank_hashes = write_rank_artifacts(torch, cfg, args)
    metadata = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_dir": str(ARTIFACT_DIR),
        "config": cfg,
        "rank_hashes": rank_hashes,
    }
    metadata["artifact_hash"] = artifact_hash(metadata)
    write_json(METADATA_PATH, metadata)
    print(f"prepared artifacts: {ARTIFACT_DIR}")
    print(f"artifact_hash={metadata['artifact_hash']}")
    return 0


def write_rank_artifacts(
    torch,
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    rank_hashes: dict[str, dict[str, str]] = {}
    for rank in range(cfg["world_size"]):
        tensors = build_rank_tensors(torch, cfg, args, rank)
        hashes = {name: tensor_sha256(tensors[name]) for name in TENSOR_NAMES}
        rank_hashes[str(rank)] = hashes
        torch.save(
            {
                "rank": rank,
                "config": cfg,
                "tensors": tensors,
                "hashes": hashes,
            },
            rank_input_path(rank),
        )
    return rank_hashes


def config_from_prepare_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "world_size": args.world_size,
        "hidden_size": args.hidden_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "quant_mode": args.quant_mode,
        "model_path": args.model_path,
        "synthetic_fallback": bool(args.synthetic_fallback),
        "layer_index": args.layer_index,
        "seed": args.seed,
        "weight_source": "",
    }


def validate_artifact_config(cfg: dict[str, Any]) -> None:
    if cfg["batch_size"] < 1:
        raise ValueError("--batch-size must be >= 1")
    if cfg["world_size"] < 1:
        raise ValueError("--world-size must be >= 1")
    if cfg["num_experts"] % cfg["world_size"] != 0:
        raise ValueError("--num-experts must be divisible by --world-size")
    if cfg["top_k"] < 1 or cfg["top_k"] > cfg["num_experts"]:
        raise ValueError("--top-k must satisfy 1 <= top_k <= num_experts")
    local_experts = cfg["num_experts"] // cfg["world_size"]
    if local_experts > 24:
        raise ValueError(
            "local experts must be <= 24 for fused op precheck, got "
            f"{local_experts}"
        )


def resolve_weight_source(torch, cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if not args.model_path:
        if args.synthetic_fallback:
            return "synthetic"
        raise ValueError("--model-path is required unless --synthetic-fallback is set")
    try:
        bench.load_weight_tensors(Path(cfg["model_path"]), int(cfg["layer_index"]))
        return "real"
    except Exception:
        if not args.synthetic_fallback:
            raise
        return "synthetic"


def build_rank_tensors(torch, cfg: dict[str, Any], args: argparse.Namespace, rank: int) -> dict[str, Any]:
    if cfg["weight_source"] == "real":
        weights = build_real_rank_weights(torch, cfg, rank)
    elif cfg["weight_source"] == "synthetic":
        weights = build_synthetic_rank_weights(torch, cfg, rank)
    else:
        raise ValueError(f"unknown weight_source={cfg['weight_source']}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["seed"]) + rank)
    hidden_states = torch.randn(
        (cfg["batch_size"], cfg["hidden_size"]),
        generator=generator,
        dtype=torch.bfloat16,
    )
    topk_ids = torch.arange(
        rank * cfg["batch_size"] * cfg["top_k"],
        (rank + 1) * cfg["batch_size"] * cfg["top_k"],
        dtype=torch.int32,
    ).view(cfg["batch_size"], cfg["top_k"]) % cfg["num_experts"]
    raw_weights = torch.rand(
        (cfg["batch_size"], cfg["top_k"]),
        generator=generator,
        dtype=torch.float32,
    )
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    quant_scale = torch.ones(
        (cfg["num_experts"] // cfg["world_size"], cfg["moe_intermediate_size"]),
        dtype=torch.float32,
    )
    return {
        "hidden_states": hidden_states.contiguous(),
        "topk_ids": topk_ids.contiguous(),
        "topk_weights": topk_weights.contiguous(),
        "quant_scale": quant_scale.contiguous(),
        **weights,
    }


def build_synthetic_rank_weights(torch, cfg: dict[str, Any], rank: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["seed"]) + 100000 + rank)
    local_experts = cfg["num_experts"] // cfg["world_size"]
    w13_weight = torch.randint(
        -8,
        8,
        (local_experts, cfg["hidden_size"], 2 * cfg["moe_intermediate_size"]),
        generator=generator,
        dtype=torch.int8,
    )
    w2_weight = torch.randint(
        -8,
        8,
        (local_experts, cfg["moe_intermediate_size"], cfg["hidden_size"]),
        generator=generator,
        dtype=torch.int8,
    )
    w13_scale = torch.ones((local_experts, 2 * cfg["moe_intermediate_size"]), dtype=torch.float32)
    w2_scale = torch.ones((local_experts, cfg["hidden_size"]), dtype=torch.bfloat16)
    return {
        "w13_weight": w13_weight.contiguous(),
        "w2_weight": w2_weight.contiguous(),
        "w13_weight_scale": w13_scale.contiguous(),
        "w2_weight_scale": w2_scale.contiguous(),
    }


def build_real_rank_weights(torch, cfg: dict[str, Any], rank: int) -> dict[str, Any]:
    tensor_map = bench.load_weight_tensors(Path(cfg["model_path"]), int(cfg["layer_index"]))
    local_experts = cfg["num_experts"] // cfg["world_size"]
    w13_weight = bench.prepare_weight(
        torch,
        tensor_map["w13_weight"],
        rank,
        local_experts,
        cfg["num_experts"],
        (cfg["hidden_size"], 2 * cfg["moe_intermediate_size"]),
        "w13_weight",
    ).to(dtype=torch.int8)
    w2_weight = bench.prepare_weight(
        torch,
        tensor_map["w2_weight"],
        rank,
        local_experts,
        cfg["num_experts"],
        (cfg["moe_intermediate_size"], cfg["hidden_size"]),
        "w2_weight",
    ).to(dtype=torch.int8)
    w13_scale = bench.prepare_scale(
        torch,
        tensor_map["w13_weight_scale"],
        rank,
        local_experts,
        cfg["num_experts"],
        2 * cfg["moe_intermediate_size"],
        "w13_weight_scale",
    ).to(dtype=torch.float32)
    w2_scale = bench.prepare_scale(
        torch,
        tensor_map["w2_weight_scale"],
        rank,
        local_experts,
        cfg["num_experts"],
        cfg["hidden_size"],
        "w2_weight_scale",
    ).to(dtype=torch.bfloat16)
    return {
        "w13_weight": w13_weight.contiguous(),
        "w2_weight": w2_weight.contiguous(),
        "w13_weight_scale": w13_scale.contiguous(),
        "w2_weight_scale": w2_scale.contiguous(),
    }


def cmd_verify_artifacts() -> int:
    import torch

    metadata = load_metadata()
    cfg = metadata["config"]
    validate_artifact_config(cfg)
    failures = []
    for rank in range(cfg["world_size"]):
        payload = torch.load(rank_input_path(rank), map_location="cpu", weights_only=False)
        hashes = payload["hashes"]
        tensors = payload["tensors"]
        for name in TENSOR_NAMES:
            expected = metadata["rank_hashes"][str(rank)][name]
            actual = tensor_sha256(tensors[name])
            if hashes[name] != expected or actual != expected:
                failures.append(f"rank={rank} tensor={name}")
    if failures:
        raise RuntimeError("artifact verification failed: " + ", ".join(failures))
    print(f"verified artifacts: {ARTIFACT_DIR}")
    print(f"artifact_hash={metadata['artifact_hash']}")
    return 0


def cmd_run_case(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True
    metadata = load_metadata()
    cfg_dict = metadata["config"]
    cfg = make_bench_config(cfg_dict, args)
    bench.validate_config(cfg)
    bench.ensure_vllm_custom_op(torch, cfg)
    runtime = bench.init_distributed(torch, dist, cfg.world_size)
    if runtime.world_size != cfg.world_size:
        raise ValueError(
            f"artifact world_size={cfg.world_size} does not match WORLD_SIZE={runtime.world_size}"
        )
    device = torch.device(f"npu:{runtime.local_rank}")
    tensors = load_rank_artifact(torch, metadata, runtime.rank)
    weights = bench.Weights(
        w13_weight=bench.to_nz(torch_npu, tensors["w13_weight"].to(device=device, dtype=torch.int8)),
        w2_weight=bench.to_nz(torch_npu, tensors["w2_weight"].to(device=device, dtype=torch.int8)),
        w13_weight_scale=tensors["w13_weight_scale"].to(device=device, dtype=torch.float32),
        w2_weight_scale=tensors["w2_weight_scale"].to(device=device, dtype=torch.bfloat16),
        source=str(cfg_dict["weight_source"]),
    )
    hidden_states = tensors["hidden_states"].to(device=device, dtype=torch.bfloat16)
    topk_ids = tensors["topk_ids"].to(device=device, dtype=torch.int32)
    topk_weights = tensors["topk_weights"].to(device=device, dtype=torch.float32)
    quant_scale = tensors["quant_scale"].to(device=device, dtype=torch.float32)
    torch.npu.synchronize()

    operations = bench.build_operations(
        torch,
        torch_npu,
        cfg,
        runtime,
        weights,
        hidden_states,
        topk_ids,
        topk_weights,
        quant_scale,
    )
    operation_name = "pangu_chain" if args.op_path == "pangu" else "vllm_fused"
    operation = operations[operation_name]
    determinism = check_determinism(torch, operation, args.determinism_repeat)
    if not determinism["passed"]:
        raise RuntimeError(
            "determinism check failed: "
            f"max_abs_diff={determinism['max_abs_diff']:.8g}, "
            f"max_rel_diff={determinism['max_rel_diff']:.8g}"
        )
    if args.dump_output:
        save_case_output(
            torch,
            args.case_name,
            runtime.rank,
            determinism["reference_output"],
            metadata,
        )

    samples_ms = bench.measure(torch, cfg, operation)
    summary = bench.summarize(samples_ms)
    if bench.is_primary_rank(torch):
        append_timing_csv(args, cfg, metadata, samples_ms, summary, determinism)
        print(
            f"case={args.case_name} op_path={args.op_path} "
            f"median_ms={summary['median_ms']:.6f} output={args.output}"
        )
    return 0


def should_launch_torchrun(args: argparse.Namespace) -> bool:
    if args.no_torchrun or "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return False
    metadata = load_metadata()
    return int(metadata["config"]["world_size"]) > 1


def launch_torchrun(args: argparse.Namespace) -> int:
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is not found in PATH")
    metadata = load_metadata()
    script = Path(__file__).resolve()
    forwarded = [arg for arg in sys.argv[1:] if arg != "--no-torchrun"]
    cmd = [
        torchrun,
        "--no-python",
        "--nproc-per-node",
        str(metadata["config"]["world_size"]),
        "--",
        sys.executable,
        str(script),
        *forwarded,
        "--no-torchrun",
    ]
    return subprocess.run(cmd, check=False).returncode


def make_bench_config(cfg: dict[str, Any], args: argparse.Namespace) -> bench.BenchConfig:
    return bench.BenchConfig(
        batch_size=int(cfg["batch_size"]),
        world_size=int(cfg["world_size"]),
        hidden_size=int(cfg["hidden_size"]),
        moe_intermediate_size=int(cfg["moe_intermediate_size"]),
        num_experts=int(cfg["num_experts"]),
        top_k=int(cfg["top_k"]),
        quant_mode=int(cfg["quant_mode"]),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        output=str(args.output),
        model_path=str(cfg.get("model_path", "")),
        synthetic_fallback=bool(cfg.get("synthetic_fallback", False)),
        layer_index=int(cfg.get("layer_index", 0)),
        seed=int(cfg.get("seed", 0)),
        op_path=args.op_path,
        check_numerics=False,
        rtol=DETERMINISM_RTOL,
        atol=DETERMINISM_ATOL,
    )


def load_rank_artifact(torch, metadata: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = torch.load(rank_input_path(rank), map_location="cpu", weights_only=False)
    tensors = payload["tensors"]
    failures = []
    for name in TENSOR_NAMES:
        expected = metadata["rank_hashes"][str(rank)][name]
        actual = tensor_sha256(tensors[name])
        if actual != expected:
            failures.append(name)
    if failures:
        raise RuntimeError(f"rank {rank} artifact hash mismatch: {failures}")
    return tensors


def check_determinism(torch, operation, repeat: int) -> dict[str, Any]:
    repeat = max(1, int(repeat))
    reference = bench.unwrap_output(bench.run_once_synced(torch, operation))
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for _ in range(repeat - 1):
        output = bench.unwrap_output(bench.run_once_synced(torch, operation))
        metrics = compare_tensors(
            torch,
            reference,
            output,
            DETERMINISM_RTOL,
            DETERMINISM_ATOL,
        )
        max_abs = max(max_abs, float(metrics["max_abs_diff"]))
        max_rel = max(max_rel, float(metrics["max_rel_diff"]))
        passed = passed and bool(metrics["allclose"])
    max_abs = bench.reduce_max_scalar(torch, max_abs)
    max_rel = bench.reduce_max_scalar(torch, max_rel)
    failed = bench.reduce_max_scalar(torch, 0.0 if passed else 1.0)
    return {
        "passed": failed == 0.0,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "reference_output": reference,
    }


def compare_tensors(torch, expected, actual, rtol: float, atol: float) -> dict[str, Any]:
    if tuple(expected.shape) != tuple(actual.shape):
        return {
            "allclose": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
        }
    expected_f32 = expected.to(torch.float32)
    actual_f32 = actual.to(torch.float32)
    abs_diff = (expected_f32 - actual_f32).abs()
    denom = torch.maximum(
        expected_f32.abs(),
        torch.tensor(atol, dtype=torch.float32, device=expected_f32.device),
    )
    rel_diff = abs_diff / denom
    return {
        "allclose": bool(torch.allclose(expected_f32, actual_f32, rtol=rtol, atol=atol)),
        "max_abs_diff": float(abs_diff.max().detach().cpu().item()),
        "mean_abs_diff": float(abs_diff.mean().detach().cpu().item()),
        "max_rel_diff": float(rel_diff.max().detach().cpu().item()),
    }


def save_case_output(
    torch,
    case_name: str,
    rank: int,
    output,
    metadata: dict[str, Any],
) -> None:
    case_dir = OUTPUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case_name": case_name,
            "rank": rank,
            "artifact_hash": metadata["artifact_hash"],
            "config": output_config_from_metadata(metadata),
            "output": output.detach().cpu().contiguous(),
            "output_hash": tensor_sha256(output.detach().cpu().contiguous()),
        },
        case_dir / f"rank{rank}.pt",
    )


def append_timing_csv(
    args: argparse.Namespace,
    cfg: bench.BenchConfig,
    metadata: dict[str, Any],
    samples_ms: list[float],
    summary: dict[str, float],
    determinism: dict[str, Any],
) -> None:
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    env = collect_env_metadata()
    fields = [
        "case_name",
        "op_path",
        "sample_ms",
        "count",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "p90_ms",
        "p99_ms",
        "determinism_passed",
        "determinism_max_abs_diff",
        "determinism_max_rel_diff",
        "determinism_rtol",
        "determinism_atol",
        "artifact_hash",
        "batch_size",
        "world_size",
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "local_experts",
        "top_k",
        "quant_mode",
        "weight_source",
        "torch_version",
        "torch_npu_version",
        "vllm_ascend_path",
        "vllm_ascend_commit",
        "ascend_rt_visible_devices",
    ]
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for sample in samples_ms:
            row = {
                "case_name": args.case_name,
                "op_path": args.op_path,
                "sample_ms": sample,
                "determinism_passed": determinism["passed"],
                "determinism_max_abs_diff": determinism["max_abs_diff"],
                "determinism_max_rel_diff": determinism["max_rel_diff"],
                "determinism_rtol": DETERMINISM_RTOL,
                "determinism_atol": DETERMINISM_ATOL,
                "artifact_hash": metadata["artifact_hash"],
                "batch_size": cfg.batch_size,
                "world_size": cfg.world_size,
                "hidden_size": cfg.hidden_size,
                "moe_intermediate_size": cfg.moe_intermediate_size,
                "num_experts": cfg.num_experts,
                "local_experts": cfg.num_experts // cfg.world_size,
                "top_k": cfg.top_k,
                "quant_mode": cfg.quant_mode,
                "weight_source": metadata["config"]["weight_source"],
                "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            }
            row.update(summary)
            row.update(env)
            writer.writerow(row)


def collect_env_metadata() -> dict[str, str]:
    metadata = {
        "torch_version": "",
        "torch_npu_version": "",
        "vllm_ascend_path": "",
        "vllm_ascend_commit": "",
    }
    try:
        import torch

        metadata["torch_version"] = str(torch.__version__)
    except Exception:
        pass
    try:
        import torch_npu

        metadata["torch_npu_version"] = str(torch_npu.__version__)
    except Exception:
        pass
    try:
        import vllm_ascend

        path = Path(inspect.getfile(vllm_ascend)).resolve()
        metadata["vllm_ascend_path"] = str(path)
        repo = find_git_root(path)
        if repo is not None:
            metadata["vllm_ascend_commit"] = git_commit(repo)
    except Exception:
        pass
    return metadata


def find_git_root(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def cmd_check_outputs(args: argparse.Namespace) -> int:
    import torch

    metadata = load_metadata()
    rows = []
    all_passed = True
    for case_name in args.case:
        max_abs = 0.0
        mean_abs = 0.0
        max_rel = 0.0
        case_passed = True
        for rank in range(metadata["config"]["world_size"]):
            golden = load_case_output(torch, args.golden_case, rank, metadata)
            candidate = load_case_output(torch, case_name, rank, metadata)
            metrics = compare_tensors(torch, golden, candidate, args.rtol, args.atol)
            max_abs = max(max_abs, float(metrics["max_abs_diff"]))
            mean_abs = max(mean_abs, float(metrics["mean_abs_diff"]))
            max_rel = max(max_rel, float(metrics["max_rel_diff"]))
            case_passed = case_passed and bool(metrics["allclose"])
        rows.append(
            {
                "golden_case": args.golden_case,
                "case_name": case_name,
                "output_allclose": case_passed,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_rel_diff": max_rel,
                "rtol": args.rtol,
                "atol": args.atol,
            }
        )
        all_passed = all_passed and case_passed
    write_rows(RESULT_DIR / "output_check.csv", rows)
    print(f"wrote {RESULT_DIR / 'output_check.csv'}")
    if not all_passed:
        return 1
    return 0


def load_case_output(torch, case_name: str, rank: int, metadata: dict[str, Any]):
    path = OUTPUT_DIR / case_name / f"rank{rank}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_artifact_hash = metadata["artifact_hash"]
    actual_artifact_hash = payload.get("artifact_hash")
    if actual_artifact_hash != expected_artifact_hash:
        raise RuntimeError(
            f"output artifact hash mismatch: {path}, "
            f"expected {expected_artifact_hash}, got {actual_artifact_hash}"
        )
    expected_config = output_config_from_metadata(metadata)
    actual_config = payload.get("config")
    if actual_config != expected_config:
        raise RuntimeError(f"output config mismatch: {path}")
    output = payload["output"]
    expected_hash = payload.get("output_hash")
    if expected_hash and tensor_sha256(output) != expected_hash:
        raise RuntimeError(f"output hash mismatch: {path}")
    return output


def output_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cfg = metadata["config"]
    keys = [
        "batch_size",
        "world_size",
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "top_k",
        "quant_mode",
        "weight_source",
        "layer_index",
        "seed",
    ]
    return {key: cfg.get(key) for key in keys}


def cmd_plot(args: argparse.Namespace) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot. Run: pip install matplotlib") from exc

    timing_path = Path(args.input)
    if not timing_path.is_absolute():
        timing_path = ROOT / timing_path
    rows = read_csv_rows(timing_path)
    samples: dict[str, list[float]] = {}
    deterministic: dict[str, bool] = {}
    for row in rows:
        if not row.get("sample_ms"):
            continue
        case_name = row["case_name"]
        samples.setdefault(case_name, []).append(float(row["sample_ms"]))
        deterministic[case_name] = str(row.get("determinism_passed", "")).lower() == "true"
    output_status = read_output_status()
    summaries = []
    for case_name, values in samples.items():
        summary = summarize_values(values)
        summaries.append(
            {
                "case_name": case_name,
                **summary,
                "determinism_passed": deterministic.get(case_name, False),
                "output_allclose": output_status.get(case_name, case_name == "pangu_chain"),
            }
        )
    add_speedups(summaries)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(RESULT_DIR / "summary.csv", summaries)
    make_boxplot(plt, samples, PLOT_DIR / "latency_boxplot.png")
    make_summary_plot(plt, summaries, PLOT_DIR / "latency_summary.png")
    print(f"wrote {PLOT_DIR / 'latency_boxplot.png'}")
    print(f"wrote {PLOT_DIR / 'latency_summary.png'}")
    print(f"wrote {RESULT_DIR / 'summary.csv'}")
    return 0


def read_output_status() -> dict[str, bool]:
    path = RESULT_DIR / "output_check.csv"
    if not path.exists():
        return {}
    return {
        row["case_name"]: str(row["output_allclose"]).lower() == "true"
        for row in read_csv_rows(path)
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "median_ms": percentile(values, 0.50),
        "min_ms": values[0],
        "max_ms": values[-1],
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
    }


def add_speedups(summaries: list[dict[str, Any]]) -> None:
    by_case = {row["case_name"]: row for row in summaries}
    base_median = by_case.get("vllm_base", {}).get("median_ms")
    pangu_median = by_case.get("pangu_chain", {}).get("median_ms")
    for row in summaries:
        row["speedup_vs_vllm_base_median"] = (
            float(base_median) / float(row["median_ms"]) if base_median else ""
        )
        row["speedup_vs_pangu_chain_median"] = (
            float(pangu_median) / float(row["median_ms"]) if pangu_median else ""
        )


def percentile(values: list[float], q: float) -> float:
    return values[int((len(values) - 1) * q)]


def make_boxplot(plt, samples: dict[str, list[float]], output: Path) -> None:
    labels = list(samples)
    data = [samples[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("sample_ms")
    ax.set_title("Pangu 92B MoE latency distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def make_summary_plot(plt, summaries: list[dict[str, Any]], output: Path) -> None:
    labels = [row["case_name"] for row in summaries]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, key in [(-width, "median_ms"), (0, "p90_ms"), (width, "p99_ms")]:
        ax.bar([i + offset for i in x], [row[key] for row in summaries], width, label=key)
    for i, row in enumerate(summaries):
        valid = row["determinism_passed"] and row["output_allclose"]
        ax.text(i, 0, "valid" if valid else "invalid", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms")
    ax.set_title("Pangu 92B MoE latency summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def tensor_sha256(tensor) -> str:
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def artifact_hash(metadata: dict[str, Any]) -> str:
    payload = {
        "config": metadata["config"],
        "rank_hashes": metadata["rank_hashes"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def rank_input_path(rank: int) -> Path:
    return INPUT_DIR / f"rank{rank}.pt"


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} does not exist. Run prepare first."
        )
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
