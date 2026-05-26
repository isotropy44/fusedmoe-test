#!/usr/bin/env python3
"""Benchmark Pangu V2 92B decode MoE dispatch-combine on Ascend NPU."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


PANGU92_DEFAULTS = {
    "hidden_size": 2560,
    "moe_intermediate_size": 1024,
    "num_experts": 256,
    "num_shared_experts": 1,
    "top_k": 8,
    "quant_mode": 2,
}

ACL_FORMAT_FRACTAL_NZ = 29
PANGU_DEQUANT_SWIGLU_QUANT_MODE = 1


@dataclass(frozen=True)
class RuntimeInfo:
    rank: int
    local_rank: int
    world_size: int
    group_ep: str


@dataclass(frozen=True)
class BenchConfig:
    batch_size: int
    world_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k: int
    quant_mode: int
    warmup: int
    repeat: int
    output: str
    model_path: str
    synthetic_fallback: bool
    layer_index: int
    seed: int
    op_path: str
    check_numerics: bool
    rtol: float
    atol: float


@dataclass(frozen=True)
class Weights:
    w13_weight: Any
    w2_weight: Any
    w13_weight_scale: Any
    w2_weight_scale: Any
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Pangu V2 92B decode MoE dispatch-combine closed interval."
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="results/pangu92_decode_dispatch_combine.csv")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--synthetic-fallback", action="store_true")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--moe-intermediate-size", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--op-path", choices=["pangu", "vllm", "both"], default="both")
    parser.add_argument(
        "--check-numerics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compare Pangu chain and vLLM fused outputs before timed measurement.",
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
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

    cfg = resolve_config(args)
    validate_config(cfg)

    import torch
    import torch.distributed as dist
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True
    ensure_vllm_custom_op(torch, cfg)
    runtime = init_distributed(torch, dist, cfg.world_size)
    if runtime.world_size != cfg.world_size:
        raise ValueError(
            f"--world-size={cfg.world_size} does not match WORLD_SIZE={runtime.world_size}"
        )
    device = torch.device(f"npu:{runtime.local_rank}")

    weights = build_weights(torch, torch_npu, cfg, runtime, device)
    hidden_states, topk_ids, topk_weights = build_inputs(torch, cfg, runtime, device)
    quant_scale = torch.ones(
        (cfg.num_experts // cfg.world_size, cfg.moe_intermediate_size),
        dtype=torch.float32,
        device=device,
    )
    torch.npu.synchronize()

    if runtime.rank == 0:
        print(
            "pangu92 decode MoE benchmark: "
            f"batch_size={cfg.batch_size} h={cfg.hidden_size} "
            f"inter={cfg.moe_intermediate_size} experts={cfg.num_experts} "
            f"top_k={cfg.top_k} world_size={cfg.world_size} "
            f"quant_mode={cfg.quant_mode} op_path={cfg.op_path} "
            f"weight_source={weights.source}"
        )

    operations = build_operations(
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
    numerics = maybe_check_numerics(torch, cfg, operations)
    if numerics["checked"] and not numerics["allclose"]:
        if is_primary_rank(torch):
            write_numerics_failure_csv(cfg, weights.source, numerics)
        raise RuntimeError(
            "numerics check failed: "
            f"max_abs_diff={numerics['max_abs_diff']:.8g}, "
            f"mean_abs_diff={numerics['mean_abs_diff']:.8g}, "
            f"max_rel_diff={numerics['max_rel_diff']:.8g}, "
            f"rtol={cfg.rtol}, atol={cfg.atol}"
        )
    results: dict[str, dict[str, object]] = {}
    for op_name, operation in operations.items():
        samples_ms = measure(torch, cfg, operation)
        summary = summarize(samples_ms)
        results[op_name] = {
            "samples_ms": samples_ms,
            "summary": summary,
        }
    if is_primary_rank(torch):
        write_csv(cfg, weights.source, results, numerics)
        payload = {
            "config": asdict(cfg),
            "weight_source": weights.source,
            "numerics": numerics,
            "results": results,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            parts = []
            for op_name, result in results.items():
                summary = result["summary"]
                parts.append(
                    f"{op_name}:mean_ms={summary['mean_ms']:.6f},"
                    f"median_ms={summary['median_ms']:.6f}"
                )
            print(
                f"model=pangu_v2_moe_92B backend=npu "
                f"{' '.join(parts)} output={cfg.output}"
            )
    return 0


def resolve_config(args: argparse.Namespace) -> BenchConfig:
    model_cfg = read_model_config(args.model_path) if args.model_path else {}
    hidden_size = first_int(
        args.hidden_size, model_cfg.get("hidden_size"), PANGU92_DEFAULTS["hidden_size"]
    )
    intermediate = first_int(
        args.moe_intermediate_size,
        model_cfg.get("moe_intermediate_size"),
        PANGU92_DEFAULTS["moe_intermediate_size"],
    )
    num_experts = first_int(
        args.num_experts,
        model_cfg.get("n_routed_experts"),
        model_cfg.get("num_experts"),
        PANGU92_DEFAULTS["num_experts"],
    )
    top_k = first_int(
        args.top_k,
        model_cfg.get("num_experts_per_tok"),
        model_cfg.get("top_k"),
        PANGU92_DEFAULTS["top_k"],
    )
    return BenchConfig(
        batch_size=args.batch_size,
        world_size=args.world_size,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate,
        num_experts=num_experts,
        top_k=top_k,
        quant_mode=PANGU92_DEFAULTS["quant_mode"],
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
        model_path=args.model_path,
        synthetic_fallback=args.synthetic_fallback,
        layer_index=args.layer_index,
        seed=args.seed,
        op_path=args.op_path,
        check_numerics=(
            args.check_numerics if args.check_numerics is not None
            else args.op_path == "both"
        ),
        rtol=args.rtol,
        atol=args.atol,
    )


def first_int(*values: object) -> int:
    for value in values:
        if value is not None:
            return int(value)
    raise ValueError("no integer value provided")


def read_model_config(model_path: str) -> dict[str, object]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return data


def validate_config(cfg: BenchConfig) -> None:
    if cfg.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if cfg.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if cfg.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if cfg.num_experts % cfg.world_size != 0:
        raise ValueError("--num-experts must be divisible by --world-size")
    if cfg.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if cfg.top_k > cfg.num_experts:
        raise ValueError("--top-k must be <= --num-experts")
    if cfg.check_numerics and cfg.op_path != "both":
        raise ValueError("--check-numerics requires --op-path both")
    if cfg.rtol < 0 or cfg.atol < 0:
        raise ValueError("--rtol and --atol must be >= 0")
    if cfg.op_path in {"vllm", "both"}:
        validate_vllm_fused_config(cfg)


def validate_vllm_fused_config(cfg: BenchConfig) -> None:
    gmm1_hidden = 2 * cfg.moe_intermediate_size
    local_experts = cfg.num_experts // cfg.world_size
    if cfg.batch_size > 256:
        raise ValueError("dispatch_gmm_combine_decode supports --batch-size <= 256")
    if cfg.hidden_size < 512 or cfg.hidden_size > 7168:
        raise ValueError("dispatch_gmm_combine_decode supports 512 <= hidden_size <= 7168")
    if gmm1_hidden < 1024 or gmm1_hidden > 6144:
        raise ValueError(
            "dispatch_gmm_combine_decode supports "
            "1024 <= 2 * moe_intermediate_size <= 6144"
        )
    if cfg.num_experts > 512:
        raise ValueError("dispatch_gmm_combine_decode supports num_experts <= 512")
    if cfg.top_k > 12:
        raise ValueError("dispatch_gmm_combine_decode supports top_k <= 12")
    if local_experts > 24:
        raise ValueError(
            "dispatch_gmm_combine_decode tiling requires "
            "num_experts / world_size <= 24 on A3, got "
            f"{local_experts}. Use 16 ranks for full Pangu 92B 256 experts."
        )


def ensure_vllm_custom_op(torch, cfg: BenchConfig) -> None:
    if cfg.op_path not in {"vllm", "both"}:
        return
    try:
        from vllm_ascend.utils import enable_custom_op
    except ImportError as exc:
        raise RuntimeError(
            "vLLM-Ascend is required for --op-path vllm or both. "
            "Install vllm-ascend in the current environment."
        ) from exc
    if not enable_custom_op():
        raise RuntimeError(
            "vLLM-Ascend custom ops are not registered. "
            "Build and install vllm-ascend first."
        )
    load_vllm_ascend_extension()
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(
        torch.ops._C_ascend, "dispatch_gmm_combine_decode"
    ):
        raise RuntimeError(
            "torch.ops._C_ascend.dispatch_gmm_combine_decode is unavailable after "
            "enable_custom_op() and importing vllm_ascend.vllm_ascend_C. "
            "For A3, rebuild vllm-ascend with SOC_VERSION=ascend910_93 and source "
            "the generated _cann_ops_custom set_env.bash. A2 ascend910b1 builds do "
            "not package this fused decode ACLNN op."
        )


def load_vllm_ascend_extension() -> None:
    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import vllm_ascend.vllm_ascend_C. The Python package may be "
            "installed, but the C++ extension that registers torch.ops._C_ascend "
            "is not loaded."
        ) from exc


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
    forwarded = [arg for arg in sys.argv[1:] if arg != "--no-torchrun"]
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


def init_distributed(torch, dist, expected_world_size: int) -> RuntimeInfo:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        if expected_world_size == 1 and "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="hccl")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    group = dist.distributed_c10d._get_default_group()
    backend = group._get_backend(torch.device("npu"))
    return RuntimeInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        group_ep=backend.get_hccl_comm_name(rank),
    )


def build_inputs(torch, cfg: BenchConfig, runtime: RuntimeInfo, device):
    torch.manual_seed(cfg.seed + runtime.rank)
    hidden_states = torch.randn(
        (cfg.batch_size, cfg.hidden_size), dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.arange(
        runtime.rank * cfg.batch_size * cfg.top_k,
        (runtime.rank + 1) * cfg.batch_size * cfg.top_k,
        dtype=torch.int32,
        device=device,
    ).view(cfg.batch_size, cfg.top_k) % cfg.num_experts
    raw_weights = torch.rand(
        (cfg.batch_size, cfg.top_k), dtype=torch.float32, device=device
    )
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_ids, topk_weights


def build_weights(torch, torch_npu, cfg: BenchConfig, runtime: RuntimeInfo, device) -> Weights:
    if cfg.model_path:
        try:
            return load_real_weights(torch, torch_npu, cfg, runtime, device)
        except Exception as exc:
            if not cfg.synthetic_fallback:
                raise
            if runtime.rank == 0:
                print(f"warning: failed to load real weights, fallback to synthetic: {exc}")
    elif not cfg.synthetic_fallback:
        raise ValueError("--model-path is required unless --synthetic-fallback is set")
    return build_synthetic_weights(torch, torch_npu, cfg, device)


def build_synthetic_weights(torch, torch_npu, cfg: BenchConfig, device) -> Weights:
    local_experts = cfg.num_experts // cfg.world_size
    w13_weight = torch.randint(
        -8,
        8,
        (local_experts, cfg.hidden_size, 2 * cfg.moe_intermediate_size),
        dtype=torch.int8,
        device=device,
    ).contiguous()
    w2_weight = torch.randint(
        -8,
        8,
        (local_experts, cfg.moe_intermediate_size, cfg.hidden_size),
        dtype=torch.int8,
        device=device,
    ).contiguous()
    w13_scale = torch.ones(
        (local_experts, 2 * cfg.moe_intermediate_size),
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.ones(
        (local_experts, cfg.hidden_size), dtype=torch.bfloat16, device=device
    )
    return Weights(
        w13_weight=to_nz(torch_npu, w13_weight),
        w2_weight=to_nz(torch_npu, w2_weight),
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        source="synthetic",
    )


def load_real_weights(torch, torch_npu, cfg: BenchConfig, runtime: RuntimeInfo, device) -> Weights:
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    tensor_map = load_weight_tensors(model_path, cfg.layer_index)
    local_experts = cfg.num_experts // cfg.world_size
    w13_weight = prepare_weight(
        torch,
        tensor_map["w13_weight"],
        runtime.rank,
        local_experts,
        cfg.num_experts,
        (cfg.hidden_size, 2 * cfg.moe_intermediate_size),
        "w13_weight",
    ).to(device=device, dtype=torch.int8)
    w2_weight = prepare_weight(
        torch,
        tensor_map["w2_weight"],
        runtime.rank,
        local_experts,
        cfg.num_experts,
        (cfg.moe_intermediate_size, cfg.hidden_size),
        "w2_weight",
    ).to(device=device, dtype=torch.int8)
    w13_scale = prepare_scale(
        torch,
        tensor_map["w13_weight_scale"],
        runtime.rank,
        local_experts,
        cfg.num_experts,
        2 * cfg.moe_intermediate_size,
        "w13_weight_scale",
    ).to(device=device, dtype=torch.float32)
    w2_scale = prepare_scale(
        torch,
        tensor_map["w2_weight_scale"],
        runtime.rank,
        local_experts,
        cfg.num_experts,
        cfg.hidden_size,
        "w2_weight_scale",
    ).to(device=device, dtype=torch.bfloat16)
    return Weights(
        w13_weight=to_nz(torch_npu, w13_weight.contiguous()),
        w2_weight=to_nz(torch_npu, w2_weight.contiguous()),
        w13_weight_scale=w13_scale.contiguous(),
        w2_weight_scale=w2_scale.contiguous(),
        source="real",
    )


def load_weight_tensors(model_path: Path, layer_index: int) -> dict[str, Any]:
    wanted = {
        "w13_weight": [
            f"layers.{layer_index}.mlp.experts.w13_weight",
            f"model.layers.{layer_index}.mlp.experts.w13_weight",
            f"layers.{layer_index}.mlp.experts.gate_up_proj.weight",
            f"model.layers.{layer_index}.mlp.experts.gate_up_proj.weight",
        ],
        "w2_weight": [
            f"layers.{layer_index}.mlp.experts.w2_weight",
            f"model.layers.{layer_index}.mlp.experts.w2_weight",
            f"layers.{layer_index}.mlp.experts.down_proj.weight",
            f"model.layers.{layer_index}.mlp.experts.down_proj.weight",
        ],
        "w13_weight_scale": [
            f"layers.{layer_index}.mlp.experts.w13_weight_scale",
            f"model.layers.{layer_index}.mlp.experts.w13_weight_scale",
            f"layers.{layer_index}.mlp.experts.gate_up_proj.weight_scale",
            f"model.layers.{layer_index}.mlp.experts.gate_up_proj.weight_scale",
            f"layers.{layer_index}.mlp.experts.gate_up_proj.scale",
            f"model.layers.{layer_index}.mlp.experts.gate_up_proj.scale",
        ],
        "w2_weight_scale": [
            f"layers.{layer_index}.mlp.experts.w2_weight_scale",
            f"model.layers.{layer_index}.mlp.experts.w2_weight_scale",
            f"layers.{layer_index}.mlp.experts.down_proj.weight_scale",
            f"model.layers.{layer_index}.mlp.experts.down_proj.weight_scale",
            f"layers.{layer_index}.mlp.experts.down_proj.scale",
            f"model.layers.{layer_index}.mlp.experts.down_proj.scale",
        ],
    }
    resolved: dict[str, Any] = {}
    try:
        resolved.update(load_safetensors(model_path, wanted))
    except ImportError:
        pass
    missing = [name for name in wanted if name not in resolved]
    if missing:
        resolved.update(load_torch_checkpoints(model_path, wanted, missing))
    missing = [name for name in wanted if name not in resolved]
    if missing:
        raise KeyError(f"missing tensors for layer {layer_index}: {missing}")
    return resolved


def load_safetensors(model_path: Path, wanted: dict[str, list[str]]) -> dict[str, Any]:
    from safetensors import safe_open

    files = list_safetensor_files(model_path)
    resolved: dict[str, Any] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for logical_name, candidates in wanted.items():
                if logical_name in resolved:
                    continue
                key = find_key(keys, candidates)
                if key is not None:
                    resolved[logical_name] = f.get_tensor(key)
    return resolved


def list_safetensor_files(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict):
            return sorted({model_path / str(file_name) for file_name in weight_map.values()})
    return sorted(model_path.glob("*.safetensors"))


def load_torch_checkpoints(
    model_path: Path,
    wanted: dict[str, list[str]],
    missing: list[str],
) -> dict[str, Any]:
    import torch

    files = sorted(model_path.glob("*.bin")) + sorted(model_path.glob("*.pt"))
    resolved: dict[str, Any] = {}
    for file_path in files:
        if not missing:
            break
        state = torch.load(file_path, map_location="cpu")
        if not isinstance(state, dict):
            continue
        keys = set(state.keys())
        for logical_name in list(missing):
            key = find_key(keys, wanted[logical_name])
            if key is not None:
                resolved[logical_name] = state[key]
                missing.remove(logical_name)
    return resolved


def find_key(keys: set[str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    for candidate in candidates:
        suffix = "." + candidate
        for key in keys:
            if key.endswith(suffix) or key.endswith(candidate):
                return key
    return None


def prepare_weight(
    torch,
    tensor,
    rank: int,
    local_experts: int,
    num_experts: int,
    desired_tail: tuple[int, int],
    name: str,
):
    tensor = tensor.detach().cpu()
    tensor = select_local_experts(tensor, rank, local_experts, num_experts, name)
    if tuple(tensor.shape[1:]) == desired_tail:
        return tensor.contiguous()
    if tuple(tensor.shape[1:]) == (desired_tail[1], desired_tail[0]):
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(
        f"{name} shape {tuple(tensor.shape)} cannot match local shape "
        f"({local_experts}, {desired_tail[0]}, {desired_tail[1]})"
    )


def prepare_scale(
    torch,
    tensor,
    rank: int,
    local_experts: int,
    num_experts: int,
    desired_width: int,
    name: str,
):
    tensor = tensor.detach().cpu()
    while tensor.dim() > 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    tensor = select_local_experts(tensor, rank, local_experts, num_experts, name)
    if tuple(tensor.shape[1:]) != (desired_width,):
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} cannot match local shape "
            f"({local_experts}, {desired_width})"
        )
    return tensor.contiguous()


def select_local_experts(tensor, rank: int, local_experts: int, num_experts: int, name: str):
    if tensor.dim() < 2:
        raise ValueError(f"{name} must have expert dimension, got {tuple(tensor.shape)}")
    if tensor.shape[0] == local_experts:
        return tensor
    if tensor.shape[0] == num_experts:
        start = rank * local_experts
        return tensor[start:start + local_experts]
    raise ValueError(
        f"{name} first dimension must be local_experts={local_experts} "
        f"or num_experts={num_experts}, got {tuple(tensor.shape)}"
    )


def to_nz(torch_npu, tensor):
    return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)


def build_operations(
    torch,
    torch_npu,
    cfg: BenchConfig,
    runtime: RuntimeInfo,
    weights: Weights,
    hidden_states,
    topk_ids,
    topk_weights,
    quant_scale,
):
    operations = {}
    if cfg.op_path in {"pangu", "both"}:
        operations["pangu_chain"] = make_pangu_operation(
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
    if cfg.op_path in {"vllm", "both"}:
        operations["vllm_fused"] = make_vllm_fused_operation(
            torch,
            cfg,
            runtime,
            weights,
            hidden_states,
            topk_ids,
            topk_weights,
        )
    return operations


def make_pangu_operation(
    torch,
    torch_npu,
    cfg: BenchConfig,
    runtime: RuntimeInfo,
    weights: Weights,
    hidden_states,
    topk_ids,
    topk_weights,
    quant_scale,
):
    def operation():
        dispatch_output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=cfg.num_experts,
            global_bs=0,
            scales=None,
            quant_mode=cfg.quant_mode,
            group_ep=runtime.group_ep,
            ep_world_size=runtime.world_size,
            ep_rank_id=runtime.rank,
            x_active_mask=None,
            comm_alg="fullmesh_v2",
        )
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = dispatch_output[0:6]
        if dynamic_scale is None:
            raise RuntimeError("npu_moe_distribute_dispatch_v2 returned dynamic_scale=None")
        if dynamic_scale.dim() > 1:
            dynamic_scale = dynamic_scale.reshape(-1)
            expand_x = expand_x.view(-1, expand_x.shape[-1])
        expert_tokens = expert_token_nums.to(torch.int64)
        gate_up_proj = torch_npu.npu_grouped_matmul(
            [expand_x],
            [weights.w13_weight],
            bias=None,
            scale=None,
            per_token_scale=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.int32,
            group_type=0,
            group_list_type=1,
            act_type=0,
        )[0]
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            x=gate_up_proj,
            weight_scale=weights.w13_weight_scale,
            activation_scale=dynamic_scale,
            bias=None,
            quant_offset=None,
            quant_scale=quant_scale,
            group_index=expert_tokens,
            activate_left=True,
            # This op uses its own enum. Keep it aligned with the Pangu chain that
            # pairs dispatch quant_mode=2 with dequant_swiglu_quant quant_mode=1.
            quant_mode=PANGU_DEQUANT_SWIGLU_QUANT_MODE,
        )
        down_proj_output = torch_npu.npu_grouped_matmul(
            [intermediate_h],
            [weights.w2_weight],
            bias=None,
            scale=[weights.w2_weight_scale.to(torch.bfloat16)],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
        )[0]
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=down_proj_output,
            expert_ids=topk_ids,
            assist_info_for_combine=expand_idx,
            expert_scales=topk_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=cfg.num_experts,
            global_bs=0,
            ep_send_counts=ep_recv_counts,
            group_ep=runtime.group_ep,
            ep_world_size=runtime.world_size,
            ep_rank_id=runtime.rank,
            tp_send_counts=tp_recv_counts,
            x_active_mask=None,
        )

    return operation


def make_vllm_fused_operation(
    torch,
    cfg: BenchConfig,
    runtime: RuntimeInfo,
    weights: Weights,
    hidden_states,
    topk_ids,
    topk_weights,
):
    gmm1_scale = weights.w13_weight_scale.to(torch.float32)
    gmm2_scale = weights.w2_weight_scale.to(torch.float32)
    x_active_mask = torch.ones(
        (cfg.batch_size,), dtype=torch.bool, device=hidden_states.device
    )

    def operation():
        output, _expert_token_nums = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=hidden_states,
            expert_ids=topk_ids,
            gmm1_permuted_weight=[weights.w13_weight],
            gmm1_permuted_weight_scale=[gmm1_scale],
            gmm2_weight=[weights.w2_weight],
            gmm2_weight_scale=[gmm2_scale],
            expert_scales=topk_weights.to(torch.float32),
            expert_smooth_scales=None,
            x_active_mask=x_active_mask,
            group_ep=runtime.group_ep,
            ep_rank_size=runtime.world_size,
            ep_rank_id=runtime.rank,
            moe_expert_num=cfg.num_experts,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            quant_mode=cfg.quant_mode,
            global_bs=0,
        )
        return output

    return operation


def maybe_check_numerics(torch, cfg: BenchConfig, operations) -> dict[str, object]:
    if not cfg.check_numerics:
        return {
            "checked": False,
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "max_rel_diff": None,
        }
    pangu_output = unwrap_output(run_once_synced(torch, operations["pangu_chain"]))
    vllm_output = unwrap_output(run_once_synced(torch, operations["vllm_fused"]))
    torch.npu.synchronize()
    if tuple(pangu_output.shape) != tuple(vllm_output.shape):
        raise RuntimeError(
            "numerics check failed: output shape mismatch "
            f"pangu={tuple(pangu_output.shape)} vllm={tuple(vllm_output.shape)}"
        )
    pangu_f32 = pangu_output.to(torch.float32)
    vllm_f32 = vllm_output.to(torch.float32)
    abs_diff = (pangu_f32 - vllm_f32).abs()
    rel_diff = abs_diff / torch.maximum(
        pangu_f32.abs(), torch.tensor(cfg.atol, dtype=torch.float32, device=pangu_f32.device)
    )
    max_abs_diff = reduce_max_scalar(torch, float(abs_diff.max().detach().cpu().item()))
    mean_abs_diff = reduce_max_scalar(torch, float(abs_diff.mean().detach().cpu().item()))
    max_rel_diff = reduce_max_scalar(torch, float(rel_diff.max().detach().cpu().item()))
    local_allclose = bool(
        torch.allclose(pangu_f32, vllm_f32, rtol=cfg.rtol, atol=cfg.atol)
    )
    global_failure = reduce_max_scalar(torch, 0.0 if local_allclose else 1.0)
    allclose = global_failure == 0.0
    result = {
        "checked": True,
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
    }
    return result


def run_once_synced(torch, operation):
    torch.npu.synchronize()
    output = operation()
    torch.npu.synchronize()
    return output


def unwrap_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def measure(torch, cfg: BenchConfig, operation) -> list[float]:
    for _ in range(cfg.warmup):
        operation()
    torch.npu.synchronize()

    samples_ms: list[float] = []
    for _ in range(cfg.repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.npu.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        samples_ms.append(maybe_reduce_max_elapsed(torch, elapsed_ms))
    return samples_ms


def maybe_reduce_max_elapsed(torch, elapsed_ms: float) -> float:
    return reduce_max_scalar(torch, elapsed_ms)


def reduce_max_scalar(torch, value: float) -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device="npu")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.cpu().item())


def is_primary_rank(torch) -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return True


def summarize(samples_ms: list[float]) -> dict[str, float]:
    samples = sorted(samples_ms)
    idx_p50 = int((len(samples) - 1) * 0.50)
    idx_p90 = int((len(samples) - 1) * 0.90)
    idx_p99 = int((len(samples) - 1) * 0.99)
    return {
        "count": float(len(samples)),
        "mean_ms": statistics.fmean(samples),
        "median_ms": samples[idx_p50],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "p90_ms": samples[idx_p90],
        "p99_ms": samples[idx_p99],
    }


def write_csv(
    cfg: BenchConfig,
    weight_source: str,
    results: dict[str, dict[str, object]],
    numerics: dict[str, object],
) -> None:
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    fields = [
        "model",
        "scenario",
        "batch_size",
        "world_size",
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "local_experts",
        "top_k",
        "quant_mode",
        "op_path",
        "weight_source",
        "layer_index",
        "numerics_checked",
        "numerics_allclose",
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "sample_ms",
        "count",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "p90_ms",
        "p99_ms",
    ]
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for op_path, result in results.items():
            samples_ms = result["samples_ms"]
            summary = result["summary"]
            for sample in samples_ms:
                row: dict[str, object] = {
                    "model": "pangu_v2_moe_92B",
                    "scenario": "decode",
                    "batch_size": cfg.batch_size,
                    "world_size": cfg.world_size,
                    "hidden_size": cfg.hidden_size,
                    "moe_intermediate_size": cfg.moe_intermediate_size,
                    "num_experts": cfg.num_experts,
                    "local_experts": cfg.num_experts // cfg.world_size,
                    "top_k": cfg.top_k,
                    "quant_mode": cfg.quant_mode,
                    "op_path": op_path,
                    "weight_source": weight_source,
                    "layer_index": cfg.layer_index,
                    "numerics_checked": numerics["checked"],
                    "numerics_allclose": numerics["allclose"],
                    "max_abs_diff": numerics["max_abs_diff"],
                    "mean_abs_diff": numerics["mean_abs_diff"],
                    "max_rel_diff": numerics["max_rel_diff"],
                    "sample_ms": sample,
                }
                row.update(summary)
                writer.writerow(row)


def write_numerics_failure_csv(
    cfg: BenchConfig,
    weight_source: str,
    numerics: dict[str, object],
) -> None:
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    fields = [
        "model",
        "scenario",
        "batch_size",
        "world_size",
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "local_experts",
        "top_k",
        "quant_mode",
        "op_path",
        "weight_source",
        "layer_index",
        "numerics_checked",
        "numerics_allclose",
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "sample_ms",
        "count",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "p90_ms",
        "p99_ms",
    ]
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": "pangu_v2_moe_92B",
                "scenario": "decode",
                "batch_size": cfg.batch_size,
                "world_size": cfg.world_size,
                "hidden_size": cfg.hidden_size,
                "moe_intermediate_size": cfg.moe_intermediate_size,
                "num_experts": cfg.num_experts,
                "local_experts": cfg.num_experts // cfg.world_size,
                "top_k": cfg.top_k,
                "quant_mode": cfg.quant_mode,
                "op_path": "numerics_check",
                "weight_source": weight_source,
                "layer_index": cfg.layer_index,
                "numerics_checked": numerics["checked"],
                "numerics_allclose": numerics["allclose"],
                "max_abs_diff": numerics["max_abs_diff"],
                "mean_abs_diff": numerics["mean_abs_diff"],
                "max_rel_diff": numerics["max_rel_diff"],
                "sample_ms": "",
                "count": "",
                "mean_ms": "",
                "median_ms": "",
                "min_ms": "",
                "max_ms": "",
                "p90_ms": "",
                "p99_ms": "",
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
