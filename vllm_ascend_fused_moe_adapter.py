#!/usr/bin/env python3
"""vLLM-Ascend fused MoE adapter for dispatch_gmm_combine_decode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


ACL_FORMAT_FRACTAL_NZ = 29


@dataclass(frozen=True)
class RuntimeInfo:
    rank: int
    local_rank: int
    world_size: int
    group_ep: str


def build_case(config) -> Callable[[], object]:
    import torch
    import torch.distributed as dist
    import torch_npu
    torch_npu.npu.config.allow_internal_format = True
    from vllm_ascend.utils import enable_custom_op

    if config.backend != "npu":
        raise ValueError("vLLM-Ascend fused MoE adapter only supports --backend npu")

    runtime = _init_distributed(torch, dist, config.world_size)
    if runtime.world_size != config.world_size:
        raise ValueError(
            f"--world-size={config.world_size} does not match torchrun WORLD_SIZE={runtime.world_size}"
        )
    _validate_config(config)

    if not enable_custom_op():
        raise RuntimeError(
            "vLLM-Ascend custom ops are not registered. Build and install vllm-ascend first."
        )
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(
        torch.ops._C_ascend, "dispatch_gmm_combine_decode"
    ):
        raise RuntimeError(
            "torch.ops._C_ascend.dispatch_gmm_combine_decode is unavailable."
        )

    device = torch.device(f"npu:{runtime.local_rank}")
    dtype = _input_dtype(torch)
    quant_mode = int(os.environ.get("FMOE_ASCEND_QUANT_MODE", "0"))
    layout = os.environ.get("FMOE_ASCEND_WEIGHT_LAYOUT", "single").strip().lower()
    seed = int(os.environ.get("FMOE_ASCEND_SEED", "20260520"))
    local_experts = config.num_experts // runtime.world_size
    gmm1_hidden = config.moe_intermediate_size * 2

    torch.manual_seed(seed + runtime.rank)
    x = torch.randn((config.m, config.hidden_size), device=device, dtype=dtype)
    expert_ids = torch.randint(
        0,
        config.num_experts,
        (config.m, config.num_experts_per_tok),
        device=device,
        dtype=torch.int32,
    )
    expert_scales = torch.softmax(
        torch.randn(
            (config.m, config.num_experts_per_tok),
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    if layout == "list":
        gmm1_weight, gmm1_scale, gmm2_weight, gmm2_scale = _build_weight_lists(
            torch,
            torch_npu,
            device,
            local_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            gmm1_hidden,
        )
    elif layout == "single":
        gmm1_weight, gmm1_scale, gmm2_weight, gmm2_scale = _build_single_weight_tensors(
            torch,
            torch_npu,
            device,
            local_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            gmm1_hidden,
        )
    else:
        raise ValueError("FMOE_ASCEND_WEIGHT_LAYOUT must be single or list")

    torch.npu.synchronize()
    if runtime.rank == 0:
        print(
            "vllm-ascend fused MoE adapter: "
            f"op=dispatch_gmm_combine_decode m={config.m} "
            f"h={config.hidden_size} inter={config.moe_intermediate_size} "
            f"experts={config.num_experts} local_experts={local_experts} "
            f"topk={config.num_experts_per_tok} world_size={runtime.world_size} "
            f"dtype={dtype} quant_mode={quant_mode} layout={layout}"
        )

    def operation():
        return torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=x,
            expert_ids=expert_ids,
            gmm1_permuted_weight=gmm1_weight,
            gmm1_permuted_weight_scale=gmm1_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_scale,
            expert_scales=expert_scales,
            expert_smooth_scales=None,
            x_active_mask=None,
            group_ep=runtime.group_ep,
            ep_rank_size=runtime.world_size,
            ep_rank_id=runtime.rank,
            moe_expert_num=config.num_experts,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            quant_mode=quant_mode,
            global_bs=config.m * runtime.world_size,
        )

    return operation


def _init_distributed(torch, dist, expected_world_size: int) -> RuntimeInfo:
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
    group_ep = _get_hccl_comm_name(torch, dist, rank)
    return RuntimeInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        group_ep=group_ep,
    )


def _get_hccl_comm_name(torch, dist, rank: int) -> str:
    group = dist.distributed_c10d._get_default_group()
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _input_dtype(torch):
    dtype_name = os.environ.get("FMOE_ASCEND_DTYPE", "bf16").strip().lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError("FMOE_ASCEND_DTYPE must be bf16 or fp16")


def _validate_config(config) -> None:
    gmm1_hidden = config.moe_intermediate_size * 2
    if config.m < 1 or config.m > 256:
        raise ValueError("dispatch_gmm_combine_decode supports 1 <= m <= 256")
    if config.hidden_size < 512 or config.hidden_size > 7168:
        raise ValueError("dispatch_gmm_combine_decode supports 512 <= hidden_size <= 7168")
    if gmm1_hidden < 1024 or gmm1_hidden > 6144:
        raise ValueError("dispatch_gmm_combine_decode supports 1024 <= 2 * moe_intermediate_size <= 6144")
    if config.num_experts > 512:
        raise ValueError("dispatch_gmm_combine_decode supports num_experts <= 512")
    if config.num_experts_per_tok > 12:
        raise ValueError("dispatch_gmm_combine_decode supports num_experts_per_tok <= 12")
    if config.num_experts % config.world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    local_experts = config.num_experts // config.world_size
    if local_experts > 24:
        raise ValueError(
            "dispatch_gmm_combine_decode tiling requires num_experts / world_size <= 24 "
            f"on A3 (aivNum / 2), got {local_experts}. "
            "Reduce --num-experts or increase --world-size."
        )


def _rand_int8(torch, shape, device):
    return torch.randint(-8, 8, shape, device=device, dtype=torch.int8).contiguous()


def _to_nz(torch_npu, tensor):
    return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)


def _build_single_weight_tensors(
    torch,
    torch_npu,
    device,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gmm1_hidden: int,
):
    w1 = _to_nz(
        torch_npu,
        _rand_int8(torch, (local_experts, hidden_size, gmm1_hidden), device),
    )
    w2 = _to_nz(
        torch_npu,
        _rand_int8(torch, (local_experts, intermediate_size, hidden_size), device),
    )
    s1 = torch.ones((local_experts, gmm1_hidden), device=device, dtype=torch.float32)
    s2 = torch.ones((local_experts, hidden_size), device=device, dtype=torch.float32)
    return [w1], [s1], [w2], [s2]


def _build_weight_lists(
    torch,
    torch_npu,
    device,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gmm1_hidden: int,
):
    gmm1_weight = []
    gmm1_scale = []
    gmm2_weight = []
    gmm2_scale = []
    for _ in range(local_experts):
        gmm1_weight.append(
            _to_nz(torch_npu, _rand_int8(torch, (hidden_size, gmm1_hidden), device))
        )
        gmm1_scale.append(torch.ones((gmm1_hidden,), device=device, dtype=torch.float32))
        gmm2_weight.append(
            _to_nz(torch_npu, _rand_int8(torch, (intermediate_size, hidden_size), device))
        )
        gmm2_scale.append(torch.ones((hidden_size,), device=device, dtype=torch.float32))
    return gmm1_weight, gmm1_scale, gmm2_weight, gmm2_scale
