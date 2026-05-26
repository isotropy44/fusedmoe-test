from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_pangu92_decode_dispatch_combine_benchmark",
    ROOT / "run_pangu92_decode_dispatch_combine_benchmark.py",
)
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


class FakeTensor:
    def __init__(self, *, shape=None, dtype=None, device="npu:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def to(self, *args, **kwargs):
        return self


class FakeAscendOps:
    def __init__(self):
        self.kwargs = None

    def dispatch_gmm_combine_decode(self, **kwargs):
        self.kwargs = kwargs
        return FakeTensor(), FakeTensor()


class FakeTorch:
    float32 = "float32"
    bool = "bool"

    ascend_ops = FakeAscendOps()
    ops = SimpleNamespace(_C_ascend=ascend_ops)

    @staticmethod
    def ones(shape, *, dtype=None, device=None):
        return FakeTensor(shape=tuple(shape), dtype=dtype, device=device)


class VllmFusedMc2ArgsTest(unittest.TestCase):
    def test_vllm_fused_decode_uses_uniform_mc2_mask_mode(self):
        cfg = SimpleNamespace(
            batch_size=24,
            world_size=16,
            num_experts=256,
            quant_mode=2,
        )
        runtime = SimpleNamespace(rank=0, world_size=16, group_ep="fake_hccl")
        weights = SimpleNamespace(
            w13_weight=FakeTensor(),
            w13_weight_scale=FakeTensor(),
            w2_weight=FakeTensor(),
            w2_weight_scale=FakeTensor(),
        )
        hidden_states = FakeTensor(device="npu:0")
        topk_ids = FakeTensor()
        topk_weights = FakeTensor()

        operation = bench.make_vllm_fused_operation(
            FakeTorch,
            cfg,
            runtime,
            weights,
            hidden_states,
            topk_ids,
            topk_weights,
        )
        operation()

        kwargs = FakeTorch.ascend_ops.kwargs
        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs["global_bs"], 0)
        self.assertIsNotNone(kwargs["x_active_mask"])
        self.assertEqual(kwargs["x_active_mask"].shape, (cfg.batch_size,))
        self.assertEqual(kwargs["x_active_mask"].dtype, FakeTorch.bool)
        self.assertEqual(kwargs["x_active_mask"].device, hidden_states.device)


if __name__ == "__main__":
    unittest.main()
