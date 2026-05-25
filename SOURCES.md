# 来源与本地证据

本文只记录当前 Pangu 92B 三 case 实验相关来源. 早期无关材料已移除.

## 1. Pangu 92B 配置与普通 MoE 链路

1. `/Users/jhyang/workcode/omni-npu/src/omni_npu/model_config/configs/match_hf_configs.json:247` 到 `:255`

  显示 Pangu V2 92B 配置: `hidden_size=2560`, `n_routed_experts=256`, `n_shared_experts=1`, `moe_intermediate_size=1024`.

2. `/Users/jhyang/workcode/omni-models/omni_models/models/pangu_v2/pangu_v2_moe.py:818` 到 `:956`

  显示 Pangu V2 dispatch-combine 单 batch 路径: `torch_npu.npu_moe_gating_top_k`, `torch_npu.npu_moe_distribute_dispatch_v2`, `torch_npu.npu_grouped_matmul`, `torch_npu.npu_dequant_swiglu_quant`, 第二次 `torch_npu.npu_grouped_matmul`, `torch_npu.npu_moe_distribute_combine_v2`.

3. `/Users/jhyang/workcode/omni-npu/src/omni_npu/layers/quantization/compressed_tensors/compressed_tensors_moe.py:89` 到 `:186`

  显示 W8A8 (8-bit Weight, 8-bit Activation) MoE 权重原始 shape, 加载后转置, NZ 格式转换, 以及 scale squeeze 后的 dtype 处理.

## 2. vLLM-Ascend fused MoE 链路

1. `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py`

  显示 fused MoE 通信路径中会调用 `torch.ops._C_ascend.dispatch_gmm_combine_decode`.

2. `vllm-ascend/csrc/torch_binding.cpp`

  显示 `dispatch_gmm_combine_decode` 的 Python binding 参数, 返回 `(output, expert_token_nums)`, 并注册到 `torch.ops._C_ascend.dispatch_gmm_combine_decode`.

3. `vllm-ascend/csrc/dispatch_gmm_combine_decode/op_host/dispatch_gmm_combine_decode_tiling.cpp`

  显示 fused op 对 `x`, `expert_ids`, `gmm1_weight`, `gmm1_scale`, `gmm2_weight`, `gmm2_scale` 的 shape 约束.

4. `vllm-ascend/vllm_ascend/envs.py`

  说明 `dispatch_gmm_combine_decode` 用于 W8A8 decode node MoE layer.

5. `vllm-ascend/vllm_ascend/ascend_forward_context.py`

  显示 A3 fused MC2 selection guard, 以及 fallback 到其他 MoE 通信路径的逻辑.

6. `vllm-ascend/vllm_ascend/quantization/w8a8_dynamic.py`

  显示 vLLM-Ascend W8A8 权重在加载后会转置, 转成 `ACL_FORMAT_FRACTAL_NZ`, 并在 dynamic EPLB 下拆成 per expert list.

## 3. A2 和 A3 环境事实

1. vLLM-Ascend `releases/v0.18.0` 环境按 CANN (Compute Architecture for Neural Networks) `9.0.0`, NNAL `9.0.0`, PyTorch `2.9.0`, torch-npu `2.9.0.post2` 组织.

2. A2 `SOC_VERSION=ascend910b1` 构建不会把 `dispatch_gmm_combine_decode` 的 ACLNN (Ascend Computing Language Neural Network) op_api 打入 custom ops 包. 因此 A2 只作为 `pangu_chain` 和 artifacts 预验证环境.

3. A3 正式对比需要用 `SOC_VERSION=ascend910_93` 构建 vLLM-Ascend, 并通过 `enable_custom_op()` 加载 `vllm_ascend.vllm_ascend_C`, 才能确认 `torch.ops._C_ascend.dispatch_gmm_combine_decode` 注册.

## 4. 当前仓库入口

1. `run_pangu92_three_case_experiment.py`

  当前唯一推荐入口. 提供 `prepare`, `verify-artifacts`, `run-case`, `check-outputs`, `plot` 5 个子命令.

2. `run_pangu92_decode_dispatch_combine_benchmark.py`

  三 case 主脚本复用的底层实现. 包含 Pangu chain 和 vLLM fused op 的构造, 权重加载, NPU event 计时, distributed max reduce, summary 计算.

3. `README.md`, `RUNNING.md`, `experiment_design.md`

  分别覆盖快速开始, 环境搭建与运行, 实验设计和读数方式.
