# 来源与假设

## 1. 公开来源

1. DeepSeek V4 Pro Base Hugging Face config:
. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json

. 使用字段: `hidden_size=7168`, `moe_intermediate_size=3072`, `n_routed_experts=384`, `n_shared_experts=1`, `num_experts_per_tok=6`, `num_hidden_layers=61`, `num_hash_layers=3`, `max_position_embeddings=1048576`, 以及 `quantization_config` 中的 `quant_method=fp8`. 

2. vLLM DeepSeek V4 Pro recipe:
. https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro

. 使用事实: DeepSeek V4 Pro 是 1.6T total, 49B active 的 MoE (Mixture of Experts) model, checkpoint 为 FP4 (4-bit Floating Point) + FP8 (8-bit Floating Point) mixed, 并给出 8 GPU (Graphics Processing Unit) 部署建议. 

3. Ascend Extension for PyTorch README:
. https://gitee.com/ascend/pytorch

. 使用事实: CANN (Compute Architecture for Neural Networks) 8.5.0 对应 PyTorch 2.7.1 与 torch-npu 2.7.1.post2, 也对应 PyTorch 2.6.0 与 torch-npu 2.6.0.post5. 本地 mirror 证据为 `../external_repos/pytorch/README.zh.md:29` 到 `:39`. 

4. NVIDIA CUDA 12.9 Linux installation guide:
. https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html

. 使用事实: CUDA Toolkit 安装需要 CUDA-capable GPU, supported Linux, GCC (GNU Compiler Collection) toolchain, 以及 toolkit installation. 

5. PyTorch get started page:
. https://pytorch.org/get-started/

. 使用事实: 官方 PyTorch binary install 支持 CUDA 12.8 wheel index. 

6. DeepGEMM README:
. https://github.com/deepseek-ai/DeepGEMM

. 本地证据为 `../external_repos/DeepGEMM/README.md:27` 到 `:37` 的 requirements, 以及 `:114` 到 `:140` 的 MegaMoE 说明. 

## 2. 本地代码证据

1. `../NOTES/moe-stream.svg`

  定义所有实验共享的 MoE stream: dispatch, GMM (Grouped Matrix Multiplication) 1, dequant, SwiGLU, dynamic quant, GMM 2, dequant, combine. 

2. `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:319` 到 `:350`

  显示 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` 选择 `dispatch_ffn_combine`, `=2` 选择 `dispatch_gmm_combine_decode`. 

3. `vllm-ascend/csrc/torch_binding.cpp:566` 到 `:624`, 以及 `:2016` 到 `:2027`

  显示 `dispatch_gmm_combine_decode` 的 Python binding 参数, 返回 `(output, expert_token_nums)`, 并注册到 `torch.ops._C_ascend.dispatch_gmm_combine_decode`.

4. `vllm-ascend/csrc/mc2/dispatch_gmm_combine_decode/op_host/dispatch_gmm_combine_decode_tiling.cpp:88` 到 `:222`

  显示 fused op 对 `x`, `expert_ids`, `gmm1_weight`, `gmm1_scale`, `gmm2_weight`, `gmm2_scale` 的 shape 约束, 并支持单个 3D local expert tensor 或 TensorList.

5. `vllm-ascend/vllm_ascend/envs.py:130` 到 `:138`

  说明 `dispatch_gmm_combine_decode` 用于 W8A8 (8-bit Weight, 8-bit Activation) 的 decode node MoE layer. 

6. `vllm-ascend/vllm_ascend/ascend_forward_context.py:257` 到 `:279`

  显示 A3 fused MC2 selection guard, 以及 fallback 到 MC2 或 ALLTOALL 的逻辑. 

7. `vllm-ascend/vllm_ascend/quantization/w8a8_dynamic.py:292` 到 `:344`

  显示 vLLM-Ascend W8A8 权重在加载后会转置, 转成 `ACL_FORMAT_FRACTAL_NZ`, 并在 dynamic EPLB 下拆成 per expert list.

8. `omni-npu/src/omni_npu/layers/fused_moe/layer.py:130` 到 `:155`

  显示 Ascend 普通 MoE 选择 experts 并进入 prepare-permute. 

9. `omni-npu/src/omni_npu/layers/fused_moe/prepare_permute_unpermute_finalize.py:216` 到 `:299`

  显示 AGRS prepare path 使用 EP all-gather 和 `torch_npu.npu_moe_init_routing_v2`. 

10. `omni-npu/src/omni_npu/layers/fused_moe/prepare_permute_unpermute_finalize.py:436` 到 `:457`

  显示普通路径的 finalize routing 和 EP reduce-scatter. 

11. `DeepGEMM/tests/test_mega_moe.py:52` 到 `:57`

  显示 MegaMoE 的 symmetric memory allocation. 

12. `DeepGEMM/tests/test_mega_moe.py:103` 到 `:121`

  显示 `deep_gemm.fp8_fp4_mega_moe` 调用. 

13. `DeepGEMM/tests/test_mega_moe.py:157` 到 `:187`

  显示 legacy CUDA baseline: EP dispatch, grouped GMM, SwiGLU/quant, grouped GMM, EP combine. 

## 3. 假设

1. Benchmark 使用 DeepSeek V4 Pro Base shape 作为公共 proxy, 不使用 Pangu V2 shape.

2. 如果本地没有真实 DeepSeek V4 权重, 允许使用 deterministic synthetic weights 做 latency 测试. 结果 metadata 必须标注这一点.

3. Ascend 上的 A/B 和 CUDA 上的 C 不能解释为纯硬件无关比较. A/B 是同平台实现对比. C 是 CUDA fused scheduling 参考.

4. `M=24` 表示被测 MoE dispatch-to-combine 段中每 rank 的本地 token 数.

5. 最终性能数字必须来自 `callable` 模式. `deepgemm` 模式只是集成启动辅助, 因为它对外部脚本做 process 级计时.
