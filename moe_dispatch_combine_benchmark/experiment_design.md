# MoE Dispatch-Combine Benchmark 设计

## 1. 结论

本目录定义一个可复现的 A/B/C 实验 , 用于比较 MoE (Mixture of Experts) 从 dispatch 到 combine 闭区间的耗时 .

A 是 Ascend 普通 MoE . B 是 Ascend fused MoE . C 是 CUDA (Compute Unified Device Architecture) MegaMoE . 唯一指标是 dispatch 开始到 combine 结束的闭区间 elapsed milliseconds . 其他变量 , 包括模型 shape , 权重来源 , 量化策略 , token 数 , world size , warmup , repeat , 都必须保持不变 . 实验设计不需要浪漫 , 需要无聊地诚实 .

默认模型 proxy 使用 DeepSeek V4 Pro Base 公开 config:

| 字段 | 数值 |
| --- | ---: |
| `hidden_size` | 7168 |
| `moe_intermediate_size` | 3072 |
| `n_routed_experts` | 384 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 6 |
| `num_hidden_layers` | 61 |
| `num_hash_layers` | 3 |
| `max_position_embeddings` | 1048576 |

默认 benchmark 输入为每 rank 本地 token 数 `M=24` , 可通过 `--m` 修改 . 默认卡数为 8 , 可通过 `--world-size` 修改 .

## 2. 公共不变量

共享 MoE 流程来自 `../NOTES/moe-stream.svg`:

`MLA layer -> norm -> matmul -> router -> dispatch -> GMM1 -> dequant -> SwiGLU -> dynamic quant -> GMM2 -> dequant -> combine -> norm`

计时窗口从 dispatch 之前立即开始 , 到 combine 之后立即结束 . Router , top-k , shared expert , residual , normalization , attention , tokenizer , sampling , KV (Key Value) cache 都不进入计时窗口 .

A/B/C 必须使用相同 routed expert 输入:

| 不变量 | 设置 |
| --- | --- |
| Local M | 默认 `24` |
| World size | 默认 `8` |
| Hidden size | `7168` |
| Expert intermediate size | `3072` |
| Routed experts | `384` |
| Top-k | `6` |
| Expert weights | 相同随机 seed 生成 , 或相同 checkpoint slice |
| Quantization | 同一轮实验保持一致 |
| Precision record | 必须写入结果 metadata |
| Metric | 只统计 dispatch-to-combine 闭区间 |

如果本地没有真实 DeepSeek V4 权重 , 合法 fallback 是固定 seed 的确定性 synthetic tensor . 这个 fallback 只能比较 kernel 机制 , 不能比较模型质量 . 假装两者等价 , 大概是另一种形式的民俗学 .

## 3. 实验组

| Case | 硬件 | 路径 | Dispatch-to-combine 算子链 |
| --- | --- | --- | --- |
| A | Ascend A2/A3 NPU (Neural Processing Unit) | 普通 MoE | EP (Expert Parallelism) collective 加独立 routing , GMM (Grouped Matrix Multiplication) , activation , quant , finalize , combine op |
| B | Ascend A2/A3 NPU | Fused MoE | A3 , W8A8 (8-bit Weight , 8-bit Activation) dynamic , decode , `VLLM_ASCEND_ENABLE_FUSED_MC2=2` 等条件满足时使用 `dispatch_gmm_combine_decode` |
| C | CUDA GPU (Graphics Processing Unit) | MegaMoE | DeepGEMM `fp8_fp4_mega_moe` , 将 EP dispatch , L1 , SwiGLU , L2 , EP combine 融合进单个 mega-kernel |

A 和 B 使用相同 Ascend runtime 与相同 checkpoint 输入 . Case A 应强制非 fused 路径 , 例如 `VLLM_ASCEND_ENABLE_FUSED_MC2=0` . Case B 应强制目标 fused 路径 , 例如 `VLLM_ASCEND_ENABLE_FUSED_MC2=2` , 并且运行后必须确认确实命中 fused op . Case C 使用 DeepGEMM MegaMoE , 必须记录 CUDA , driver , PyTorch , DeepGEMM commit .

## 4. 计时方法

内部计时使用 device event , 不使用 wall clock:

1. 构造或加载相同的 `x` , `topk_ids` , `topk_weights` , expert weights , quant scales .
2. 使用 `--warmup` 预热目标路径 .
3. 在 dispatch 前立即记录 device event .
4. 只执行 dispatch-to-combine 路径 .
5. 在 combine 后立即记录 device event .
6. 同步 device .
7. 重复 `--repeat` 次 .

随附 Python harness 对导入的 callable 实现这套计时约束 . 导入的 callable 必须只包含 dispatch-to-combine 路径 . 如果 callable 包含 router 或 post-MoE 工作 , 结果无效 . 额外测了东西还说自己没测 , 这不叫优化 , 叫自我安慰 .

## 5. 输出

每次运行追加 CSV (Comma-Separated Values) 行:

`case, backend, world_size, m, hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, sample_ms, mean_ms, median_ms, min_ms, max_ms, p90_ms, p99_ms`

同一不变量集合下 , 主要比较值只看 `mean_ms` 或 `median_ms` . 派生 speedup 后续计算:

`speedup(A,B) = A_median_ms / B_median_ms`

`speedup(A,C) = A_median_ms / C_median_ms`

## 6. 本地代码证据

`../external_repos/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:319` 为 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` 选择 `dispatch_ffn_combine` , `:334` 到 `:350` 为值 `2` 调用 `torch.ops._C_ascend.dispatch_gmm_combine_decode` .

`../external_repos/vllm-ascend/vllm_ascend/envs.py:130` 到 `:138` 说明 fused MC2 开关 , 并说明 `dispatch_gmm_combine_decode` 用于 W8A8 decode node MoE layer .

`../external_repos/vllm-ascend/vllm_ascend/ascend_forward_context.py:257` 到 `:279` 展示 A3 选择条件: fused MC2 需要 W8A8 dynamic , decode 分支中值 `2` 只有在额外 speculative 检查通过时才映射到 `dispatch_gmm_combine_decode` .

`../external_repos/DeepGEMM/README.md:114` 到 `:140` 说明 MegaMoE 将 EP dispatch , linear 1 , SwiGLU , linear 2 , EP combine 融合为单个 mega-kernel .

`../external_repos/DeepGEMM/tests/test_mega_moe.py:263` 到 `:275` 给出 DeepGEMM 默认进程数和模型参数 , 包括默认 `--num-processes=8` .
