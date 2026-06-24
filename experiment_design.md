# Pangu 92B MoE 三 Case 实验设计

## 1. 实验目标

本实验比较 Pangu 92B decode 场景下 MoE (Mixture of Experts) 从 dispatch 到 combine 的闭区间耗时. 当前只保留 3 个 case:

| Case | 方案 | `run-case` 参数 | 算子链 |
| --- | --- | --- | --- |
| `pangu_chain` | Pangu 现有 5 步链路 | `--case-name pangu_chain --op-path pangu` | `npu_moe_distribute_dispatch_v2 -> npu_grouped_matmul -> npu_dequant_swiglu_quant -> npu_grouped_matmul -> npu_moe_distribute_combine_v2` |
| `vllm_base` | 未修改版 vLLM-Ascend fused MoE | `--case-name vllm_base --op-path vllm` | `torch.ops._C_ascend.dispatch_gmm_combine_decode` |
| `vllm_modified` | 修改版 vLLM-Ascend fused MoE | `--case-name vllm_modified --op-path vllm` | `torch.ops._C_ascend.dispatch_gmm_combine_decode` |

`pangu_chain` 是数值 golden. `vllm_base` 和 `vllm_modified` 必须先通过各自 case 的确定性校验, 再与 `pangu_chain` 做 output allclose. 只有这些校验都通过, timing 才能作为性能结论.

正式性能结论只看 A3. A2 只做预验证, 用于确认环境, artifacts, `pangu_chain`, output dump 和绘图链路可以跑通. 如果 A2 环境只用 `SOC_VERSION=ascend910b1` 构建 vLLM-Ascend, 不强求 `dispatch_gmm_combine_decode`.

## 2. 公共不变量

三 case 必须使用同一份 artifacts. 固定目录为:

`artifacts/pangu92_moe_weights_sync`

`prepare` 每次都会覆盖该目录, 并写入 `metadata.json`, rank 输入 tensor, tensor hash 和 `artifact_hash`. 后续 `verify-artifacts`, `run-case`, `check-outputs` 都围绕同一份 metadata 校验.

默认 Pangu 92B MoE 配置来自 `run_pangu92_decode_dispatch_combine_benchmark.py`:

| 字段 | 默认值 |
| --- | ---: |
| `batch_size` | 24 |
| `world_size` | 16 |
| `hidden_size` | 2560 |
| `moe_intermediate_size` | 1024 |
| `num_experts` | 256 |
| `num_shared_experts` | 1 |
| `top_k` | 8 |
| `quant_mode` | 2 |

A3 正式实验推荐 `world_size=16`, `num_experts=256`, 即每 rank 16 个 local experts. A2 smoke test 推荐 `world_size=8`, `num_experts=128`, 同样保持每 rank 16 个 local experts.

必须保持一致的变量:

| 不变量 | 约束 |
| --- | --- |
| 输入 tensors | `hidden_states`, `topk_ids`, `topk_weights`, `quant_scale` 来自同一份 artifacts |
| 权重 tensors | `w13_weight`, `w2_weight`, `w13_weight_scale`, `w2_weight_scale` 来自同一份 artifacts |
| Shape | `batch_size`, `world_size`, `hidden_size`, `moe_intermediate_size`, `num_experts`, `top_k` 一致 |
| 量化 | `quant_mode` 一致. Pangu 链路中 `npu_dequant_swiglu_quant` 的 `quant_mode=1` 是该 op 自身枚举 |
| 权重来源 | 全部使用 real checkpoint slice, 或全部使用 deterministic synthetic fallback |
| Rank 映射 | `ASCEND_RT_VISIBLE_DEVICES` 与 torchrun world size 对齐 |
| 计时窗口 | 只统计 dispatch-to-combine 闭区间, 不包含 gate, router, shared expert, final add |

## 3. 公共执行阶段

一次完整实验由 5 个阶段组成:

1. `prepare`: 生成固定 artifacts, 并写入 `metadata.json` 与 `artifact_hash`.
2. `verify-artifacts`: 重新读取每个 rank 的 tensor, 校验 tensor hash 与 metadata 一致.
3. `run-case`: 单独运行一个 case. 脚本先做确定性校验, 确定性通过后才 dump output 并计时. 指定 `--profile` 时改为 warmup 后采集 device trace, 不执行普通计时.
4. `check-outputs`: 以 `pangu_chain` 为 golden, 对 `vllm_base` 和 `vllm_modified` 的 output 做 allclose 校验.
5. `plot`: 汇总 timing CSV, 生成 `summary.csv`, boxplot 和 latency summary 图.

`run-case` 内部的确定性校验逻辑是: 对同一 operation 同步执行多次, 第一轮输出作为 reference, 后续输出与 reference 按 `rtol=1e-4`, `atol=1e-4` 比较. 各 rank 的最大误差通过 distributed max reduce 汇总. 若 `determinism_passed=False`, 脚本直接报错, 不写入该 case 的 timing.

跨 case output 校验默认仍使用 `rtol=1e-2`, `atol=1e-2`. 它验证的是不同实现之间的 bf16 decode 输出一致性, 和同一 case 内的确定性校验不是同一个阈值.

## 4. 单 case 测试设计

### 4.1 `pangu_chain` 单独测试

目的:

验证 Pangu 现有 dispatch-to-combine 链路可以在当前 artifacts 和 rank 配置下稳定运行, 并生成后续 vLLM output 校验所需的 golden output.

前置条件:

1. 已执行 `prepare`.
2. `verify-artifacts` 通过.
3. `ASCEND_RT_VISIBLE_DEVICES` 与 `metadata.json` 中的 `world_size` 匹配.

命令形态:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

通过标准:

1. `determinism_passed=True`.
2. `outputs/pangu_chain/rank<N>.pt` 写出完整, 且包含当前 `artifact_hash`, 关键 config, output 和 `output_hash`.
3. `timing.csv` 中出现 `case_name=pangu_chain` 的样本行.

`pangu_chain` 不需要与其他 case 做 output allclose; 它是 golden. 若它本身不确定, 整轮实验无效.

### 4.2 `vllm_base` 单独测试

目的:

验证未修改版 vLLM-Ascend fused MoE 在同一份 artifacts 上可运行, 可重复, 并且 output 与 `pangu_chain` 等价.

前置条件:

1. 已经完成 `pangu_chain` 单独测试并 dump golden output. 
2. 当前 Python 环境安装的是未修改版 vLLM-Ascend. 
3. A3 正式实验中 vLLM-Ascend 需要以 `SOC_VERSION=ascend910_93` 构建, 并且 `torch.ops._C_ascend.dispatch_gmm_combine_decode` 已注册. 

命令形态:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

随后只校验这个 case:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --rtol 1e-2 --atol 1e-2`

通过标准:

1. `determinism_passed=True`. 
2. `outputs/vllm_base/rank<N>.pt` 的 `artifact_hash` 与当前 metadata 一致. 
3. `output_check.csv` 中 `case_name=vllm_base` 的 `output_allclose=True`. 

若确定性通过但 output 不一致, 该 case 的 latency 只能用于诊断, 不能用于说明 fused 方案性能收益. 

### 4.3 `vllm_modified` 单独测试

目的:

验证修改版 vLLM-Ascend fused MoE 在同一份 artifacts 上可运行, 可重复, 并且 output 与 `pangu_chain` 等价. 

前置条件:

1. 已经完成 `pangu_chain` 单独测试并 dump golden output. 
2. 当前 Python 环境安装的是修改版 vLLM-Ascend. 
3. A3 正式实验中 custom op 注册和 A3 `SOC_VERSION=ascend910_93` 构建要求与 `vllm_base` 相同. 

命令形态:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_modified --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

随后只校验这个 case:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_modified --rtol 1e-2 --atol 1e-2`

通过标准:

1. `determinism_passed=True`. 
2. `outputs/vllm_modified/rank<N>.pt` 的 `artifact_hash` 与当前 metadata 一致. 
3. `output_check.csv` 中 `case_name=vllm_modified` 的 `output_allclose=True`. 

`vllm_modified` 的核心比较对象是 `vllm_base`, 但数值正确性仍然只对齐 `pangu_chain` golden. 

## 5. 三 case 纵向测试设计

三 case 纵向测试使用同一份 artifacts, 同一组参数, 同一个 timing CSV, 按 case 依次追加结果. 推荐顺序是:

1. 生成 artifacts:

`python3 run_pangu92_three_case_experiment.py prepare --batch-size 24 --world-size 16 --num-experts 256 --top-k 8 --quant-mode 2 --synthetic-fallback`

2. 校验 artifacts:

`python3 run_pangu92_three_case_experiment.py verify-artifacts`

3. 在 Pangu 环境运行 `pangu_chain`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

4. 在未修改版 vLLM-Ascend 环境运行 `vllm_base`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

注意: `--case-name vllm_base` 只是结果标签. 真正运行的是当前 Python 环境里安装的 vLLM-Ascend. 运行后必须核对终端摘要或 `timing.csv` 里的 `vllm_ascend_commit`, 否则把修改版环境标成 base, 脚本也不会替人类羞愧.

5. 在修改版 vLLM-Ascend 环境运行 `vllm_modified`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_modified --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

6. 一次性校验两个 vLLM case 的 output:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --case vllm_modified --rtol 1e-2 --atol 1e-2`

`check-outputs` 会重写 `results/output_check.csv`. 因此三 case 纵向汇总前不要分别运行两次单 case output check 后直接 `plot`; 最终应使用上面这条命令一次性写入两个 vLLM case 的校验结果. 

7. 生成汇总和图:

`python3 run_pangu92_three_case_experiment.py plot --input artifacts/pangu92_moe_weights_sync/results/timing.csv`

纵向测试的关键是不要在三个 case 之间重新执行 `prepare`. 如果不同 case 必须在不同 Python 环境或不同 vLLM-Ascend checkout 下运行, 需要共享或复制同一份 `artifacts/pangu92_moe_weights_sync`, 并保持 `metadata.json`, inputs, outputs 与 `timing.csv` 属于同一轮 `artifact_hash`.

## 6. 如何阅读实验结果

关键输出文件:

| 文件 | 含义 |
| --- | --- |
| `results/timing.csv` | 每个 case 每次 repeat 的样本行, 同时带 summary, 确定性结果, artifact hash 和环境信息 |
| `results/output_check.csv` | `vllm_base`, `vllm_modified` 相对 `pangu_chain` 的 output allclose 结果 |
| `results/output_check_ranks.csv` | output check 的 rank 级明细, 用于定位失败 rank 和最大误差 |
| `results/summary.csv` | `plot` 后生成的按 case 汇总表, 包含 mean, median, p90, p99, validity 和 speedup |
| `plots/latency_boxplot.png` | 各 case 的 sample_ms 分布, 用于看波动和离群点 |
| `plots/latency_summary.png` | 各 case 的 median, p90, p99 柱状图, 并标注 valid 或 invalid |
| `outputs/<case>/rank<N>.pt` | `--dump-output` 保存的每 rank output, `artifact_hash`, 关键 config 和 `output_hash` |
| `profiles/<case>/` | `torch_npu.profiler` 生成的多 rank device profiling 数据 |

正式读数顺序:

1. 确认 `verify-artifacts` 通过.
2. 确认 3 个 case 的 `determinism_passed=True`.
3. 确认 `output_check.csv` 中 `vllm_base` 和 `vllm_modified` 的 `output_allclose=True`.
4. 确认 3 个 case 的 `artifact_hash`, shape, `top_k`, `quant_mode`, `world_size` 一致.
5. 以上都满足后, 再比较 `median_ms`, 再用 `p90_ms`, `p99_ms` 判断尾延迟.

`sample_ms` 是每次 repeat 的 dispatch-to-combine 闭区间耗时. `median_ms` 是中位数, 代表典型延迟. `p90_ms` 是第 90 百分位延迟, 意味着大约 90 % 的样本不超过这个值. `p99_ms` 是第 99 百分位延迟, 意味着大约 99 % 的样本不超过这个值.

直观说, median 看平时表现, p90 看较慢但常见的尾部, p99 看极慢尾部. 如果 `vllm_modified` 的 median 变小但 p90 或 p99 明显变大, 它可能只是把平均体验包装得更好看, 同时把尾延迟问题藏到了角落里. 这种优化很会化妆, 但不一定值得上线.

当前脚本使用样本排序后的近似百分位: `p90_ms=sorted_samples[int((n - 1) * 0.90)]`, `p99_ms=sorted_samples[int((n - 1) * 0.99)]`. 因此 `repeat` 太小时, p90 和 p99 的分辨率会很差. 正式实验建议 `repeat >= 100`.

`summary.csv` 中的 speedup 字段按 median 计算:

| 字段 | 公式 | 读法 |
| --- | --- | --- |
| `speedup_vs_pangu_chain_median` | `pangu_chain.median_ms / case.median_ms` | 大于 1 表示该 case 比 Pangu 链路快 |
| `speedup_vs_vllm_base_median` | `vllm_base.median_ms / case.median_ms` | 对 `vllm_modified` 大于 1 表示修改版比未修改版快 |

profiling 模式只用于拆解 device kernel 和通信任务. 它在 `--warmup` 后采集 `--profile-repeat` 轮, 未指定时使用 `--repeat`. profiling 采集会引入额外开销, 因此 trace 数据不能替代关闭 profiling 后的 event timing 结论.

建议主要比较:

1. `vllm_modified.median_ms` vs `vllm_base.median_ms`: 判断修改版是否带来直接收益.
2. `vllm_base.median_ms` vs `pangu_chain.median_ms`: 判断未修改 fused MoE 相对 Pangu 链路的基线差异.
3. `vllm_modified.p90_ms` 和 `vllm_modified.p99_ms`: 判断修改版是否引入尾延迟波动.

任一 case 确定性失败, output allclose 失败, 或 artifacts 不一致, timing 都只能作为失败诊断材料, 不能作为性能结论.

## 8. CSV 追加保护

三 case 通常在不同环境里依次向同一个 `timing.csv` 追加样本. 为避免上一轮写入的最后一行没有 newline 导致下一轮 case name 拼进前一行, 脚本在追加前会做三件事:

1. 如果文件尾部没有 `\n` 或 `\r`, 自动补一个 `\n`.
2. 校验 header 必须等于当前脚本定义的 timing 字段.
3. 校验已有每一行的列数和 `case_name`.

如果已有 CSV 已经被拼坏, 脚本会直接报错, 不会继续把坏数据滚大. 这种情况下应换一个新的 `--output` 文件, 或人工修复旧 CSV 后再追加. 数据坏了还继续画图, 只是在给错误做排版.
