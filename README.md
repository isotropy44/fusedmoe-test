# Pangu 92B MoE 三 Case 对比实验

这个仓库用于验证和对比 Pangu 92B decode 场景下 MoE (Mixture of Experts) 从 dispatch 到 combine 的闭区间耗时.

要对比的 3 个 case:

1. `pangu_chain`: `npu_moe_distribute_dispatch_v2 -> npu_grouped_matmul -> npu_dequant_swiglu_quant -> npu_grouped_matmul -> npu_moe_distribute_combine_v2`
2. `vllm_base`: 未修改版 vLLM-Ascend fused MoE 算子
3. `vllm_modified`: 修改版 vLLM-Ascend fused MoE 算子

正式结论只看 A3. A2 只做预验证, 用来提前发现环境, artifacts, output 校验, 确定性和可视化链路问题. 别拿 A2 延迟去推 A3 结论, 硅片不会负责迁就这种乐观.

## 1. 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_pangu92_three_case_experiment.py` | 三 case artifacts, run, output check, plot 主入口 |
| `run_pangu92_decode_dispatch_combine_benchmark.py` | 三 case 主脚本复用的底层 Pangu / vLLM op 实现 |
| `RUNNING.md` | 从零搭建 A2 / A3 环境并运行实验 |
| `SOURCES.md` | 公开来源和本地代码证据 |
| `experiment_design.md` | 当前三 case 实验设计和读数说明 |

实验产物固定在:

`artifacts/pangu92_moe_weights_sync`

这个目录不会进入 git. 每次执行 `prepare` 都会覆盖旧 artifacts.

## 2. A2 预验证快速命令

A2 用 8 rank 小配置. 推荐 `num_experts=128`, 每 rank 16 experts.
A2 只验证环境, artifacts, `pangu_chain`, output 和画图链路. 如果 vLLM-Ascend 是 `SOC_VERSION=ascend910b1` 构建, 不要在 A2 上跑 `dispatch_gmm_combine_decode`; 该 fused decode Ascend Computing Language Neural Network (ACLNN) custom op 不会被打进 A2 custom op 包.

1. 生成固定 artifacts:

`python3 run_pangu92_three_case_experiment.py prepare --batch-size 24 --world-size 8 --num-experts 128 --top-k 8 --quant-mode 2 --synthetic-fallback`

2. 校验 artifacts:

`python3 run_pangu92_three_case_experiment.py verify-artifacts`

3. 跑 `pangu_chain`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 2 --dump-output --warmup 2 --repeat 5 --output artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

4. 可选: 如果当前环境确实安装了 A3 `ascend910_93` custom op 包, 才跑当前环境里的 vLLM fused 算子:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 2 --dump-output --warmup 2 --repeat 5 --output artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

5. 可选: 只有执行了上一步 vLLM fused case, 才对比 output:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --rtol 1e-2 --atol 1e-2`

6. 生成图. 如果只跑了 `pangu_chain`, 图里只会有一个 case, 这对 A2 smoke 是正常结果:

`python3 run_pangu92_three_case_experiment.py plot --input artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

Linux 桌面打开 PNG:

`xdg-open artifacts/pangu92_moe_weights_sync/plots/latency_summary.png`

纯 SSH 服务器查看 PNG:

`python3 -m http.server 8080 -d artifacts/pangu92_moe_weights_sync/plots`

然后在本地浏览器打开:

`http://服务器IP:8080/latency_summary.png`

## 3. A3 正式实验快速命令

A3 8 卡按 16 die 作为 16 rank 使用. 正式配置推荐 `num_experts=256`, 每 rank 16 experts.
A3 vLLM-Ascend 必须用 `SOC_VERSION=ascend910_93` 构建. 验证 custom op 时需要执行 `enable_custom_op()`, 并显式 import `vllm_ascend.vllm_ascend_C`; 只 import `vllm_ascend` 不足以证明 `torch.ops._C_ascend.dispatch_gmm_combine_decode` 已注册. 这不是玄学, 只是懒加载.

量化设置只有一个实验入口: `--quant-mode`, 默认值为 `2`. Pangu chain 中 `npu_moe_distribute_dispatch_v2` 使用这个值. `npu_dequant_swiglu_quant` 的 `quant_mode=1` 是该 op 自己的枚举, 用来复现 Pangu 链路里和 dispatch `quant_mode=2` 配套的 W8A8 (8-bit Weight, 8-bit Activation) decode 路径. vLLM fused case 仍显式传入同一个 `--quant-mode=2`.

1. 生成正式 artifacts:

`python3 run_pangu92_three_case_experiment.py prepare --batch-size 24 --world-size 16 --num-experts 256 --top-k 8 --quant-mode 2 --synthetic-fallback`

2. 校验 artifacts:

`python3 run_pangu92_three_case_experiment.py verify-artifacts`

3. 跑 `pangu_chain`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

4. 在未修改 vLLM-Ascend 环境跑 `vllm_base`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

5. 在修改版 vLLM-Ascend 环境跑 `vllm_modified`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_modified --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

6. output 校验:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --case vllm_modified --rtol 1e-2 --atol 1e-2`

7. 生成图:

`python3 run_pangu92_three_case_experiment.py plot --input artifacts/pangu92_moe_weights_sync/results/timing.csv`

## 4. 结果怎么看

关键输出:

`artifacts/pangu92_moe_weights_sync/results/timing.csv`

`artifacts/pangu92_moe_weights_sync/results/output_check.csv`

`artifacts/pangu92_moe_weights_sync/results/summary.csv`

`artifacts/pangu92_moe_weights_sync/plots/latency_boxplot.png`

`artifacts/pangu92_moe_weights_sync/plots/latency_summary.png`

`--dump-output` 生成的每个 case output 会记录 `artifact_hash` 和关键 config. `check-outputs` 会校验这些字段, 防止把旧 artifacts 的输出拿来对比. 人会偷懒, hash 不会.

正式 A3 对比只有这些条件都满足, 才能看 timing:

1. `verify-artifacts` 通过
2. 每个 case 的 `determinism_passed=True`, 且确定性校验使用 `rtol=1e-4`, `atol=1e-4`
3. `vllm_base` 和 `vllm_modified` 相对 `pangu_chain` 的 `output_allclose=True`

否则图和 CSV 只能当失败诊断材料, 不是性能结论.

## 5. 继续阅读

第一次接触 NPU (Neural Processing Unit) 的用户先读 `RUNNING.md`. 里面从 root 登录, 创建个人用户, 安装 CANN (Compute Architecture for Neural Networks), 创建 conda 环境, 安装 vLLM-Ascend, 到跑 A2 smoke 都写了. 很啰嗦, 但新手文档不啰嗦通常只是把痛苦推迟.
