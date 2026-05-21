# MoE Dispatch-Combine Benchmark 入口

## 1. 目录内容

本目录用于存放 A/B/C 三组 MoE (Mixture of Experts) dispatch-to-combine 对比实验的设计, 脚本和运行说明.

| 文件 | 作用 |
| --- | --- |
| `experiment_design.md` | 固定变量, 实验组定义, 计时窗口, 输出约束 |
| `benchmark_moe_dispatch_combine.py` | Python benchmark 脚本 |
| `vllm_ascend_fused_moe_adapter.py` | 直接调用 vLLM-Ascend fused MoE 算子的 adapter |
| `run_vllm_ascend_fused_moe_a3.py` | A3 上一键运行 vLLM-Ascend fused MoE 的 Python runner |
| `run_pangu92_decode_dispatch_combine_benchmark.py` | Pangu V2 92B decode dispatch-combine microbenchmark |
| `RUNNING.md` | Ascend A2/A3 与 CUDA 环境搭建, 以及运行命令 |
| `SOURCES.md` | 公开来源, 本地代码证据, 未确认假设 |

## 2. 快速检查

不依赖硬件的配置检查:

`python3 benchmark_moe_dispatch_combine.py --case A --mode plan`

在 A3 上真实调用 vLLM-Ascend fused MoE:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_vllm_ascend_fused_moe_a3.py --m 24 --world-size 8 --output results/b_vllm_ascend_fused.csv`

在 A3 上对比 Pangu V2 92B decode dispatch-combine 链路和 vLLM-Ascend fused MoE:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_decode_dispatch_combine_benchmark.py --batch-size 24 --world-size 16 --op-path both --synthetic-fallback --output results/pangu92_decode_compare_synth.csv`

接入真实 dispatch-to-combine callable 后运行 A/B/C:

`python3 benchmark_moe_dispatch_combine.py --case A --mode callable --backend npu --entrypoint my_moe_adapter:build_case`

`python3 benchmark_moe_dispatch_combine.py --case B --mode callable --backend npu --entrypoint my_moe_adapter:build_case`

`python3 benchmark_moe_dispatch_combine.py --case C --mode callable --backend cuda --entrypoint my_megamoe_adapter:build_case`

拿任何数字当结论前, 先读 `RUNNING.md`. 脚本很擅长输出小数, 但它不负责保证人类真的量到了该量的东西. 真体贴, 也真冷漠.
