# MoE Dispatch-Combine Benchmark 入口

## 1. 目录内容

本目录用于存放 A/B/C 三组 MoE (Mixture of Experts) dispatch-to-combine 对比实验的设计, 脚本和运行说明.

| 文件 | 作用 |
| --- | --- |
| `experiment_design.md` | 固定变量, 实验组定义, 计时窗口, 输出约束 |
| `benchmark_moe_dispatch_combine.py` | Python benchmark 脚本 |
| `RUNNING.md` | Ascend A2/A3 与 CUDA 环境搭建, 以及运行命令 |
| `SOURCES.md` | 公开来源, 本地代码证据, 未确认假设 |

## 2. 快速检查

不依赖硬件的配置检查:

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case A --mode plan`

接入真实 dispatch-to-combine callable 后运行 A/B/C:

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case A --mode callable --backend npu --entrypoint my_moe_adapter:build_case`

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case B --mode callable --backend npu --entrypoint my_moe_adapter:build_case`

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case C --mode callable --backend cuda --entrypoint my_megamoe_adapter:build_case`

拿任何数字当结论前, 先读 `RUNNING.md`. 脚本很擅长输出小数, 但它不负责保证人类真的量到了该量的东西. 真体贴, 也真冷漠.
