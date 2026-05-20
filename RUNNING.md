# A/B/C Benchmark 运行说明

## 1. 脚本作用

主脚本为:

`python3 fusedmoe-test/benchmark_moe_dispatch_combine.py`

它提供 3 种模式:

| 模式 | 用途 |
| --- | --- |
| `plan` | 打印精确配置和环境变量, 不执行设备侧工作 |
| `callable` | 导入 Python callable, 使用 NPU 或 CUDA event 计时, 是 A 和 B 的主模式 |
| `deepgemm` | 启动 DeepGEMM `tests/test_mega_moe.py` 跑 case C, 这是粗粒度集成检查, 不是首选内层 event 计时 |

严格路径是 `callable`: 入口函数返回一个零参数 callable, callable 内部只包含 dispatch-to-combine. 脚本负责 warmup, repeat, device event, CSV 写入和固定 metadata. 

## 2. Ascend A2/A3 环境

推荐基础栈如下:

| 组件 | 推荐版本 |
| --- | --- |
| CANN (Compute Architecture for Neural Networks) | 9.0.0 |
| PyTorch | 2.8.0 |
| torch-npu | 2.8.0.post2 |
| Python | 3.10 或 3.11 |

最小搭建步骤:

1. 安装目标 OS 和 CPU 架构对应的 Ascend driver, firmware, CANN 9.0.0 packages. 
2. 加载 CANN 环境:

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

3. 创建 Python 环境:

`conda create -n moe-a3 python=3.10 -y`

`conda activate moe-a3`

4. 安装与 CANN 9.0.0 和 CPU 架构匹配的 PyTorch 与 torch-npu wheel:

vLLM-Ascend `releases/v0.13.0` 的 `requirements.txt` 明确固定:

`torch==2.8.0`

`torch-npu==2.8.0.post2`

所以这里不要再手工装 2.10.x. 直接按 vLLM-Ascend 依赖安装, 或安装同版本 wheel:

`pip install torch==2.8.0 torch-npu==2.8.0.post2`

5. 安装 A/B callable 使用的项目 runtime, 通常是匹配分支的 vLLM-Ascend 或 omni-npu: [vllm-ascend/releases/v0.13.0](https://github.com/vllm-project/vllm-ascend/tree/releases/v0.13.0)

`pip install -r vllm-ascend/requirements.txt`

`pip install -e vllm-ascend`

6. 验证 NPU:

`python3 -c "import torch, torch_npu; print(torch.npu.is_available()); print(torch.__version__); print(torch_npu.__version__)"`

在 A3 上跑 case B 时设置:

`export VLLM_ASCEND_ENABLE_FUSED_MC2=2`

跑 case A 时设置:

`export VLLM_ASCEND_ENABLE_FUSED_MC2=0`

脚本会按 case 自动设置这些变量, 但显式 export 能让日志更容易读. 人类还是需要一点点可读性, 很麻烦, 但没办法. 

## 3. CUDA MegaMoE 环境

DeepGEMM MegaMoE 推荐基础栈:

| 组件 | 推荐版本 |
| --- | --- |
| GPU | NVIDIA SM100, 用于 MegaMoE 目标路径 |
| CUDA Toolkit | SM100 使用 12.9 或更新版本 |
| PyTorch | 2.9.0 或更新版本 |
| Python | 3.10 或 3.11 |
| [MegaMoE](https://github.com/deepseek-ai/DeepGEMM/tree/mega-update) | `mega-update` 分支, 或当前包含 MegaMoE 的分支 |

DeepGEMM 说明 SM100 需要 CUDA 12.9 或更高版本, PyTorch 通用要求为 2.1 或更高版本, MegaMoE symmetric memory 备注要求 PyTorch >= 2.9. C 路径建议使用 PyTorch 2.9.0+ 和官方 CUDA 12.8 wheel, 同时保留 CUDA Toolkit 12.9+ 用于编译. 这组合不优雅, 但现实从来不负责审美.

最小搭建步骤:

`conda create -n moe-cuda python=3.11 -y`

`conda activate moe-cuda`

`pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128`

`cd repos/DeepGEMM`

`git submodule update --init --recursive`

`./develop.sh`

`pip install -e. `

验证 CUDA:

`python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"`

验证 DeepGEMM import:

`python -c "import deep_gemm; print(deep_gemm.__file__)"`

## 4. 配置检查

`plan` 模式可在 macOS 或任意 host 上运行:

`python fusedmoe-test/benchmark_moe_dispatch_combine.py --case A --mode plan`

`python fusedmoe-test/benchmark_moe_dispatch_combine.py --case B --mode plan --m 24 --world-size 8`

`python fusedmoe-test/benchmark_moe_dispatch_combine.py --case C --mode plan --backend cuda`

## 5. 在 Ascend A3 上真实运行 vLLM-Ascend fused MoE

本目录已经提供直接调用 vLLM-Ascend fused MoE 算子的脚本:

`run_vllm_ascend_fused_moe_a3.py`

它会调用:

`torch.ops._C_ascend.dispatch_gmm_combine_decode`

也就是 vLLM-Ascend `VLLM_ASCEND_ENABLE_FUSED_MC2=2` 路径中的 fused MoE decode 算子. 该算子内部覆盖 dispatch, GMM1, dequant, SwiGLU, dynamic quant, GMM2, dequant, combine. benchmark 计时只包住这个 op 调用, 初始化, 造输入, 造权重不进入计时窗口.

先确认 vLLM-Ascend 已经安装并能注册自定义算子:

`python3 -c "import torch, torch_npu; from vllm_ascend.utils import enable_custom_op; print(enable_custom_op()); print(hasattr(torch.ops._C_ascend, 'dispatch_gmm_combine_decode'))"`

一键运行 8 卡 A3:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 fusedmoe-test/run_vllm_ascend_fused_moe_a3.py --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/b_vllm_ascend_fused.csv`

单卡 smoke test 可以直接跑, 不会再绕 torchrun:

`ASCEND_RT_VISIBLE_DEVICES=0 python3 fusedmoe-test/run_vllm_ascend_fused_moe_a3.py --m 24 --world-size 1 --num-experts 16 --warmup 2 --repeat 5 --output /tmp/b_smoke.csv`

注意: 单卡 smoke test 不适合沿用 DeepSeek V4 的 `--num-experts 384`, 因为 fused op 的本地 expert 数有 tiling 限制. 单卡只用于验证 op 能被调用, 正式对比仍用 8 卡默认配置.

如果你已经在 torchrun 环境里, 也可以显式运行:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 fusedmoe-test/run_vllm_ascend_fused_moe_a3.py --no-torchrun --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/b_vllm_ascend_fused.csv`

可改参数:

| 参数或环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `--m` | `24` | 本 rank 输入 token 数 |
| `--world-size` | `8` | NPU 数, 必须等于 torchrun 进程数 |
| `--hidden-size` | `7168` | DeepSeek V4 参考 hidden size |
| `--moe-intermediate-size` | `3072` | DeepSeek V4 参考 MoE intermediate size |
| `--num-experts` | `384` | DeepSeek V4 参考 expert 数 |
| `--num-experts-per-tok` | `6` | top-k |
| `FMOE_ASCEND_DTYPE` | `bf16` | 可设为 `bf16` 或 `fp16` |
| `FMOE_ASCEND_QUANT_MODE` | `0` | 传给 fused op 的 quant_mode |
| `FMOE_ASCEND_WEIGHT_LAYOUT` | `single` | 可设为 `single` 或 `list`; `single` 对应 vLLM 常规非 dynamic EPLB 权重布局 |

脚本在多 rank 下会对每轮 device event 耗时做 `all_reduce(max)`, rank 0 写 CSV. 这是集体通信算子的合理统计方式, 不然拿 rank 0 的单点时间冒充全局时间, 只是更精致的自欺.

## 6. 在 Ascend 上运行自定义 A 和 B

创建一个 adapter module, 例如 `my_moe_adapter.py`, 其中提供:

`def build_case(config):`

它必须返回一个零参数 callable. 该 callable 只能执行:

`dispatch -> GMM1 -> dequant -> SwiGLU -> dynamic quant -> GMM2 -> dequant -> combine`

运行 case A:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fusedmoe-test/benchmark_moe_dispatch_combine.py --case A --mode callable --backend npu --entrypoint my_moe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/a.csv`

运行 case B:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fusedmoe-test/benchmark_moe_dispatch_combine.py --case B --mode callable --backend npu --entrypoint my_moe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/b.csv`

B 跑完后, 必须检查日志是否显示命中 fused path. 对 vLLM-Ascend, 这意味着 `VLLM_ASCEND_ENABLE_FUSED_MC2=2`, 且实际调用链到达 `torch.ops._C_ascend.dispatch_gmm_combine_decode`. 

## 7. 在 CUDA 上运行 C

首选 callable 模式:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fusedmoe-test/benchmark_moe_dispatch_combine.py --case C --mode callable --backend cuda --entrypoint my_megamoe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/c.csv`

粗粒度 DeepGEMM 集成模式:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fusedmoe-test/benchmark_moe_dispatch_combine.py --case C --mode deepgemm --backend cuda --m 24 --world-size 8 --output results/c_deepgemm.csv`

最终对比使用 callable 模式. `deepgemm` 模式只用于证明 CUDA 栈和 DeepGEMM MegaMoE 路径可以启动. 

## 8. 对比结果

所有 case 必须使用相同的 `--m`, `--world-size`, 模型字段, warmup, repeat, 权重来源. 

主表:

| Case | Median ms | Mean ms | 说明 |
| --- | ---: | ---: | --- |
| A | TBD (To Be Determined) | TBD | Ascend 普通 MoE |
| B | TBD | TBD | Ascend fused MoE |
| C | TBD | TBD | CUDA MegaMoE |

只比较相同机器类型和相同不变量集合下的 `sample_ms`. A2/A3 和 CUDA 属于不同硬件家族, 所以 A/B 是同平台 kernel 对比, C 是跨平台参考. 把 C 当架构参考, 不要当道德审判. 硅片没有兴趣照顾人的自尊心. 
