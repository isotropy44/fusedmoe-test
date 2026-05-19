# A/B/C Benchmark 运行说明

## 1. 脚本作用

主脚本为:

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py`

它提供 3 种模式:

| 模式 | 用途 |
| --- | --- |
| `plan` | 打印精确配置和环境变量 , 不执行设备侧工作 |
| `callable` | 导入 Python callable , 使用 NPU 或 CUDA event 计时 , 是 A 和 B 的主模式 |
| `deepgemm` | 启动 DeepGEMM `tests/test_mega_moe.py` 跑 case C , 这是粗粒度集成检查 , 不是首选内层 event 计时 |

严格路径是 `callable`: 入口函数返回一个零参数 callable , callable 内部只包含 dispatch-to-combine . 脚本负责 warmup , repeat , device event , CSV (Comma-Separated Values) 写入和固定 metadata .

## 2. Ascend A2/A3 环境

截至 2026-05-19 , 推荐基础栈如下:

| 组件 | 推荐版本 |
| --- | --- |
| CANN (Compute Architecture for Neural Networks) | 8.5.0 |
| PyTorch | 2.7.1 |
| torch-npu | 2.7.1.post2 |
| Python | 3.10 或 3.11 |

理由: 本地 Ascend PyTorch README 将 CANN 8.5.0 映射到 PyTorch 2.7.1 加 torch-npu 2.7.1.post2 , 也支持 PyTorch 2.6.0 加 torch-npu 2.6.0.post5 . 除非目标 vLLM-Ascend 分支强制 pin 到 2.6.0 , 否则优先用 2.7.1 . 版本选择不是抽签 , 尽管有时看起来很像 .

最小搭建步骤:

1. 安装目标 OS 和 CPU 架构对应的 Ascend driver , firmware , CANN 8.5.0 packages .
2. 加载 CANN 环境:

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

3. 创建 Python 环境:

`conda create -n moe-a3 python=3.10 -y`

`conda activate moe-a3`

4. 安装与 CANN 8.5.0 和 CPU 架构匹配的 PyTorch 与 torch-npu wheel:

`pip install torch-2.7.1-*.whl`

`pip install torch_npu-2.7.1.post2-*.whl`

5. 安装 A/B callable 使用的项目 runtime , 通常是匹配分支的 vLLM-Ascend 或 omni-npu:

`pip install -r external_repos/vllm-ascend/requirements.txt`

`pip install -e external_repos/vllm-ascend`

6. 验证 NPU (Neural Processing Unit):

`python -c "import torch, torch_npu; print(torch.npu.is_available()); print(torch_npu.__version__)"`

在 A3 上跑 case B 时设置:

`export VLLM_ASCEND_ENABLE_FUSED_MC2=2`

跑 case A 时设置:

`export VLLM_ASCEND_ENABLE_FUSED_MC2=0`

脚本会按 case 自动设置这些变量 , 但显式 export 能让日志更容易读 . 人类还是需要一点点可读性 , 很麻烦 , 但没办法 .

## 3. CUDA MegaMoE 环境

DeepGEMM MegaMoE 推荐基础栈:

| 组件 | 推荐版本 |
| --- | --- |
| GPU (Graphics Processing Unit) | NVIDIA SM100 , 用于 MegaMoE 目标路径 |
| CUDA Toolkit | SM100 使用 12.9 或更新版本 |
| PyTorch | 2.9.0 或更新版本 |
| Python | 3.10 或 3.11 |
| DeepGEMM | `mega-update` 分支 , 或当前包含 MegaMoE 的分支 |

DeepGEMM 说明 SM100 需要 CUDA 12.9 或更高版本 , PyTorch 通用要求为 2.1 或更高版本 , MegaMoE symmetric memory 备注要求 PyTorch >= 2.9 . C 路径建议使用 PyTorch 2.9.0+ 和官方 CUDA 12.8 wheel , 同时保留 CUDA Toolkit 12.9+ 用于编译 . 这组合不优雅 , 但现实从来不负责审美 .

最小搭建步骤:

`conda create -n moe-cuda python=3.11 -y`

`conda activate moe-cuda`

`pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128`

`cd external_repos/DeepGEMM`

`git submodule update --init --recursive`

`./develop.sh`

`pip install -e .`

验证 CUDA:

`python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"`

验证 DeepGEMM import:

`python -c "import deep_gemm; print(deep_gemm.__file__)"`

## 4. 配置检查

`plan` 模式可在 macOS 或任意 host 上运行:

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case A --mode plan`

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case B --mode plan --m 24 --world-size 8`

`python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case C --mode plan --backend cuda`

## 5. 在 Ascend 上运行 A 和 B

创建一个 adapter module , 例如 `my_moe_adapter.py` , 其中提供:

`def build_case(config):`

它必须返回一个零参数 callable . 该 callable 只能执行:

`dispatch -> GMM1 -> dequant -> SwiGLU -> dynamic quant -> GMM2 -> dequant -> combine`

运行 case A:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case A --mode callable --backend npu --entrypoint my_moe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/a.csv`

运行 case B:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case B --mode callable --backend npu --entrypoint my_moe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/b.csv`

B 跑完后 , 必须检查日志是否显示命中 fused path . 对 vLLM-Ascend , 这意味着 `VLLM_ASCEND_ENABLE_FUSED_MC2=2` , 且实际调用链到达 `torch.ops._C_ascend.dispatch_gmm_combine_decode` .

## 6. 在 CUDA 上运行 C

首选 callable 模式:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case C --mode callable --backend cuda --entrypoint my_megamoe_adapter:build_case --m 24 --world-size 8 --warmup 20 --repeat 100 --output results/c.csv`

粗粒度 DeepGEMM 集成模式:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python moe_dispatch_combine_benchmark/benchmark_moe_dispatch_combine.py --case C --mode deepgemm --backend cuda --m 24 --world-size 8 --output results/c_deepgemm.csv`

最终对比使用 callable 模式 . `deepgemm` 模式只用于证明 CUDA 栈和 DeepGEMM MegaMoE 路径可以启动 .

## 7. 对比结果

所有 case 必须使用相同的 `--m` , `--world-size` , 模型字段 , warmup , repeat , 权重来源 .

主表:

| Case | Median ms | Mean ms | 说明 |
| --- | ---: | ---: | --- |
| A | TBD (To Be Determined) | TBD | Ascend 普通 MoE |
| B | TBD | TBD | Ascend fused MoE |
| C | TBD | TBD | CUDA MegaMoE |

只比较相同机器类型和相同不变量集合下的 `sample_ms` . A2/A3 和 CUDA 属于不同硬件家族 , 所以 A/B 是同平台 kernel 对比 , C 是跨平台参考 . 把 C 当架构参考 , 不要当道德审判 . 硅片没有兴趣照顾人的自尊心 .
