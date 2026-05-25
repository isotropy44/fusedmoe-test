# A2 / A3 NPU 运行说明

本文面向第一次接触 NPU (Neural Processing Unit) 的用户. 目标不是优雅, 是让你能把实验跑起来, 并且知道每一步失败时该看哪里.

## 1. 基本概念

NPU 是昇腾 AI 处理器. A2 和 A3 是不同机器平台. A2 这次只做预验证, A3 做正式性能结论.

rank 是分布式进程编号. 8 rank 表示启动 8 个进程. A3 8 卡通常可以按 16 die 暴露为 16 rank. MoE (Mixture of Experts) 的 expert 会按 rank 切分, 所以 `num_experts / world_size` 是每 rank 本地 expert 数.

EP (Expert Parallel) 是 expert 并行域. 实验要求 3 个 case 的 `world_size`, rank 映射, `ASCEND_RT_VISIBLE_DEVICES`, `num_experts`, `top_k`, `quant_mode` 一致. 不同进程里的 `group_ep` 字符串不要求相同, 只要求当前环境可创建并可用.

## 2. 推荐版本

vLLM-Ascend `releases/v0.18.0` 按官方安装文档使用:

| 组件 | 版本 |
| --- | --- |
| CANN (Compute Architecture for Neural Networks) | 9.0.0 |
| NNAL | 9.0.0 |
| Python | >= 3.10, < 3.12 |
| PyTorch | 2.9.0 |
| torch-npu | 2.9.0.post2 |
| vLLM | v0.18.0 |
| vLLM-Ascend | releases/v0.18.0 |

先按这些版本跑通. 不要一上来就混版本, 那不是自由, 是噪声.

## 3. root 登录后检查机器

root 登录:

`whoami`

确认 NPU 可见:

`npu-smi info`

如果这里看不到卡, 不要继续装 Python 包. 先处理 driver, firmware, device 权限.

## 4. 创建个人用户

以下假设用户名是 `jhyang`:

`useradd -m -s /bin/bash jhyang`

`passwd jhyang`

如果需要 sudo 权限, openEuler / CentOS 常见命令:

`usermod -aG wheel jhyang`

Ubuntu 常见命令:

`usermod -aG sudo jhyang`

创建工作目录:

`mkdir -p /home/jhyang/workcode`

`chown -R jhyang:jhyang /home/jhyang`

切到个人用户:

`su - jhyang`

后续除系统依赖安装外, 都在个人用户下执行.

## 5. 个人目录安装 CANN 9.0.0 和 NNAL 9.0.0

创建下载目录:

`mkdir -p /home/jhyang/Ascend/downloads`

`cd /home/jhyang/Ascend/downloads`

下载 CANN 包. 文件名会随 CPU 架构不同而变化, 常见架构是 `x86_64` 或 `aarch64`:

`ARCH=$(uname -i)`

`wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-${ARCH}.run"`

`wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-${ARCH}.run"`

`wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-nnal_9.0.0_linux-${ARCH}.run"`

安装:

`chmod +x Ascend-cann-*.run`

`./Ascend-cann-toolkit_9.0.0_linux-${ARCH}.run --full --install-path=/home/jhyang/Ascend/cann-9.0.0`

`source /home/jhyang/Ascend/cann-9.0.0/ascend-toolkit/set_env.sh`

`./Ascend-cann-910b-ops_9.0.0_linux-${ARCH}.run --install --install-path=/home/jhyang/Ascend/cann-9.0.0`

`./Ascend-cann-nnal_9.0.0_linux-${ARCH}.run --install --install-path=/home/jhyang/Ascend/cann-9.0.0`

`source /home/jhyang/Ascend/cann-9.0.0/nnal/atb/set_env.sh`

写入 `~/.bashrc`:

`echo 'source /home/jhyang/Ascend/cann-9.0.0/ascend-toolkit/set_env.sh' >> ~/.bashrc`

`echo 'source /home/jhyang/Ascend/cann-9.0.0/nnal/atb/set_env.sh' >> ~/.bashrc`

如果安装器不接受 `--install-path`, 就让 root 安装到默认路径 `/usr/local/Ascend`, 然后把对应 `set_env.sh` 写入个人用户 `~/.bashrc`.

## 6. 创建 conda 环境

`conda create -n vllm-a2-018 python=3.11 -y`

`conda activate vllm-a2-018`

`pip install -U pip setuptools wheel`

安装画图依赖:

`pip install matplotlib`

如果系统缺编译工具, root 安装.

Ubuntu:

`apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq`

openEuler / CentOS:

`yum install -y gcc gcc-c++ cmake numactl-devel wget git curl jq`

## 7. 安装未修改版 vLLM-Ascend

拉代码:

`cd /home/jhyang/workcode`

`git clone --branch v0.18.0 https://github.com/vllm-project/vllm.git`

`git clone --branch releases/v0.18.0 https://github.com/vllm-project/vllm-ascend.git`

安装 vLLM:

`cd /home/jhyang/workcode/vllm`

`VLLM_TARGET_DEVICE=empty pip install -v -e .`

安装 vLLM-Ascend:

`cd /home/jhyang/workcode/vllm-ascend`

`git submodule update --init --recursive`

A3 正式实验必须按 A3 SoC 构建:

`SOC_VERSION=ascend910_93 pip install -v -e .`

如果遇到 build isolation 导致依赖环境混乱, 用:

`SOC_VERSION=ascend910_93 pip install --no-build-isolation -v -e .`

如果 CANN 头文件报 `profiling/prof_api.h` 找不到, 先确认 `pkg_inc` 存在, 再把它加入编译 include path:

`export ASCEND_HOME=/usr/local/Ascend/cann-9.0.0`

`export C_INCLUDE_PATH="${ASCEND_HOME}/$(uname -m)-linux/pkg_inc:${C_INCLUDE_PATH:-}"`

`export CPLUS_INCLUDE_PATH="${ASCEND_HOME}/$(uname -m)-linux/pkg_inc:${CPLUS_INCLUDE_PATH:-}"`

如果 CANN 安装在个人目录, 把 `ASCEND_HOME` 改成 `/home/jhyang/Ascend/cann-9.0.0`. 这不高雅, 但比改系统 include 目录更像人类工程.

A2 预验证如果只安装 `SOC_VERSION=ascend910b1`, 只能验证环境, artifacts, `pangu_chain` 和普通路径. `dispatch_gmm_combine_decode` 的 Ascend Computing Language Neural Network (ACLNN) custom op 只在 A3 `ascend910_93` 构建分支打包. 在 A2 上强跑 vLLM fused decode case, 常见结果是 Python op schema 可见, 运行时 `libopapi.so` 找不到 `aclnnDispatchGmmCombineDecode` 符号.

## 8. 验证环境

确认版本:

`python3 -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available())"`

期望看到 PyTorch `2.9.0`, torch-npu `2.9.0.post2`, NPU 可用.

确认 custom op. 这一步用于 A3 fused decode 实验. 如果你在 A2 上只做 `pangu_chain` smoke, 可以跳过:

`python3 -c "import importlib, torch, torch_npu; from vllm_ascend.utils import enable_custom_op; print(enable_custom_op()); importlib.import_module('vllm_ascend.vllm_ascend_C'); print(hasattr(torch.ops._C_ascend, 'dispatch_gmm_combine_decode'))"`

A3 环境期望两个输出都是 `True`. A2 `SOC_VERSION=ascend910b1` 环境里 Python op schema 可能可见, 但运行时仍可能因为 custom ops 包没有 `aclnnDispatchGmmCombineDecode` 而失败. 这不是你的手太笨, 是包里没有那块肉.

如果后续运行时报 custom op 符号找不到, source vLLM-Ascend 安装产物里的 custom op 环境:

`source $(python3 -c "import pathlib, vllm_ascend; print(next(pathlib.Path(vllm_ascend.__file__).resolve().parent.glob('_cann_ops_custom/**/set_env.bash')))")`

## 9. 拉取实验仓库

`cd /home/jhyang/workcode`

SSH:

`git clone git@github.com:isotropy44/fusedmoe-test.git`

HTTPS:

`git clone https://github.com/isotropy44/fusedmoe-test.git`

`cd /home/jhyang/workcode/fusedmoe-test`

## 10. A2 预验证

A2 只有 8 rank, 推荐先用 `num_experts=128`, 每 rank 16 experts.

如果使用 `--model-path` 加载真实权重, `prepare` 会先判定整次实验使用 real 还是 synthetic. 判定后所有 rank 必须使用同一种来源. 不允许一部分 rank real, 一部分 rank synthetic, 这种混搭只会制造精致的废数据.

生成 artifacts:

`python3 run_pangu92_three_case_experiment.py prepare --batch-size 24 --world-size 8 --num-experts 128 --top-k 8 --quant-mode 2 --synthetic-fallback`

校验 artifacts:

`python3 run_pangu92_three_case_experiment.py verify-artifacts`

跑 `pangu_chain`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 2 --dump-output --warmup 2 --repeat 5 --output artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

不要把 A2 smoke 当作 `vllm_base` fused decode 验证. 如果当前环境是 `SOC_VERSION=ascend910b1` 构建, 这一步应跳过:

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 2 --dump-output --warmup 2 --repeat 5 --output artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

只有在该环境确实安装了包含 `DispatchGmmCombineDecode` ACLNN op 的 A3 custom op 包时, 才运行上面这条命令和后续 output 校验. 如果跳过 vLLM fused case, 也跳过这条 output 校验:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --rtol 1e-2 --atol 1e-2`

生成 PNG. 如果只跑了 `pangu_chain`, 图里只有一个 case 是正常的 A2 预验证结果:

`python3 run_pangu92_three_case_experiment.py plot --input artifacts/pangu92_moe_weights_sync/results/a2_smoke.csv`

Linux 桌面打开:

`xdg-open artifacts/pangu92_moe_weights_sync/plots/latency_summary.png`

纯 SSH 查看:

`python3 -m http.server 8080 -d artifacts/pangu92_moe_weights_sync/plots`

然后浏览器打开:

`http://A2服务器IP:8080/latency_summary.png`

## 11. A3 正式三 case 实验

A3 8 卡按 16 die 作为 16 rank 使用.

生成正式 artifacts:

`python3 run_pangu92_three_case_experiment.py prepare --batch-size 24 --world-size 16 --num-experts 256 --top-k 8 --quant-mode 2 --synthetic-fallback`

校验:

`python3 run_pangu92_three_case_experiment.py verify-artifacts`

跑 `pangu_chain`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name pangu_chain --op-path pangu --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

在未修改 vLLM-Ascend 环境跑 `vllm_base`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_base --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

在修改版 vLLM-Ascend 环境跑 `vllm_modified`:

`ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 python3 run_pangu92_three_case_experiment.py run-case --case-name vllm_modified --op-path vllm --determinism-repeat 3 --dump-output --warmup 20 --repeat 100 --output artifacts/pangu92_moe_weights_sync/results/timing.csv`

跨 case output 校验:

`python3 run_pangu92_three_case_experiment.py check-outputs --golden-case pangu_chain --case vllm_base --case vllm_modified --rtol 1e-2 --atol 1e-2`

`check-outputs` 会同时校验 output 里的 `artifact_hash` 和关键 config. 如果你重新执行过 `prepare`, 旧 output 会被判为 invalid. 这是故意的.

生成图:

`python3 run_pangu92_three_case_experiment.py plot --input artifacts/pangu92_moe_weights_sync/results/timing.csv`

## 12. 通过标准

A2 预验证通过标准:

1. `npu-smi info` 可见 8 张 A2
2. `torch.npu.is_available()` 为 `True`
3. `prepare` 和 `verify-artifacts` 通过
4. `pangu_chain` 的 `determinism_passed=True`, 且确定性校验使用 `rtol=1e-4`, `atol=1e-4`
5. `pangu_chain` 能写出 timing CSV 和 output artifact
6. `plot` 能生成 PNG

A2 上 `dispatch_gmm_combine_decode` 和 `vllm_base` 属于可选验证. 只有当前环境安装的是包含 `DispatchGmmCombineDecode` ACLNN op 的 A3 custom op 包时, 才要求 `vllm_base` 确定性通过和 `check-outputs` 通过.

A3 正式结论通过标准:

1. 3 个 case 都使用同一份 `artifacts/pangu92_moe_weights_sync`
2. 3 个 case 都通过确定性验证, 确定性 tolerance 为 `rtol=1e-4`, `atol=1e-4`
3. `vllm_base` 和 `vllm_modified` 都与 `pangu_chain` output allclose
4. timing 只统计 dispatch 到 combine 闭区间

任一条件失败, 结果 invalid.
