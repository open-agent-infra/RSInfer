# 在四张 SM100 上启用 DS1017 DeepInfer prefill 特调

算子包包含 Attention FP8 投影、融合 MLA、Indexer、shifted mHC 和 EP4 MegaMoE。每个 DeepInfer 实例根据自己的配置选择实现，沿用调用方的输出缓冲区、scale、stride 和通信组。默认使用官方实现，可以直接启动未安装算子包的环境。

分支从 `4e26636` 切出。原性能实验快照的 DeepInfer `ops.py` 和 `mega.py` 与该提交逐字节一致。公开 PR #67 的头提交是 `f0d6692`，这两个文件均与原实验快照有差异。

## 构建、安装和切换

当前支持 Linux x86-64、NVIDIA SM100、PyTorch 2.13.0+cu132、CUDA 13.2 及 DS1017 部署使用的 DeepInfer 依赖。将 wheel 安装在实际启动 worker 的 Python 环境中。引擎会隔离 worker 的导入路径，仅在启动终端设置临时 PYTHONPATH 会导致冷启动时找不到算子包。wheel 自带 CUDA 源码和绑定，首次启动需要 CUDA 开发工具链。完整构建要求见[算子包说明](../../third_party/aka-ds1017-kernels/README.md)。

```sh
uv build --wheel third_party/aka-ds1017-kernels --out-dir dist
python -m pip install --no-deps dist/aka_ds1017_kernels-0.1.0-py3-none-any.whl
python -m aka_ds1017 explain --profile expanded --budget 8192
python -m aka_ds1017 warmup --profile validated --budget 32768
```

| 启动参数 | 行为 |
|---|---|
| `--set deepinfer.aka_profile=official` | 默认值，所有算子走官方实现 |
| `--set deepinfer.aka_profile=validated` | 在每卡预算 32K、64K 选择适用的特调实现 |
| `--set deepinfer.aka_profile=expanded` | 额外启用 4K、8K、16K 的特调路径 |
| `--set deepinfer.aka_disable=mhc,megamoe` | 两个算子单独回到官方实现，其余按所选 profile 运行 |

将 `AKA_DS1017_CACHE` 设为 worker 可写的私有缓存目录。`warmup` 构建扩展，DeepGEMM 在首次遇到实际形状时完成对应的 JIT 编译。`explain` 可离线查看选择规则。在 worker 设置 `AKA_DS1017_TRACE=1` 可以从日志核对具体调用命中了哪个实现。

整组应用配置变更，等待旧计划排空并重录 CUDA Graph。更新 wheel 时重启 worker。整组回滚时将 profile 改回 `official`。

## 算子选择规则

验收使用 DP4/EP4，请求总长 131,072 token，缓存 126,976，新输入 4,096，保留后缀 140。32K 每卡预算对应满步 32,768 行、紧凑阶段 8 × 140 = 1,120 行。64K 对应 65,536 行和 2,240 行。

| 算子 | 运行时判断 |
|---|---|
| FP8 投影 | 检查投影维度。小预算满步的 QB 可以启用特调，QKV 和输出投影分别按自己的规则派发。 |
| 融合 MLA | sparse prefill 和 paged 路径分别派发。4K、8K 满步仅在索引宽度为 640 时启用特调。 |
| Indexer | 复用调用方的索引缓冲区，区分 dense、source 和 user 模式。 |
| Shifted mHC | 按物理行数和 scale 布局派发。紧凑阶段 140、280、560 行走官方实现。 |
| MegaMoE | 32K 下，EP4 缓冲区选型与调用使用同一条件。所有 domain 的有序 token 计数、位置和 prompt 长度均相同才启用特调。空闲、混合或非均匀步，以及 64K 预算，走官方实现。 |

单卡可使用 `python third_party/aka-ds1017-kernels/tests/replay_mhc.py` 重放输入持续变化的 mHC CUDA Graph 回归。

## 复现端到端计时

在两种配置的独立实验快照中使用相同的逐步计数和 top-5 采集探针。仓库的[官方测量脚本](../../tests/gpu/ds1017_bench/run_rsinfer_official.sh)中包含这些探针的准备方式。服务设置为 DP4/EP4、实际路由（`deepinfer.force_random_gate=false`）、`yoco-clip=140`，两侧使用相同的图配置和采样设置。切换预算后先清空 radix cache，再开启缓存。

下例使用每卡 32K 预算。先将部署的 `sched.token_budget` 与 `BUDGET` 设为相同值，`RUN_DIR` 指向正在运行的引擎目录，`MODEL_DIR` 指向权重目录。每轮使用单独的结果文件。

```sh
BUDGET=32768
python tests/gpu/ds1017_bench/dsbench_pareto_sweep.py \
  --run "$RUN_DIR" --model "$MODEL_DIR" --mode prefill \
  --context-length 131072 --non-cached-tokens 4096 --yoco-clip 140 \
  --prefill-batch-size "$BUDGET" --seed-base 20000 \
  --prefill-measure-steps 32 --prefill-tail-steps 4 --step-stats \
  --result-jsonl "$RESULT_JSONL"
```

按调度器完成步的墙钟时间计算吞吐。逐步核实四个 rank 都达到指定预算，再用总 token 数除以两轮测量窗口的总时间，得到每卡 tokens/s。此指标同时覆盖框架与算子接入开销。

## 回归记录（2026-11-20）

- 验收环境使用 NVIDIA 驱动 595.71.05、NVCC 13.2.86 和 PyTorch 2.13.0+cu132。MegaMoE 的预热会通过 C++ 绑定实际调用 Python 显存布局接口，覆盖运行时模块查找。
- Linux Python 3.12 框架默认测试集 778 项通过，Linux 目标类型检查通过。算子包另有 4 项独立派发测试。
- 紧凑 mHC 在 1,120 和 2,240 行分别覆盖 attention、MoE 两种 scale 布局。部署采样和独立生成的输入各检查 512 次图回放，使用两个图、两个回放流，并在每一步改变输入。所有输出均与对应参考结果逐字节一致。

### 四卡端到端结果

同机测量。每档、每种实现各两轮，每轮 32 个满步。问答使用相同的源码输入与贪心采样，比较全部输出 token 和结束原因。答题列按源码参考答案判分。

| 每卡预算 | 官方 tokens/s/GPU | 特调 tokens/s/GPU | 加速比 | 官方 → 特调，ms/步 | 输出一致 | 答对，官方 → 特调 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 64,320 | 65,454 | 1.0176× | 63.68 → 62.58 | 8/8 | 8 → 8 |
| 8K | 71,782 | 72,845 | 1.0148× | 114.12 → 112.46 | 16/16 | 13 → 13 |
| 16K | 75,931 | 77,217 | 1.0169× | 215.77 → 212.18 | 32/32 | 25 → 25 |
| 32K | 77,350 | 79,685 | 1.0302× | 423.63 → 411.22 | 64/64 | 54 → 54 |
| 64K | 76,252 | 78,984 | 1.0358× | 859.46 → 829.74 | 128/128 | 101 → 101 |

共有 248 次配对请求，包含 128 个不同的完整输入。同一输入可在不同预算复用。在另一台四卡 SM100 上，使用同一 wheel 和从 Git 导出的源码，直接以 32K 的 validated 配置冷启动，未加计时探针。两组共 64 个请求的输出 token 和结束原因也全部与官方一致。

交付 wheel 的 SHA-256 为 `e6f798896f485a720b7b188689834445e25f7ff3069ada1a39e1b3150c42d278`。
