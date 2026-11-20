# DS1017 Benchmark 与 AgentX 数据集脚本使用说明（`optimized-deepinfer`）

本目录包含可直接登录 GPU 物理机一键运行的评测脚本（4×SM100，DP4/EP4），覆盖：

1. **DS1017 官方 Prefill / Decode Benchmark**（`agxinfer` + AKA 特调算子，与原版 `DeepInfer` 对照）
2. **AgentX 数据集生成与回放 Benchmark**（`agxinfer` 与原版 `DeepInfer` 对照）

> 所有临时测量探针（`admit_probe.py`、`dsbench_step_probe.py`、`decode_counts_probe.py`、`top5-measurement.patch`）都会自动拷贝到 `.runs/` 下的隔离副本里执行并在结束后清理，**不会修改你的 git 工作树**。

---

## 0. 环境准备与 AKA 算子包安装

- 4 张空闲 SM100 GPU，通过 `--gpus 0,1,2,3`（或 `GPUS`）指定；DS1017 权重目录通过 `--model`（或 `AGX_MODEL`）指定；Python 默认使用当前激活环境（也可 `--python` / `AGX_PY` 指定）。
- 在 `optimized-deepinfer` 分支上，先将 `third_party/aka-ds1017-kernels` 安装到 worker 所在的 Python 环境即可自动启用特调算子（详见 `doc/rsi/ds1017-deepinfer-sm100.md`）：

```bash
uv build --wheel third_party/aka-ds1017-kernels --out-dir dist
python -m pip install --no-deps dist/aka_ds1017_kernels-0.1.0-py3-none-any.whl
```

`run_agx_official.sh` 与 `run_agentx.sh` 支持 `--aka-profile <auto|official|validated|expanded>`（默认 `auto`：检测到已安装 `aka_ds1017` 自动选 `validated`，否则回退 `official`）、`--aka-disable`、`--aka-cache <DIR>` 与 `--aka-warmup`。

```bash
BENCH=tests/gpu/ds1017_bench
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 1. DS1017 官方 Prefill & Decode Benchmark

固定工况：131,072 上下文（126,976 共享前缀 + 4,096 未缓存后缀）、`yoco_clip=140`、`sampler_force_topk=1024`、32 个冗余专家、随机路由、4×SM100（DP4/EP4）。

```bash
# Prefill 64K（首次可加 --aka-warmup 预编译扩展；不传档位则跑 4096 8192 16384 32768 65536）
"$BENCH/run_agx_official.sh" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated --aka-warmup 65536

# Decode D=256（不传档位则跑 4 8 16 32 64 128 256）
"$BENCH/run_agx_official.sh" decode  --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated 256

# 原版 DeepInfer 对照
"$BENCH/run_di_official.sh" prefill --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536
"$BENCH/run_di_official.sh" decode  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

结果默认保存在 `.runs/results/`（可用 `--out <DIR>` 指定）。

---

## 2. AgentX 数据集生成与回放 Benchmark

```bash
# 生成数据集
"$BENCH/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k

# 回放（可传 32 32 在同一实例连跑两轮，排除首次 32K shape 的 DeepGEMM 冷 JIT 编译）
"$BENCH/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated ./agentx-data-131k/agentx_eval 32 32

# 原版 DeepInfer 对照
"$BENCH/run_agentx.sh" run-di  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```
