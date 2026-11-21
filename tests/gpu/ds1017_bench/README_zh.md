# DS1017 Benchmark 与 AgentX 数据集脚本使用说明（`exp/agentx-opt` / Wave 6A）

[English](README.md) | [简体中文](README_zh.md)

本目录包含可直接登录 GPU 物理机一键运行的评测脚本（4×SM100，DP4/EP4），覆盖：

1. **DS1017 官方 Prefill / Decode Benchmark**（`agxinfer` 与原版 `DeepInfer` 对照）；图 6(b) 使用此分支的 Decode sweep。
2. **AgentX 数据集生成与回放 Benchmark**（默认启用 Wave 6A 的 NVLink5 词表并行贪心 LM Head 与多流 decode overlap 优化）

> 所有临时测量探针都会自动拷贝到 `.runs/` 下的隔离副本里执行并在结束后清理，**不会修改你的 git 工作树**。

---

## 0. 环境准备

- 4 张空闲 SM100 GPU（`--gpus 0,1,2,3` 或 `GPUS`）；DS1017 权重目录（`--model` 或 `AGX_MODEL`）；Python 默认使用当前激活环境（也可 `--python` / `AGX_PY` 指定），自动补齐 `nvidia.cu13` 头文件与 `nvcc`。

```bash
BENCH=tests/gpu/ds1017_bench
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 1. DS1017 官方 Prefill & Decode Benchmark

```bash
# Prefill（可传单点如 65536，或不传跑全档 4096 8192 16384 32768 65536）
"$BENCH/run_rsinfer_official.sh" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 65536

# Decode（可传单点如 256，或不传跑全档 4 8 16 32 64 128 256）
"$BENCH/run_rsinfer_official.sh" decode  --model "$AGX_MODEL" --gpus 0,1,2,3 256

# 原版 DeepInfer 对照
"$BENCH/run_di_official.sh" prefill --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536
"$BENCH/run_di_official.sh" decode  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

---

## 2. AgentX 数据集生成与回放 Benchmark

```bash
# 生成 131K（默认）或 256K（加 --max-context 262144）数据集
"$BENCH/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k

# 回放（默认 16 24 48；也可传单点如 32；256K 集加 --max-context 262144）
"$BENCH/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32

# 原版 DeepInfer 对照
"$BENCH/run_agentx.sh" run-di  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

结果默认保存在 `.runs/results/`（可用 `--out <DIR>` 指定）。
