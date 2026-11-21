# DS1017 and AgentX benchmarks on `optimized-deepinfer`

[English](README.md) | [简体中文](README_zh.md)

Run these scripts from the repository root on a host with four SM100 GPUs (DP4/EP4). This branch supports two workloads:

1. The official DS1017 prefill and decode sweeps, with AKA-tuned prefill kernels and a stock DeepInfer baseline.
2. AgentX dataset preparation and replay on RSInfer and stock DeepInfer.

The scripts stage measurement probes (`admit_probe.py`, `dsbench_step_probe.py`, `decode_counts_probe.py`, and `top5-measurement.patch`) in a disposable copy under `.runs/`.

---

## 1. Install the AKA kernels

- Select four available GPUs with `--gpus 0,1,2,3` or `GPUS`. Set the checkpoint with `--model` or `AGX_MODEL`. The scripts use the active Python interpreter unless you specify `--python` or `AGX_PY`.
- Build the wheel from this checkout and install it in the serving workers' Python environment. The [kernel package README](../../../third_party/aka-ds1017-kernels/README.md) lists the dependencies. The [deployment record](../../../doc/rsi/ds1017-deepinfer-sm100.md) (Chinese) describes validation and rollback.

```bash
uv build --wheel third_party/aka-ds1017-kernels --out-dir dist
python -m pip install --no-deps dist/aka_ds1017_kernels-0.1.0-py3-none-any.whl
```

Both `run_rsinfer_official.sh` and `run_agentx.sh` accept `--aka-profile auto|official|validated|expanded`. The default, `auto`, selects `validated` when `aka_ds1017` is installed and `official` otherwise. For Figure 6(a), use `expanded` for 4K, 8K, and 16K and `validated` for 32K and 64K. Use `--aka-disable` to keep selected operators on the official path, `--aka-cache <DIR>` to set the build cache, and `--aka-warmup` to prebuild extensions.

```bash
BENCH=tests/gpu/ds1017_bench
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 2. Official DS1017 prefill and decode sweeps

Each request has 131,072 tokens: 126,976 cached prefix tokens and 4,096 uncached suffix tokens. The benchmark uses YoCo with 140 retained tokens, sampler top-k 1,024, 32 redundant experts, and random expert routing across four GPUs (DP4/EP4).

```bash
# Measure 64K prefill and warm the extensions first. Omit the budget to sweep 4K–64K.
"$BENCH/run_rsinfer_official.sh" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated --aka-warmup 65536

# Measure decode at D=256/GPU. Omit D to sweep 4, 8, 16, 32, 64, 128, and 256.
"$BENCH/run_rsinfer_official.sh" decode  --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated 256

# Stock DeepInfer baselines
"$BENCH/run_di_official.sh" prefill --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536
"$BENCH/run_di_official.sh" decode  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

Results go to `.runs/results/` by default. Use `--out <DIR>` to select another directory.

---

## 3. Prepare and replay AgentX

```bash
# Prepare the dataset.
"$BENCH/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k

# Repeat C=32 on one instance to distinguish the first 32K-shape DeepGEMM JIT from a warm run.
"$BENCH/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated ./agentx-data-131k/agentx_eval 32 32

# Stock DeepInfer replay
"$BENCH/run_agentx.sh" run-di  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```
