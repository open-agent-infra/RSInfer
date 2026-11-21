# DS1017 and AgentX benchmarks on `exp/agentx-opt` (Wave 6A)

[English](README.md) | [简体中文](README_zh.md)

Run the scripts from the repository root on a host with four SM100 GPUs (DP4/EP4). This branch supports:

1. The official DS1017 prefill and decode sweeps against stock DeepInfer. Use the decode sweep for Figure 6(b).
2. AgentX dataset preparation and replay with the Wave 6A NVLink5 vocabulary-parallel greedy LM head and multi-stream decode overlap enabled by default for applicable workloads.

The benchmark stages measurement probes in a disposable source copy under `.runs/`. Successful default runs clean up the copy.

---

## 1. Set up the environment

Select four available GPUs with `--gpus 0,1,2,3` or `GPUS` and supply the DS1017 checkpoint via `--model` or `AGX_MODEL`. The scripts select the active Python environment and detect the CUDA compiler and `nvidia.cu13` headers. To override the Python interpreter, use `--python` or `AGX_PY`.

```bash
BENCH=tests/gpu/ds1017_bench
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 2. Official DS1017 prefill and decode sweeps

```bash
# Measure 64K prefill. Omit the budget to sweep 4096, 8192, 16384, 32768, and 65536.
"$BENCH/run_rsinfer_official.sh" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 65536

# Measure decode at D=256/GPU. Omit D to sweep 4, 8, 16, 32, 64, 128, and 256.
"$BENCH/run_rsinfer_official.sh" decode  --model "$AGX_MODEL" --gpus 0,1,2,3 256

# Stock DeepInfer baselines
"$BENCH/run_di_official.sh" prefill --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536
"$BENCH/run_di_official.sh" decode  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

---

## 3. Prepare and replay AgentX

```bash
# Prepare the 131K dataset. Add --max-context 262144 to prepare the 256K dataset.
"$BENCH/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k

# Replay at C=32. The default sweep is 16, 24, 48. For the 256K dataset, add --max-context 262144.
"$BENCH/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32

# Stock DeepInfer replay
"$BENCH/run_agentx.sh" run-di  --model "$AGX_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

Results go to `.runs/results/` by default. Use `--out <DIR>` to select another directory.
