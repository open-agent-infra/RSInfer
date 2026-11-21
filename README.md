# RSInfer · `optimized-deepinfer`

[English](README.md) | [简体中文](README_zh.md)

Use this branch for the DS1017 prefill curve in Figure 6(a) of the technical report. On four B200 GPUs (DP4/EP4), the engine selects optimized projection, MLA, indexer, shifted mHC, and MegaMoE kernels according to each input layout.

## Run the prefill sweep

Run from the repository root. First [install the AKA kernel wheel](tests/gpu/ds1017_bench/README.md) in the workers’ Python environment, then set the path to the DS1017 checkpoint.

```bash
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
bench=./tests/gpu/ds1017_bench/run_rsinfer_official.sh

"$bench" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile expanded 4096 8192 16384
"$bench" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated 32768 65536
```

Each request reuses 126,976 cached tokens and adds 4,096 new tokens. The trailing command arguments are per-GPU step budgets. At 64K the policy keeps the official MegaMoE implementation. To measure the engine baseline, repeat the same budgets with `--aka-profile official`.

## Documentation

- [Kernel package, build requirements, and dispatch policy](third_party/aka-ds1017-kernels/README.md)
- [Four-GPU deployment, rollback, and validation (Chinese)](doc/rsi/ds1017-deepinfer-sm100.md)
- [Delegate a new kernel research task to AKA (Chinese)](.claude/skills/optimize-kernels-with-aka/SKILL.md)
- [Project overview and mainline quickstart](https://github.com/open-agent-infra/RSInfer/blob/master/README.md)
- [Technical report (PDF)](https://github.com/open-agent-infra/RSInfer/blob/master/doc/RSInfer_Technical_Report.pdf)
