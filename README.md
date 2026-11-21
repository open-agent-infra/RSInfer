# RSInfer · `exp/agentx-opt`

[English](README.md) | [简体中文](README_zh.md)

Use this branch for the DS1017 decode curve in Figure 6(b) of the technical report and for AgentX multi-turn replay. Its Wave 6A changes overlap independent decode work inside CUDA Graphs and provide an NVLink vocabulary-parallel LM head for greedy generation.

## Run decode and AgentX benchmarks

Run from the repository root on four available B200 GPUs. The decode command sweeps 4, 8, 16, 32, 64, 128, and 256 requests per GPU. The next two commands prepare the AgentX suite from the Weka traces and replay it at concurrency 32.

```bash
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
bench=./tests/gpu/ds1017_bench

"$bench/run_rsinfer_official.sh" decode --model "$AGX_MODEL" --gpus 0,1,2,3
"$bench/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k
"$bench/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

See the [benchmark guide](tests/gpu/ds1017_bench/README.md) for environment setup, the 256K dataset, and the stock DeepInfer baseline. For the AKA prefill curve in Figure 6(a), use [`optimized-deepinfer`](https://github.com/open-agent-infra/RSInfer/blob/optimized-deepinfer/README.md).

## Documentation

- [Project overview and mainline quickstart](https://github.com/open-agent-infra/RSInfer/blob/master/README.md)
- [Technical report (PDF)](https://github.com/open-agent-infra/RSInfer/blob/master/doc/RSInfer_Technical_Report.pdf)
