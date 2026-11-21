# RSInfer · `exp/agentx-opt`

[English](README.md) | [简体中文](README_zh.md)

此分支用于复现技术报告图 6(b) 的 DS1017 Decode 曲线，以及 AgentX 多轮请求回放。Wave 6A 在 CUDA Graph 中重叠执行独立的 decode 工作，并为贪心生成提供 NVLink 词表并行 LM head。

## 运行 Decode 和 AgentX 评测

在四张空闲 B200 上，从仓库根目录运行。第一条评测命令遍历每卡并发 4、8、16、32、64、128、256。后两条命令从 Weka traces 生成 AgentX 数据，再以并发 32 回放。

```bash
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
bench=./tests/gpu/ds1017_bench

"$bench/run_rsinfer_official.sh" decode --model "$AGX_MODEL" --gpus 0,1,2,3
"$bench/run_agentx.sh" prepare --model "$AGX_MODEL" /path/to/cc-traces-weka-062126-256k.jsonl ./agentx-data-131k
"$bench/run_agentx.sh" run-agx --model "$AGX_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

完整的环境要求、256K 数据集选项和原版 DeepInfer 对照命令见[评测指南](tests/gpu/ds1017_bench/README_zh.md)。需要复现图 6(a) 的 AKA Prefill 曲线时，请切到 [`optimized-deepinfer`](https://github.com/open-agent-infra/RSInfer/blob/optimized-deepinfer/README_zh.md)。

## 相关文档

- [项目总览与主干快速上手](https://github.com/open-agent-infra/RSInfer/blob/master/README_zh.md)
- [技术报告（PDF）](https://github.com/open-agent-infra/RSInfer/blob/master/doc/RSInfer_Technical_Report.pdf)
