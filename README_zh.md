# RSInfer · `optimized-deepinfer`

[English](README.md) | [简体中文](README_zh.md)

此分支用于复现技术报告图 6(a) 的 DS1017 Prefill 曲线。它在四张 B200（DP4/EP4）上，按输入布局选择投影、MLA、Indexer、shifted mHC 和 MegaMoE 的优化实现。

## 运行 Prefill 评测

从仓库根目录运行。先按[评测指南](tests/gpu/ds1017_bench/README_zh.md)把 AKA 算子 wheel 安装到 worker 使用的 Python 环境，再将模型路径指向 DS1017 checkpoint。

```bash
export AGX_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
bench=./tests/gpu/ds1017_bench/run_rsinfer_official.sh

"$bench" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile expanded 4096 8192 16384
"$bench" prefill --model "$AGX_MODEL" --gpus 0,1,2,3 --aka-profile validated 32768 65536
```

每个请求命中 126,976 个缓存 token，再计算 4,096 个新 token。命令末尾的数字是每卡每步的 token 预算。64K 保留官方 MegaMoE；做同引擎对照时，用相同预算和 `--aka-profile official`。

## 相关文档

- [算子包、构建要求和派发策略](third_party/aka-ds1017-kernels/README.md)
- [四卡部署、回滚和验收记录](doc/rsi/ds1017-deepinfer-sm100.md)
- [通过 AKA 委托新的算子研究](.claude/skills/optimize-kernels-with-aka/SKILL.md)
- [项目总览与主干快速上手](https://github.com/open-agent-infra/RSInfer/blob/master/README_zh.md)
- [技术报告（PDF）](https://github.com/open-agent-infra/RSInfer/blob/master/doc/RSInfer_Technical_Report.pdf)
