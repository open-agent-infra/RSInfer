# RSI-native 推理引擎

[English](README.md) | [简体中文](README_zh.md)

一个为“AI 自我改进”而设计的大模型推理引擎。此分支（`optimized-deepinfer`）包含AKA 优化的 DS1017 Prefill 算子。复现步骤见[四卡评测指南](tests/gpu/ds1017_bench/README_zh.md)。

**算子优化开发工具**　agxInfer 使用 Atrex Kernel Agent（AKA）委托算子优化。在 [third_party](third_party) 中点击 `aka` 可查看当前版本使用的确切提交，后续开发在上游 [aka-for-rsi 分支](https://github.com/alibaba/atrex-kernel-agent/tree/aka-for-rsi) 进行。安装、提交任务和接回候选见 [AKA 使用指南](.claude/skills/optimize-kernels-with-aka/SKILL.md)。

## 动机

越来越多的工作在让 AI agent 去改进 AI 基础设施本身：写更快的 kernel、调调度策略、改整个推理引擎的代码。但现在的推理引擎（vLLM、SGLang 等）不是为这件事设计的：

- **改一点就要重启。** 换一个调度策略、一个 kernel，往往要重启整个引擎，几十秒到几分钟，agent 的每一轮尝试都被拖慢。
- **改了什么、影响了什么，没人说得清。** 换了 kernel 却没重新录 CUDA graph、换了权重却没重算缩放系数，这类错误会悄悄产生错误结果。
- **结果不可信。** 在反复修改过的进程里测出来的数字，依赖修改的历史；agent 还可能有意无意地“绕过”评测（比如去掉结束符、只算一部分词表）。
- **改进不能叠加。** 一次改进很难成为下一次改进的可靠起点。

## 使命

做一个**架构上就支持“安全地改任何东西”的推理引擎**：
- 任何改动都能安全地应用和回滚，并且如实报告实际做了什么；
- 任何结果都能追溯到确切的系统版本；
- 评测在被改的系统之外、在全新启动的实例上进行；
- 被接受的改进可以成为新的起点，可以分支、合并、继续叠加。

它首先是一个像 vLLM 一样可以部署的推理引擎；支持自我改进的部分是可以拆掉的插件。

## 核心思路

- **每个模块都是一个“单元”**，单元是管理和重载的最小单位，进程只是装单元的容器。改了一个调度策略，只重载这一个单元，不重启任何进程。
- **框架本身几乎不带领域概念**：没有内建的“调度器”“KV cache”“TP”，这些全是单元。框架只负责把单元拼起来、换下来，像一组微服务拼成一个引擎。
- **持有显存的“状态单元”比逻辑单元活得长**，所以改代码不会丢掉权重和 KV。
- **改动的代价由框架推导**：从“改了什么”算出“哪些单元要重载、哪些 CUDA graph 要重录、要不要排空请求”，只动真正受影响的部分。

![单元全景图](doc/figures/modules-map.png)

上图是引擎的单元全景图：彩色大框是进程，小框是单元；白色是逻辑单元，青色是状态单元，灰色粗框是框架本体。可交互的版本（点单元看职责、看场景演示）见 [doc/modules.html](doc/modules.html)。

## 目标

| 版本 | 内容 |
|---|---|
| **v0** | 在单台 8×SM100 上跑 Qwen3.8-27B。**不追求性能，追求架构正确、有表达力**：所有变更路径都能端到端走通、能被测试验证 |
| v0.1 | 支持 DeepSeek-V4.1-Flash（单机 DP8 + EP8，含 EPLB）。由 agent 在 v0 上一步步演进出来，每一步都 apply 到正在运行的引擎上 |
| v0.2 | 扩到两台机器 16 卡（DP16 + EP16），同样由演进得到 |

v0.1、v0.2 本身就是对 v0 架构的检验：如果演进的某一步不得不改框架本体，就说明设计还不够通用。

## 文档

- [doc/modules.html](doc/modules.html)：**设计的真源**。单元全景图、单元清单、单元模型与重载、边与组合规则、两个思想实验、已定的决定和待讨论的问题。
- [doc/roadmap.md](doc/roadmap.md)：做的顺序和每一步的验收。
- [doc/specs/](doc/specs/)：框架本体七个组件的规格。
- [doc/v0-plan.md](doc/v0-plan.md)：v0 计划（依据 modules.html 更新）。
- [doc/discussion-summary.md](doc/discussion-summary.md)：讨论纪要和调研记录（背景资料）。
- [doc/archive/](doc/archive/)：旧版本文档。

三者冲突时，以 modules.html 为准，其次是 v0-plan.md，再次是 discussion-summary.md。

## 设计历史

最初的阶段 0（纸面规格）和阶段 1（可验证的骨架）见 [doc/roadmap.md](doc/roadmap.md)。
