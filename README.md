# RSInfer

[English](README.md) | [简体中文](README_zh.md)

RSInfer is an LLM serving engine that developers and AI agents can update while it serves requests. The `optimized-deepinfer` branch contains optimized DS1017 prefill kernels from AKA. Start with the [four-GPU benchmark guide](tests/gpu/ds1017_bench/README.md).

RSInfer can delegate operator research to the Atrex Kernel Agent (AKA). The [submodule](third_party/aka) pins the SDK revision. Development continues on [aka-for-rsi](https://github.com/alibaba/atrex-kernel-agent/tree/aka-for-rsi). The [AKA skill](.claude/skills/optimize-kernels-with-aka/SKILL.md) (Chinese) covers SDK installation, task submission, and candidate integration.

## Motivation

Optimizing a serving engine involves more than writing a faster kernel. Four problems recur when an agent changes a running system:

- A small scheduler or kernel edit can require a full restart, adding seconds or minutes to each experiment.
- A change can invalidate a captured CUDA graph or weight scale and silently alter the output.
- Measurements in a repeatedly modified process depend on its history. Changing EOS handling or evaluating only part of the vocabulary can also distort a benchmark.
- A successful experiment needs a reproducible starting point before another improvement can build on it.

## Goals

RSInfer derives the scope of each update, supports rollback, and records what the engine actually changed. Results identify the exact system version. Candidates are evaluated on fresh instances before promotion so that accepted versions can become new baselines. The serving engine can be deployed independently of the tools for self-improvement.

## How updates work

- A **unit** is the smallest component the engine manages and reloads. A process hosts units. Editing a scheduling policy can reload just that policy.
- The framework composes units. The scheduler, KV cache, and tensor parallelism are implemented in units.
- State units own resources such as GPU weights and KV buffers and can outlive the logic that uses them.
- The framework derives which units need reloading, which graphs need recapture, and whether requests must drain.

![Processes, units, and framework components](doc/figures/modules-map.png)

In the diagram, colored containers are processes. White boxes are logic units, cyan boxes are state units, and thick gray boxes denote the framework. The [interactive diagram](doc/modules.html) (Chinese) explains each unit and illustrates update scenarios.

## Original design milestones

| Version | Target |
|---|---|
| v0 | Run Qwen3.8-27B on one eight-SM100 host and verify the update paths end to end. |
| v0.1 | Extend v0 to DeepSeek-V4.1-Flash with DP8, EP8, and expert load balancing on one host. Apply each development step to the running engine. |
| v0.2 | Extend the system to 16 GPUs on two hosts with DP16 and EP16. |

The later milestones test whether new models and topologies fit the existing framework.

## Design documents

The following design records are in Chinese:

- [Interactive unit map](doc/modules.html) — authoritative design, unit inventory, composition and reload rules, worked scenarios, decisions, and open questions.
- [Roadmap](doc/roadmap.md) — implementation order and acceptance criteria, starting with the specifications and verifiable skeleton.
- [Specifications](doc/specs/) — the seven framework components.
- [v0 plan](doc/v0-plan.md) — the plan derived from the unit map.
- [Discussion notes](doc/discussion-summary.md) — background research and design discussions.
- [Archive](doc/archive/) — earlier versions.

Where design records disagree, the unit map takes precedence over the v0 plan, followed by the discussion notes.
