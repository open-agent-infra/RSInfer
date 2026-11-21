---
name: optimize-kernels-with-aka
description: 从 RSInfer 的真实推理负载选取算子，向 AKA 异步委托 kernel 研究，重连并导出已验收候选，在原服务调用点评估正确性、开销和端到端收益。适用于新建研究任务，也适用于接手已有 task ID 或接入一个已有候选。
---

# 用 AKA 优化 RSInfer 的服务路径

你对用户指定负载的正确性和性能负责。AKA 研究 agent 可以改实现、编译、profile、按冻结契约验收候选。你决定优化边界、怎样接回、哪些负载值得采用。有 task ID 时先检查现有任务；已有候选时可以直接从接入开始。

## 从真实调用点抽题

- 用用户关心的指标记录当前 RSInfer revision 和负载基线。在服务中确认走了哪个实现、每步调用几次、时间花在哪里。若消除上游重复计算或修正派发就能解决问题，直接改调用路径。
- 选一个有参考实现或已有正确性测试的调用点。把当前实际实现作为 baseline，把最有希望的可用实现作为 seed。将独立的参考、输入生成器、验证器和计时代码放在冻结契约中，候选源码单独存放。先用相同输入验证抽出的 baseline 能复现服务调用的语义和时延量级。
- 记录会改变结论的实参。例如投影输出切片的行步长、分页 KV 与逻辑索引、量化 scale 的布局、调用方提供的输出缓冲区。多 rank 算子还要涵盖各 rank 的输入分布、通信与结果归并。按实际命中频率选 case；当两个阶段的接口或调优取舍不同，再拆成两题。保留逐 case 结果，便于分形状派发。

## 异步提交与接手

从 RSInfer 仓库根目录安装本技能固定的 SDK，使用已配好算子测试依赖的开发环境。

```sh
uv pip install --python .venv/bin/python -r .claude/skills/optimize-kernels-with-aka/requirements.txt
```

需要在本机阅读 SDK 源码时，从仓库根目录执行 `git submodule update --init -- third_party/aka`。升级 SDK 时，在同一 PR 更新 `third_party/aka` 指针、`requirements.txt` 和下方接口文档链接，使三处指向同一提交。

`local` 在执行主机上运行契约，GPU 题需要在有相应设备和工具链的主机运行。配置当前可用的研究 agent。自建命令契约时，在 `contract/contract.json` 的 `config` 里写明 `command`、`instructions` 和 `delivery`。验证器从 `candidate/` 加载实现，完成正确性和计时后打印一行 `AKA_TASK_RESULT=` 加 JSON，包含 `verdict`、`complete` 和带名称、数值、单位的 `metrics`。详见[固定版本的任务接口](https://github.com/alibaba/atrex-kernel-agent/blob/e6e2926d5251c7b815597af165d6396c42e4001c/docs/task-execution.md)。

```python
from pathlib import Path
import json
import aka

# Run inside the extracted experiment, alongside contract/, baseline/ and seed/.
root = Path("aka-tasks").resolve()
job = aka.submit(
    contract="contract", baseline="baseline", seed="seed",
    execution={"backend": "local"}, agent={"runtime": "codex"},
    objective={"metrics": [{"name": "latency_us", "unit": "us", "direction": "minimize"}]},
    key="prefill-projection-v1", root=root,
)
Path("aka-job.json").write_text(json.dumps({"root": str(root), "task_id": job.id}))
```

让 metric 名与 harness 输出一致。任务输入相同可复用 key；真实布局、基线或验收规则改变时，冻结新题并使用新 key。需要研究某个瓶颈时把现象和 profile 目标写入 notes。研究侧用任务绑定的 `aka execution` 执行实验。

在另一进程读取 `aka-job.json`，用 `aka.job(record["task_id"], root=record["root"])` 重连。通过 `job.events()` 和 `job.results()` 查看进展，按目标 case 选择 `results["candidates"]` 中已验收记录的 `id`，用 `job.export(candidate_id, destination)` 导出。一起保存 RSInfer revision、契约 ID、候选 ID 和接入改动。已发布候选可以在研究继续时试接。

## 在服务里兑现收益

在原调用点接入候选，验证它与宿主及其他扩展能在同一进程加载，且目标请求确实走到该实现。按仓库的 [RSInfer skill](../rsinfer/SKILL.md) 热改并检查实际 worker；用一次性探针记录形状、阶段和派发，计时窗口移除探针。

发现接口不合适时可以直接改绑定、kernel 或调用方。例如让 TopK 写进已有的带步长输出视图，让 attention 直接读分页 KV。把必要的重排、拷贝、分配、同步和额外 launch 计入调用点对比。扩展单独运行通过、同进程加载失败时，先区分 Python 模块名、原生动态链接符号和绑定类型注册，再修实际发生冲突的一层。对接入后修改过的源码重新运行原算子验证。

用相同输入、硬件、路由与计时边界比较原实现和接入版。分别给出独立 kernel、含适配的实际调用、用户负载三层结果。端到端收益小于预期时，先核对命中率、调用次数、拷贝与计算通信重叠。输出有差异时，先测基线自对照，再用真实激活或逐算子替换定位差异，并按既定容差判断。

给出采用范围及其证据，可按阶段或形状保留现有派发。用 `job.feedback(candidate_id, value)` 回传实际命中、收益或回退原因。若服务暴露题包遗漏的语义或成本，把可复现输入带回下一轮。
