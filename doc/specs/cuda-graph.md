# CUDA Graph 支持：实现方案

- 日期：2026-11-10
- 状态：已实现；按 PR #45 的三份 review 改过一轮，见下面「实现选择」和 [cuda-graph-review-fixes.md](../evaluations/cuda-graph-review-fixes.md)。之前的记录：[A/B/C 对抗验证](../evaluations/cuda-graph-abc.md)、[早期验证记录](../evaluations/cuda-graph.md)。
- 代码基线：`master@efce690`；开发分支：`gw/dev/cuda-graph`。该基线已记录 M2a 完成，下一步包含 M2b 的真实 kernel 与 M2c 的 graph。
- 目标：在现有 Qwen3.8 TP1 worker 上运行完整 decode forward 的 CUDA Graph，并在 `apply` 后正确失效、重录和释放。
- 实现原则：**minimal patch，fail fast**。先完成最小正确闭环，复用已有执行路径和变更协议；配置不兼容和资源不足尽早拒绝，执行失败不能伪装成正常 eager 回退。

### 实现选择（2026-11-10，三份 review 之后）

PR #45 的三份 review（Gemini、Claude Opus 5.5 在 SM100 上的完整模型、Claude Fable 5.2 在 4090D 上的 4 层模型）之后改成下面这样；测量见 [cuda-graph-review-fixes.md](../evaluations/cuda-graph-review-fixes.md)。**§3–§7 是动手之前的方案，和这里不一致的地方以这里为准**，不一致的条目列在本节末尾。

- **平台**：`cuda_graph`（`engine/platforms/cuda_graph/`），PyTorch 的 CUDA Graph，一个进程一条 capture stream。一个 recorder 录的图共用一个内存池（SGLang、vLLM 的做法），从最大的桶录起；recorder 的图全部关闭后，下一次录制用新的池。
- **padding**：照 SGLang（`PaddingPolicy.ZERO`）。填充的 lane 是持久 batch 的 padding 行（下标 0）、长度 1，读写的只有 KV 的 padding 页和 GDN 的 padding 槽。状态池和 holder 与不开 graph 时完全一样，不读 plan 的配置。backend 每个桶的固定 buffer 只有一维的 `req_pool_indices`、`seq_lens`、`out_cache_loc`（GDN 是 `mamba_cache_indices`），`req_to_token` 直接用持久 batch 的表。
- **kernel**：固定容量的 decode attention 在单独的单元 `kernels.torch_graph` 里，声明 `requires=[CAPTURE]`：目录只在 `select(capture=True)`（从 graph 回放的步）时选它，eager 的 decode 仍然用变长的参照实现。它不展开 GQA（q 按 K/V head 分组后直接 matmul）。代价仍然随 `batch.max_context_len` 增长，由 M2b 的 paged decode kernel 取代。backend 的 `resolve` 多返回一张 `graph` 表：decode 的各个 definition 在回放步里用哪个 solution，没有可捕获的就是空串。
- **哪些情况不录图**：`off`；平台设备能力里没有 `cuda_graph`；backend 的某个 decode kernel 没有可捕获的 solution；forward 经 `catalog.call` 用到的 kernel（norm、linear 等，在一次 eager forward 里用 `catalog.recording()` 记下）有不可捕获的；图留下的显存超过 `plan.graph_memory_bytes` 或录制时 OOM。这些情况下 `auto` 走 eager，原因写进 `core.plan` 日志的 `cuda_graph.reason`；`required` 启动失败。其余的录制错误（capture 报错、回放的 logits 和 eager 不一致）在两种模式下都让启动失败。
- **录制**：每个桶一次 eager forward（预热，同时是对照）、capture、一次回放，比较 logits；不比较状态。完整模型上的状态对照由 `tests/gpu` 负责。
- **显存**：`plan.graph_memory_bytes`（默认 1 GiB）是图可以留下的设备显存上限，用录制前后 `memory_reserved` 的差来量（先 `empty_cache()`），所有桶合计；它来自 `holder.mem_fraction` 留给激活的那部分，holder 不为它另外预留。关闭图以后 `empty_cache()`，把池还给设备。
- **桶**：`plan.graph_buckets` 为空时是不超过 `min(batch.max_rows, 8)` 的 2 的幂；更大的步走 eager。`required` 要求最大的桶等于 `batch.max_rows`，所以行数多于 8 时要显式列出。
- **依赖守卫**：所有桶共用一份记录（提供者实例、目录版本、每个提供者一个 binding generation），每步在 `PlanService.buckets()` 里检查一次；`required` 进了 `PlanService` 契约，executor 在写状态之前拒绝没有图的 decode 步。
- **变更**：没有 `rebuild_graphs` 这个 action。`core.plan` 用 `ctx.derived("decode_graphs", rebuild="", units=<登记过 kernel 的单元和目录单元>)` 登记；`core.deps` 对没有 rebuild action 的派生产物一律重载它的 owner（不看平台），在它依赖的单元之后加载（`_next_ready` 把 capture 单元排在后面）；撤回时反向重载，旧的 plan 在恢复后的 kernel 上重新录。新加一个向目录注册的单元也会重载 plan。
- **部署**：`deploy/qwen38.py` 默认 `--cuda-graph off`（`torch_eager` 平台）；`auto` / `required` 用 `cuda_graph` 平台。

与下文不一致的条目：§3.3「候选构建失败不降级」（显存不够时 `auto` 降级）；§4.2 末尾「capture 与 eager 不能各选一套 kernel」（attention decode 现在就是两套，差别在 kernel 的容差以内）；§5.2 每个 lane 独立 scratch（改为共用 padding 项）；§5.3 每个桶独立的私有池和 holder 预留（改为共用一个池，holder 不预留）；§6 的调试用布局比较（删了）；§7.2 的重建 action（改为重载）。

## 1. 设计依据与范围

[讨论纪要](../discussion-summary.md) §15、§16.1 要求 decode 默认启用 graph、prefill 先用 eager、v0 不开启 `torch.compile`。绑定方式和单元职责以 [modules.html](../modules.html) 为准，执行接缝沿用 [stage2-contract.md](stage2-contract.md) §8–§10。

**变更时机采用已实现的 [empty-apply.md](empty-apply.md)。** 除只改 live 配置外，变更必须先使引擎为空，再暂停 worker。该文明确取代旧文档中的边服务边准备、分片重录和状态迁移要求。本轮在启动或暂停期内录制，不实现后台 capture 服务线程。

首版范围：

- 单卡 TP1，Qwen3.8 的完整模型 forward：embedding、所有层及其 KV/GDN 更新、最终 norm 和完整词表 LM head。
- 每行一个 token 的普通 decode，按行数分桶；prefill、mixed 继续 eager。
- 采样、`from_previous` 输入组织、状态恢复和快照均在 graph 外，保持现有执行顺序。
- 验证一步 overlap、固定 buffer 更新、依赖失效、空引擎热更新和资源回收。

不在本轮增加：`torch.compile`、分段 graph、graph 存档、VMM/grow、跨 rank capture 编排、投机 verify、共享 graph 私有池。未来 prefix 功能接入时再加组合验收；当前已有的 `StateCopy` 仍需验证。

### 1.1 最小改动的边界

1. 保留 `executors.pytorch` 为唯一 executor，沿用 `PlanKey`、`Plan.run()`、`PlanService.buckets()/get()`；不新增一套 CUDA worker/executor。
2. graph 的缓存、依赖快照、构建状态机放在 `core.plan` 内部，不做共享的 graph 对象框架。
3. 复用现有 backend、kernel 目录、状态存储和 `ctx.derived` / `ctx.on_action`；只有实际需要跨单元交接的信息才补共享接口。
4. 框架改动仅限真实 CUDA 平台接入及经证实缺失的必要原语。每处说明现有接口为何不足；不夹带格式整理或无关重构。
5. 目录沿用 `engine/platforms/`、`engine/core/plan/`、`engine/kernels/`、`tests/units/`、`tests/gpu/`。确需新增子目录时归入对应现有目录，不改顶层结构。

## 2. 当前实现与阻塞点

下表描述代码基线中的事实，不是待实现功能的声明。

| 位置 | 已有能力 | 本轮需要处理 |
|---|---|---|
| [executor](../../engine/executors/pytorch/unit.py) | 先选 plan；decode 取最小就绪 bucket；`StaleGraph` 回退 eager；异步 `Step` | 检查 decode 每行一个 token 等输入条件；验证固定输出在 overlap 中的消费时序 |
| [core.plan](../../engine/core/plan/unit.py) | `EagerPlan`、backend 启动解析、执行配置日志 | graph 构建、发布、依赖检查、释放及重建 action |
| [平台](../../engine/platforms/base.py) | `Graph`、`GraphRecorder`、capture 权限保护 | 真实 CUDA recorder、可复用 capture stream、失败清理；`torch_eager.graph()` 当前明确不支持 |
| [attention backend](../../engine/layers/attn/unit.py)、[GDN backend](../../engine/layers/gdn/unit.py) | `prepare` / `forward` 分离 | 当前 `binding` 无固定 buffer，`prepare` 每步生成新 tensor；需增加按 bucket 固定的元数据 |
| [参照 kernel](../../engine/kernels/reference/unit.py) | 完整 eager 参照 | 所有 solution 都声明 `capturable=False`；不能直接改标志就认为支持 capture |
| [attention decode](../../engine/kernels/reference/ops.py) | 按请求调用 PyTorch SDPA | `int(seq_lens[seq_idx])` 及随长度变化的切片不适合作为可复用 graph 的执行路径 |
| [持久 batch](../../engine/worker/batch/unit.py) | 稳定请求表、padding 行、binding generation | GDN 槽位映射首次收到 assign 才分配，启动 capture 前必须准备好所需表 |
| [存储接缝](../../engine/core/contracts/storage.py) | `bind()`、`scratch()`、`clear()`、`copy()` | 当前 KV/GDN 各只有一个 padding/scratch item；多个 lane 并发写同一个 item 的安全性尚未成立 |
| [kernel 目录](../../engine/core/kernels/unit.py) | solution 选择、目录 `version`、eager 缓存刷新 | 公共接缝尚未完整交出已选 solution 的 capture 能力及提供者信息 |
| [单元 context](../../engine/unit/runtime/context.py) | `plan_scope`、`derived`、可撤销 effect、action | 真 graph 尚未接入这些机制；服务代理被撤销并不能自动阻止绕过 Python 的 GPU replay |

## 3. 执行路径

### 3.1 正常一步

```text
校验记录、行和状态边界
  → 选择当前配置下依赖有效的最小 bucket，或 eager
  → 应用 RowDelta、清新请求状态、StateCopy(before)
  → 组织 input_ids / positions / last_index
  → GraphPlan 复核依赖，backend.prepare 更新固定元数据
  → 拷入固定输入，replay 完整 forward
  → 对真实行采样，记录 last_token，StateCopy(after)，记录完成事件
```

`BatchPlan.bucket` 仍只是提示。选 graph 前确认 phase 为 decode、每行恰好一个 token、行数和上下文长度在已校验容量内。非法输入应在任何本步状态写入前拒绝；合法但没有合适 graph 的输入按 §3.3 处理。

`GraphPlan.run()` 返回真实行的 logits 视图。采样在同一 step stream 上消费后，下一步才可覆盖该输出。跨步保留 logits 的测试或诊断要在覆盖前排入独立复制；当前 `Step.logits` 的 eager 独立分配行为不能直接推广到 graph 输出。第一版不增加无条件每步 logits clone，而是明确并验证需要保留它的调用方。

### 3.2 分桶与配置

配置声明放在 `core.plan/unit.toml`：

| 配置 | 语义 |
|---|---|
| `plan.cuda_graph` | `auto`（默认）、`off`、`required`。`auto` 在支持且配置合法的 CUDA 路径启用；`off` 使用 eager；`required` 要求配置的 decode bucket 全部就绪 |
| `plan.graph_buckets` | 显式正整数列表；空列表按有效行数上限生成 2 的幂。不超过上限的非幂尾部可走 eager；需要覆盖尾部时显式配置该容量 |
| `plan.graph_memory_bytes` | 正整数，默认 1073741824；holder 为 graph 保留的字节数，也是构建峰值的上限 |

首版这些键使用 `startup` 绑定，在单元初始化阶段读取，通过现有排空、暂停和单元重载生效，不新增 live 开关或在线分桶策略。构建使用该配置快照；不能只在 `plan_scope` 内读取这些键，因为现有规划器可能据 `plan_build` 记录派发重建 action，而不是重载。用 `dry-run` 测试确认实际计划。校验重复值、非正值、超出 `batch.max_rows` 的值；`required` 还需覆盖配置允许的 decode 行数。暂不增加自动调参、按负载补录或 LRU 驱逐。

`auto` 在 CPU / `torch_eager` 或尚不支持 capture 的合法 backend 组合上走 eager，并在启动日志记录具体原因；在 `cuda_graph` 平台的可捕获组合上默认录 graph。显式指定的 backend 不可用、布局不兼容等配置错误，在任何模式下都失败，不能借 `auto` 偷换 backend。

`required` 在启动或变更中就检查平台、所有捕获路径和资源，不等第一条请求触发失败。它要求普通 decode 使用 graph，prefill/mixed 仍按定义走 eager。

### 3.3 失败分类

| 情况 | 处理 |
|---|---|
| 合法输入没有匹配的就绪 bucket | `auto` 走同配置 eager；`required` 在本步状态更新前拒绝 |
| bucket、backend、dtype、布局或设备组合不合法 | 启动/变更校验失败，报告冲突项 |
| graph 依赖失效 | 从就绪集合移除；`auto` 可走当前配置 eager；不能在步内重录 |
| capture、warm-up 或候选验证失败 | 清理未发布产物，启动失败或变更撤回；不把失败当作正常能力探测 |
| replay 已提交后报错 | 沿用 `StepFailure`，不再次执行本步；请求处理由现有 worker/scheduler 故障路径负责 |
| 无法证明撤回后资源有效 | 保持失败/恢复状态，不能报告撤回成功后继续使用旧 graph |

现有 executor 捕获 `StaleGraph` 后只重跑模型 forward，不重复前面的状态操作。必须保证该异常在 graph 的 metadata 写入和模型执行之前抛出。选择之后禁止配置/绑定切换；若仍发生此类变化，视为实现错误。`required` 模式需把拒绝判定提前到状态更新之前，不能照搬现有无条件 eager 回退。

## 4. 平台与 kernel 接入

### 4.1 CUDA 平台

增加 `engine/platforms/cuda_graph/`，复用 torch 平台已有的张量分配能力，保留 `torch_eager` 的 eager 定位。首版使用通用 PyTorch CUDA 路径，按实际 GPU 记录验收结果；架构专用 kernel 留待后续。

必要接入点为 `platforms.PLATFORMS`、`unit.runtime.host.make_platform`、`engine/framework.toml` 和部署拓扑/平台身份的生成。平台实现及其构造入口属于框架边界，首次接入需要重启引擎；之后 plan、backend、kernel 的开发继续用 `apply`。

平台负责：

- 管理一个可复用的非默认 capture stream；warm-up 与 capture 使用同一个 stream。step stream 与 capture stream 的交接显式同步。
- 提供 recorder、replay、销毁和失败状态。优先保留现有 `ctx.platform.graph()` 权限入口；若必须补 warm-up 进入方式或释放方法，只补这一处实际交接。
- 验证 `CUDAGraph.capture_begin/capture_end` 的受控封装，避免每个 bucket 隐式执行全局 `empty_cache()`；异常时结束/撤销捕获并恢复 stream 上下文。该路径尚需真实 GPU 探针，不能当作已实现原语。
- 不在单元卸载时清理进程级 cuBLAS workspace，不允许 plan 自己随重载反复创建 stream。

固定地址、禁止捕获期间 CPU/GPU 同步，以及同进程不得同时执行非捕获 CUDA 工作，是 [PyTorch 2.11 CUDA Graph 约束](https://docs.pytorch.org/docs/2.11/notes/cuda.html#constraints)。独立 stream 本身不能替代进程的暂停边界。

### 4.2 Kernel 能力校验

先使同一套真实 kernel 的 eager 路径与参照对齐，再录 graph。attention 优先评估 FlashInfer，GDN 优先评估 SGLang 使用的 FLA/causal-conv 路径；移植前核对本仓库既有 SGLang 基线 `e838360` 的对应实现，记录来源、库版本和许可证。此处是实现选型方向，不代表已经完成第三方库兼容验证。

必须检查完整 forward，而不只检查 attention：linear、embedding、norm、RoPE、activation、KV store、GDN conv/recurrent update 和 LM head 均在捕获范围。对本来可捕获的参照算子，只在专项测试成立后更新其能力声明；保留不支持 capture 的参照实现用于 eager。

kernel 目录只补交出已选实现的必要描述（solution id、`capturable`、依赖版本/提供者）；不让 plan 读取目录私有注册表，不重做选核框架。先保守使用目录 `version` 使旧选择失效，再通过派生产物登记实际用到的 kernel 单元，让变更计划知道该重建什么。

`auto` 选择的组合也必须先固定；不能让 capture 与之后的 eager 回退各选一套不同语义的 kernel。

## 5. 固定 buffer、scratch 与显存预算

### 5.1 固定元数据

每个 bucket 预先分配 input ids、positions、last index、真实行数所需元数据和 backend workspace。`prepare(batch, plan, key)` 在 capture 外更新内容，`forward` 使用这些地址。request/page/slot id 和序列长度必须由设备运行时读取，不能在 capture 时经 `.item()`、`int(tensor)` 或 Python 分支冻结。

backend 的 graph 元数据与 eager 元数据分开持有，交替执行不得替换 graph 绑定。新增 bucket 时不应替换已有 bucket 的 buffer；如果提供者使用粗粒度 generation，必须先分配全部 bucket，再统一取依赖快照，以免后录 bucket 使先录 bucket 刚建好就失效。

`worker.batch` 在启动 capture 前准备全部所需槽位表，禁止第一条真实请求才创建 graph 将引用的表。若需要一个跨单元初始化调用，只传模型实际使用的状态种类，由 batch 自己分配，避免 plan 导入 batch 内部对象。

### 5.2 Scratch 与 padding

warm-up 和候选 replay 会真实写 KV/GDN 状态，只能使用存储保留的 scratch item；不能为了省事借用当前空闲但属于调度账本容量的真实请求 item。

当前一个 padding item 不足以证明多个 lane 同时写入安全。实现时按下列顺序收敛：

1. 核查所选 kernel 是否支持明确的无效 lane 标记，且对无效 lane 不读写状态；用 GPU 测试证明。
2. 无此保证时，每个并发写入 lane 分配独立 scratch item，所有 bucket 可按执行顺序复用这一组。由存储维护逻辑 id 到物理位置的映射，不让 kernel 或 plan 猜负数 id 的含义。
3. scratch 不计入 `Capacity`，其内存计入 holder 预算；warm-up、验证前恢复 scratch 的初始内容。masked 路径也不能通过“所有 lane 都无效”的空计算绕过对真实读写分支的验证。

独立 GDN scratch 可能很贵；必须在选定 kernel 后量出需求，再决定 bucket 上限。若预算不足，显式拒绝配置并建议降低 bucket/状态容量，不用多个 lane 共享可写槽位来规避限制。

### 5.3 显存与输出生命周期

首版每个 bucket 使用独立 graph 私有池。权重、KV/GDN、输入和 backend workspace 仍由原所有者持有；它们不会因被 capture 引用而自动归 graph 池所有。

holder 当前只预算权重和状态池，需为 graph 私有池、metadata、workspace、scratch 和临时构建峰值预留空间。最小实现是在 holder 的预算计算中增加明确的 graph 预留量，不建设全局资源调度器。预留默认值和最大 bucket 由目标模型实测确定，不能在本文凭空保证可用。

显式容量和预算矛盾在分配前拒绝；第三方 workspace 无法精确预估的部分，逐 bucket 记录实际峰值并设置构建预算边界。OOM 不作为搜索合适 bucket 的算法。发生 OOM 时按候选失败处理，确认 CUDA 上下文和旧资源是否还能安全使用。

释放顺序为：停止提交、等待相关在途 work 和输出消费者、移出可选集合、释放 graph 与捕获输出、释放 metadata/workspace、最后允许状态/权重提供者卸载。保留输出引用会保留私有池，不能仅丢掉 graph 对象就宣称已释放；参见 [PyTorch graph memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#graph-memory-management)。

## 6. 依赖登记与 replay 守卫

每个 graph 在 `core.plan` 内记录下表，不新增公共 `GraphDep` 对象体系。

| 依赖 | 检查方式 | 内容变化是否需要重录 |
|---|---|---|
| 权重、KV/GDN、持久 batch、backend buffer | 当前提供者实例身份 + binding generation + 可用性 | 同地址同布局更新内容不需要 |
| 模型/层/kernel 代码、选定 solution | 单元实例及实际选择的版本；目录 version 作保守失效检查 | 需要 |
| bucket、影响前向的配置与运行时标志 | 构建时固定值和当前有效值 | 需要 |

`Binding.generation` 在同一提供者实例内单调递增，地址、shape、stride、dtype、buffer 集合变化和撤销均覆盖；同地址 refit 不能仅因权重内容版本变化就重录。但 refit 后请求状态的处理仍服从空引擎规则。

已有服务代理带提供者 `InstanceId` 和撤销检查。实现应复用这条身份链，通过有效服务读取最新 binding，不能只反复检查 capture 时保存的旧 Binding；也不能用 Python `id(proxy)` 代替单元实例身份。若用现有 `provision_of` 即可完成身份检查，不为此增加 runtime API。只对缺失的实际数据交接补最小接缝。

构建前后收集的依赖必须覆盖构造期和选核期已缓存的值；仅观察 capture 区间的配置读取不够。用 `ctx.plan_scope` 归因读取，用 `ctx.derived` 登记配置与代码单元，交给 `core.deps` 推导重建。未被服务依赖覆盖的 kernel 注册来源，需要由目录准确交出，不能凭 solution 名字推断单元 id。

新增 kernel 单元也要覆盖：当前规划器对 artifact 的代码依赖主要取已登记单元与重载/删除集合的交集，新增实现可能改变 `auto` 的选择却不在旧依赖集合中。实现时必须验证这种变更能安排重建；必要时在 `core.deps` 增加针对目录注册变化的保守重建规则。仅靠 replay 守卫退回 eager 能阻止跑旧 graph，但不能算完成热更新后的 graph 恢复。

测试/调试模式再比对真实 tensor 地址、shape、stride、dtype 和 buffer 集合，抓漏增 generation；生产每步不遍历张量、不计算内容 hash、不做全设备同步。旧提供者被撤销、绑定不可用或代数变化时，graph 在任何 GPU 提交前拒绝。

## 7. 启动、热更新与撤回

### 7.1 启动

依赖单元加载并完成权重/状态分配 → 校验配置和 kernel → 准备所有固定表、scratch 和预算 → 在平台 capture stream 上逐 bucket warm-up/capture/验证 → 统一复核依赖 → 发布 graph 集合 → worker 开始服务。

只有完成验证和依赖复核的 graph 才进入 `buckets("decode")`。`get()` 和 `run()` 不隐式构建 graph。启动中途失败要撤销已分配但未发布的资源；构造函数抛错前也必须能清理，不能等成功提供服务后才注册 effect。

### 7.2 变更阶段

**空引擎规则不等于 `prepare()` 已经安全。** 当前协议先 Prepare，再 Drain/Pause；action 的 prepare 可能在旧版本仍服务时执行。因此：

| 阶段 | graph 相关工作 |
|---|---|
| Prepare | 解析变更参数、静态校验和预算检查；不 capture，不修改 active metadata，不启动 GPU warm-up |
| Drain / Pause | 沿用前端门和调度器排空；worker `quiesce()` 收集结果并同步相关设备工作 |
| Execute | 依赖动作/重载产生最终绑定后，构建和验证 graph；暂存可撤回的发布状态 |
| Commit / Resume | 确认新配置与 graph 一致后恢复服务；释放不再用于撤回的资源 |
| Abort | 清理部分候选；恢复旧配置，必要时基于恢复后的依赖重录旧 graph；校验通过才恢复服务 |

仅重建 graph 的 action 可在原实例上执行；被重载的 plan 在暂停期新实例上构建。`prepare` 被框架延后执行时仍只做校验，不利用这项特殊行为偷偷 capture。

必须验证重建 action 在影响绑定的状态动作之后执行。`ctx.derived` 不自动解决全部动作顺序；若现有 `core.deps` 排序不足，先在该单元的规划逻辑补最小顺序约束。不要一开始改事务协议。

不要求新旧版本并行服务，也不要求常驻两套完整 graph。若为撤回暂留旧 graph，必须确认旧依赖仍有效且预算可承受；若已经释放，撤回后按旧源码/配置重新构建。无论哪条路径，禁止重用已撤销提供者上的旧 graph。

### 7.3 开发方式

遵循 [仓库 skill](../../.claude/skills/rsinfer/SKILL.md)：平台基础接入后起常驻引擎，后续修改执行 `check → apply --dry-run → apply → 请求验证`；有请求时使用 `--wait-empty`，只有明确需要中止请求才用 `--force`。比较热改结果与同代码冷启动结果。

热改跑到旧代码、漏重建、撤回失败或持续资源增长时，在 [doc/framework-bugs/](../framework-bugs/README.md) 写报告；只有确认无法继续热改时退回冷启动。该开发流程不替代提交前测试。

## 8. 实现拆分与合并条件

每一步保持可单独审查；前一步的契约和测试成立后再进入下一步，不为后续优化提前扩接口。

| 步骤 | 最小改动范围 | 完成条件 |
|---|---|---|
| A：平台原语 | CUDA 平台、构造/身份接入、平台测试 | 真 GPU capture/replay、stream 复用、异常清理；非 plan 单元不能 capture |
| B：可捕获 forward | 真实 kernel 包装、现有 backend、必要目录描述 | 同组合 GPU eager 对齐；每个捕获算子能力明确，显式不兼容配置启动失败 |
| C：固定输入与 graph | `core.plan`、batch 初始化、scratch/预算、必要 executor 校验 | 单 bucket 再多 bucket；长度/行变化与 padding 下输出、状态均对齐，正常步不 capture |
| D：失效与热更新 | plan 的依赖登记/action、必要的 `core.deps` 顺序补充 | 换 kernel/改 bucket 正确重录；旧绑定拒绝；refit 保持绑定可复用；失败撤回与冷启动一致 |
| E：部署验收 | 部署开关/日志、`tests/gpu/`、带日期评估记录 | SM100 完整模型 TP1、overlap、重复 apply 和资源测量通过，默认 auto 启用已验收组合 |

涉及 `core/contracts` 时只增加真实交接定义，并按仓库规则更新 `tests/units/contracts/test_contracts.py` 的 `REACHES`。新增平台子目录及测试位置在实现 PR 中同步 skill 的目录地图；本设计 PR 不提前登记不存在的目录。

## 9. 验证矩阵

| 用例 | 断言 | 层级 |
|---|---|---|
| 分桶、非法配置、非法 decode | 最小就绪 bucket；不合法输入在状态修改前失败 | CPU 单元 + GPU 集成 |
| 实际行数小于 bucket、行重排/复用、长度跨页 | logits、真实 KV、GDN conv/recurrent 状态正确；padding 不写真实 item | GPU |
| A bucket → B bucket → A bucket；graph/eager 交替 | 元数据不串用，保留的输出不被覆盖 | GPU |
| overlap 开/关、`from_previous`、before/after copy | 采样、状态和完成顺序一致，不提前覆盖输入/输出 | GPU |
| scratch warm-up / 候选验证 | 真实请求状态不变，多个 scratch lane 无写冲突 | GPU |
| kernel 重载、同 generation 的新提供者、绑定变化 | 旧 graph 在提交前拒绝；新版本输出与冷启动一致 | CPU 故障注入 + GPU |
| 同地址换 stride、替换 buffer 漏增代数 | 调试守卫能发现；正式替换入口递增代数 | 单元 + GPU |
| 原地 refit | 绑定不变、graph 可复用，输出确实反映新权重；与新权重 eager 对齐 | GPU |
| 第 N 个 bucket 构建失败、预算不足 | 不发布部分候选，不误报成功；撤回恢复正确，未泄漏资源 | 单元 + GPU |
| 有请求时 apply / wait-empty | 正常拒绝或等待；capture 只发生在排空暂停之后 | 真实引擎 |
| 多轮 apply 与卸载 | graph/输出/workspace 生命周期正确，显存不随更新持续增长 | GPU |

数值口径沿用 `stage2-contract.md` §1.3：fp32 的完整 logits 在事先固定的容差内，greedy 第一名和第二名差距足够大时 token 必须相同；近似并列时记录差异。bf16 报告首个分歧及误差，不擅自把“逐 token 恒等”设为已有保证。KV/GDN 状态按对应算子的参照容差比较。比较图与 eager 后还要保留 HF 对照，避免两条路径共有错误。

测试只用固定 prompt 反复 replay 不够，必须改变 token、长度、行映射、page/slot 映射。fake/CPU 只能验证选择和失效机制，不能替代真实 capture、时序、padding 写入与显存生命周期验收。GPU 测试不因缺工具自动 skip；未执行就明确记为未验证。

提交前执行 `uv run pyright`、`uv run pytest -n 8`。实现修改 `unit.runtime.host` 或仓库规定的其他关键框架组件时，另跑 formal 全流程。GPU 用例放 `tests/gpu/`，通过显式路径运行。性能记录包含首次 capture、重录暂停、host/device step 时间、TPOT、峰值 allocated/reserved 显存及重复更新后的趋势；不预设加速倍数，也不将探针耗时当成整引擎结果。

## 10. 后续优化与未验证范围

- 优化 kernel 接入时锁定 FlashInfer/FLA/causal-conv 版本，重新验证完整 forward、padding 和 scratch。
- 完整 27B 真权重、SM100、长上下文和默认大 bucket 的显存与性能尚需专门验收。
- 首版没有共享 graph 池，未设定加速倍数或 TPOT 承诺。

第一个可交付里程碑仍是：**TP1 Qwen3.8 decode graph 与同配置 eager 对齐，并通过一次 kernel 热更新完成失效、重录和恢复服务。** 优化和完整模型性能验收不由小模型专项测试代替。
