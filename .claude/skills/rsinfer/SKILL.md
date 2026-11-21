---
name: rsinfer
description: Working in the RSInfer repository - where each component, unit, test and document lives (the directory map), and how to develop on a running engine by hot-applying changes (check, apply --dry-run, apply, --wait-empty, --force), what a normal refusal looks like, how to swap weights on a running engine (weights.path, model.dtype), what to do when a rollback fails and status shows stale units (introspect, restart), and how to report a bug of the update mechanism in doc/framework-bugs/. Use before changing code, tests or docs in this repository, or when starting, changing or debugging a running engine.
---

# RSInfer：目录地图与热改开发

规矩在 [AGENTS.md](../../../AGENTS.md)；这里是东西在哪，以及怎么在跑着的引擎上改东西。本文件的目录地图要和代码保持一致（AGENTS.md「文档是开发笔记」）。

## 目录地图

需要把真实算子交给 AKA 研究或接回已验收候选时，见[用 AKA 优化服务路径](../optimize-kernels-with-aka/SKILL.md)。

### 顶层

| 路径 | 内容 |
|---|---|
| `engine/` | 引擎源码树（框架本体 + 将来的单元） |
| `tests/` | 全部测试 |
| `doc/` | 设计文档、规格、评估记录 |
| `.github/workflows/ci.yml`、`.github/scripts/framework_changed.py` | CI：`uv sync --python 3.12 --extra cpu`、`uv run pyright`、`uv run pytest -n 8`。跑在组织自己的 Linux runner 上（`runs-on: ubuntu-latest`，2026-11-12 起；不用 GitHub 的 runner，那边按分钟算配额），`notify.yml` 也是。runner 是常驻的，uv 的缓存留在它的 `~/.cache/uv`。CI 里 `OMP_NUM_THREADS=1`：runner 的机器核多，不限的话每个引擎进程的 torch 都按核数开线程，两次 CI 同时跑会互相拖到单个测试超时；测试用的模型很小，一个线程跑得一样快。PR 每次 push 只跑一遍；push 只在 `master` 上触发。PR 和 push 没碰框架本体（`engine/framework.toml` 列出的模块）、`tests/framework/`、共用的测试设置、依赖和 `.github/` 时，跳过 `tests/framework/` |
| `.github/workflows/notify.yml`、`.github/scripts/notify.py` | 每个合进 `master` 的 PR 都由 LLM 写一句话摘要发到通知群；密钥 `PR_NOTIFY_WEBHOOK`、`PR_NOTIFY_SECRET`、`LLM_API_KEY`、`LLM_BASE_URL`，可选变量 `LLM_MODEL` |
| `pyproject.toml`、`uv.lock` | 依赖与 pytest / pyright 配置 |
| `third_party/` | `aka` 是固定版本的 AKA SDK 子模块；`aka-ds1017-kernels/` 是可选的 DS1017 特调算子包源码，支持 SM100 上的 prefill，接入与配置见 `doc/rsi/ds1017-deepinfer-sm100.md` |
| `.claude/skills/rsinfer/SKILL.md` | 本文件：目录地图与热改开发。Claude Code 按 skill 自动加载，其他 agent 照 AGENTS.md 的指引读它。用户同意为它新开 `.claude/` |
| `deploy/` | 部署工具，不属于 pytest，用户同意为它新开这个顶层目录。`agxinfer.py`：命令行，在一台机器上起引擎、看状态、停、睡和醒（`serve` / `status` / `stop` / `sleep` / `wake`），改跑着的引擎（`control`，即控制面客户端），发请求（`chat` / `complete`），生成测试用的 checkpoint（`checkpoint`）；`models/`：每个模型家族一个 recipe（现在只有 `qwen38.py`），按 `config.json` 选；说明见 `deploy/README.md`；在 pyright 的范围内 |

### `engine/`

不安装成包（`[tool.uv] package = false`），靠 `pythonpath = ["engine", "."]` 导入。导入名是 `core.*`、`unit.*`、`platforms.*`，**不是** `engine.*`。运行时进程从按身份物化的快照目录加载这棵树（`doc/specs/nodeagent.md` §3）。

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/framework.toml` | 哪些模块属于框架本体；改它们只能重启引擎，不能热重载 | `doc/specs/README.md` |
| `engine/core/common/` | 标识、单元清单解析、消息信封、`Ok`/`Failed` 结果、可调参数 | `doc/specs/README.md` |
| `engine/core/context/` | 配置项、运行时 flag、资源、capture 依赖记录 | `context.md` |
| `engine/core/ipc/` | RPC、pubsub、step stream（同机共享内存环 / 跨机 TCP）、open records、宿主替单元持有的数据通道 `proc.streams@1`（`channels.py`） | `ipc.md` |
| `engine/core/nodeagent/` | 每台机器一个的节点代理（`python -m core.nodeagent`） | `nodeagent.md` |
| `engine/core/supervisor/` | 进程管家、拓扑、管家与节点代理之间的消息 | `supervisor.md` |
| `engine/core/change/` | 变更事务：协调者、参与者、事件日志、blob 存储、队列；`build.py` 是单元构建步骤跑在里面的那个进程 | `change.md` |
| `engine/core/control/` | 控制面进程（`python -m core.control serve`）、控制接口的消息和客户端（`python -m core.control.client`） | `control-plane.md` |
| `engine/unit/runtime/` | 单元运行时：加载 / 重载 / 撤销、`ProcessHost`、注册表、静态 import 图 | `unit-runtime.md` |
| `engine/platforms/` | 平台接口 `base.py`；CI 用的 `fake/`；真张量、只做 eager 的 `torch_eager/`（CPU 或进程自己的 CUDA 卡）；`cuda_graph/` 复用 torch 分配并提供进程 capture stream、warm-up、capture/replay/close。合法平台名 `PLATFORMS`（`__init__.py`），由拓扑角色的 `platform` 决定 | `platform.md`、`cuda-graph.md` |

控制面单元（单元，不属于框架本体，不在 `framework.toml` 里）：

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/core/identity/` | 系统身份 `core.identity` | `control-plane.md` §2 |
| `engine/core/manifest/` | 清单与静态检查 `core.manifest` | `control-plane.md` §3 |
| `engine/core/deps/` | 影响分析（动作计划、dry-run）`core.deps` | `control-plane.md` §4 |
| `engine/core/api/` | 控制接口 `core.api` | `control-plane.md` §5 |
| `engine/core/sourcetree.py` | 读源码树（清单、模块）的库，`core.manifest` 和 `core.deps` 共用；不是单元 | |
| `engine/core/contracts/` | 单元目录之外的共享接口库，避免 `manifest.cross_unit_import`；只放实际交接的数据和必要调用，不 import torch，不放内部算法和整对象镜像；修改原则见 AGENTS.md「约定」 | `stage2-contract.md` §3 |

调度宿主里的调度单元（阶段 2 第 3 步，来源、行为和已知缺口见 `engine/core/sched/README.md`）：

| 路径 | 内容 |
|---|---|
| `engine/core/sched/` | 调度器 `core.sched`（`proc.loop@1`）：两个决策点、在途步队列（深度 1，开 `sched.overlap` 为 2）、多 rank、调度域（`sched.domains`：各 rank 跑各自的请求、一起走，DS 1017 的 DP 加 EP）、出错不冻结（`loop.py`）；变更前排空（`drain.py`，`doc/specs/empty-apply.md`） |
| `engine/policies/` | 决策点的策略单元：`fcfs`（`policy.order@1`）、`prefill_first`（`policy.compose@1`），接口在 `core/contracts/policy.py` |
| `engine/core/state/alloc/` | 联合账本 `core.state.alloc`（`ledger@1`，接口在 `core/contracts/ledger.py`）：默认是 `radix.py` 的前缀缓存（照 SGLang 的 radix cache 加 mamba `extra_buffer`，`doc/specs/prefix-cache.md`），`alloc.radix_cache = false` 时是 `pool.py` 不带缓存的账本 |

阶段 2 的文本前端：`engine/frontend/http/`、`tokenizer/`、`detok/` 是前端的三个单元（`frontend.http` 也是变更时引擎的门：变更期间新请求回 503，`api.py` 的 `Gate`），`frontend/interfaces.py`（它们之间的本地接口）和 `transport.py`（前端的两条记录流）是它们共用的库；`engine/sched/reqs/` 是调度宿主里的请求表状态单元 `sched.reqs`。来源、行为和运行方法见 `engine/frontend/README.md`。

worker 里的执行层单元（阶段 2，eager 和 TP1 decode graph，`stage2-contract.md` §6）：

| 路径 | 内容 |
|---|---|
| `engine/core/kernels/` | kernel 目录 `core.kernels`（`kernels.catalog@1`）：登记 definition 和 solution，按 shape 和平台能力选，没有合适的就用参照实现；`select(capture=True)` 给从 graph 回放的步选可捕获的 solution |
| `engine/kernels/reference/` | `kernels.reference`：Qwen3.8 要的全部 definition 及 PyTorch 参照（`ops.py`），从 SGLang 搬运，签名和状态排布照 SGLang；`paged.py` 是 paged attention 两个 definition 的参照，它们的 solution 是每步先 plan 再 run 的 operator；`spec.py` 是投机解码的验证 conv 和递归、提交、接受（doc/specs/mtp.md） |
| `engine/kernels/torch_graph/` | `kernels.torch_graph`：只给从 graph 回放的步用的 PyTorch solution（`requires=[CAPTURE]`），固定容量的 decode attention，代价随 `batch.max_context_len` 增长。有 FlashInfer 的 bf16 模型不会选到它；fp32 的 graph、没有 FlashInfer 的 GPU 和 CPU 上的回放测试还用 |
| `engine/kernels/triton/` | `kernels.triton`：SGLang 的 Triton kernel，登记成 `triton.*`，只在有 CUDA 设备的进程里登记：GDN 的 prefill conv、gating、chunk，decode 的"投影解包 + conv"和 packed decode，gated RMSNorm；attention 层的两个融合 kernel（Q/K norm + RoPE + gate 拆分、输出 gate 的 sigmoid·mul）；以及 paged attention 每步拼 KV 槽列表的 kernel（`attn.kv_indices@1`）。`ops.py` 把 definition 的签名接到 SGLang 的调法上；其余文件和 `fla/` 是搬来的 kernel，只去掉了对 `sglang.srt` 的依赖，第一行 `# type: ignore` |
| `engine/kernels/flashinfer/` | `kernels.flashinfer`：FlashInfer 的 paged prefill / decode wrapper（`flashinfer.attn.paged_*`），以及它的 Gemma RMSNorm（`flashinfer.norm.gemma_rms@1`，带 residual 时原地改）和 SiLU·mul（`flashinfer.act.silu_and_mul@1`），同样只在有 CUDA 设备时登记。attention 和 SiLU·mul 不收 fp32；attention 还要求 KV 的 dtype 和模型的相同。第一次 plan 某个 shape 时要编译，`PATH` 上要有 `ninja` 和 CUDA 编译器（`uv run` 下有） |
| `engine/kernels/cuda/` | `kernels.cuda`：自己用 C++/CUDA 写、单元加载时编译的 kernel，现在只有 `cuda.kv.store@1`（KV 写入合成一个 kernel，照 SGLang 的 `store_kvcache` 写的）。源码放在 Python 模块 `kv_store_cu.py` 的字符串里，这样改它能被 `core.deps` 看到；`ops.build` 用 PyTorch 的扩展加载器编译，产物按源码摘要命名，放在 `TORCH_EXTENSIONS_DIR`。没编译过的源码第一次加载要约一分钟，`PATH` 上要有 `ninja` 和 CUDA 编译器；只在有 CUDA 设备的进程里编译和登记。`ops.build` 也是它的构建步骤（`unit.toml` 的 `[unit] build`）：改了源码的变更在准备阶段另起进程编译，引擎照常服务 |
| `engine/core/state/weights/` | `core.state.weights`（`state.weights@1`）：参数存储、safetensors 读取、dummy 权重、`refit` |
| `engine/models/qwen38/` | `models.qwen38`（`model@1`）：SGLang `Qwen3_5ForCausalLM` 的搬运，参数名、`load_weights`、前向 |
| `engine/layers/` | `layers.linear`、`embedding`、`norm`、`rope`（cos/sin 表，和 attention 层在 attention 之前的那一个融合调用 `attn.qk_norm_rope_gate@1`）、`activation`（SiLU·mul 和输出 gate）、`sampler`；`layers.attn`、`layers.gdn` 是 attention / GDN 外壳加 backend（SGLang 的 `TorchNativeAttnBackend`、`GDNAttnBackend`；GDN 的 backend 直接接两个输入投影，自己拆）；`layers/attn/paged.py` 是 paged attention kernel 的 backend（SGLang `FlashInferAttnBackend` 的普通路径），`layers.attn` 按阶段在它和 `TorchNativeAttnBackend` 之间选；`solutions.py` 是 attention 和 GDN 的 backend 共用的库：按 `attn.*` / `gdn.*` 的 backend 配置在启动时固定每个阶段的 kernel |
| `engine/state/` | 存储单元 `state.kv_paged`（`MHATokenToKVPool`；一层声明什么张量就放什么，Qwen 是 `k`、`v`，DS 是压缩 latent `kv` 和 indexer 的 `ik`，按 DeepInfer 的格式存字节，一页存 `page_size / ratio` 条：一个张量每页的条目数可以少于每页的 token 数）、`state.gdn_slots`（`MambaPool`） |
| DS 1017（v1，2026-11-14，[doc/specs/v1-ds1017.md](../../../doc/specs/v1-ds1017.md)） | 另一族模型的 worker 单元，和 Qwen 的那些并列，一个 worker 只装其中一族（见下面「两族模型」）： |
| `engine/models/ds1017/`、`engine/models/ds1017_config.py` | `models.ds1017`（`model@1`）：DeepSeek-V4.1 文本模型，照 checkpoint 自带的参照实现 `inference/model.py`；参数按 checkpoint 的名字和字节声明（FP8、FP4 存成 `u8`），一层的路由专家叠成两个张量（每个专家的 `w1` 在 `w3` 上面，和 `w2`），专家和 FP8 权重的 scale 另按 GEMM kernel 读的排法存一份（`formats.pack_scales`），都由 `load_weights` 填、`refit` 会重写。`ds1017_config.py` 是库：config 字段、每层存什么状态（`states`）、Engram 的哈希布局 |
| `engine/layers/ds1017/` | `layers.ds1017`（`layers.ds1017@1`）：稀疏 attention（连同压缩器和 indexer）、Engram、MoE、Hyper-Connections，数据流照 DeepInfer：decode 步在设备上按固定形状推出一步的布局（能录进 graph），在状态的缓存字节上做 attention（`ds.cache_attn@1`），indexer 读页（`ds.index_topk_paged@1`）；其他步按计划里每行的几个数在设备上排（`Layout`，不从设备读回任何值：读回要等设备上排着的活，开重叠时等的是上一步），每层把环里的窗口 KV、这一步的 KV、各行的 latent 装进一个源（load segments）再做一次 sparse attention，indexer 在装载的 key 上按区间打分（`ds.index_topk@1`）。只在 prefill 或只在 decode 用的 definition 分开解析（`PREFILL_ONLY`、`DECODE_ONLY`），graph 只录 decode 的。kernel 后端由 `dsv4.backend` 定 |
| `engine/state/swa_slots/` | `state.swa_slots`（`state.swa.slots@1`）：每个请求一个槽，放每层的滑动窗口 KV、没凑满的压缩组、n-gram 往回看的几个 token |
| `engine/kernels/ds1017/`、`engine/kernels/ds1017_formats.py`、`engine/kernels/ds1017_compose.py` | `kernels.ds1017`：DS 的 definition（`ds.*`）和纯 PyTorch 参照。参照是 batch 无关的：求和顺序固定（`total`）、矩阵乘每行单独算（`matmul`），所以一个 token 的结果不随这一步怎么排而变。`ds1017_formats.py` 是库：FP8、FP4 的量化和反量化、RoPE 表；状态里三种记录（窗口 KV 528 B、压缩 latent 288 B、indexer key 68 B，DeepInfer 的缓存格式）的打包和解包也在 `ds1017_formats.py`；`ds1017_compose.py` 是库：缓存上的几个组合 definition（存、取、indexer 的选择、压缩器）写一次，参照和 TileLang 后端各给自己的算术（`Pieces`），所以 TileLang 后端仍是官方的算术 |
| `engine/state/expert_placement/`、`engine/policies/eplb/` | EPLB（2026-11-15，照 DeepInfer 的 `model_executor/eplb/`）。`state.expert_placement`（`expert.placement@1`）是机制：每层的专家放在哪个 rank 的哪个槽位（`eplb.extra_experts` 个多余槽位放热点专家的副本）、kernel 读的映射和计数、每 `eplb.interval` 步按最近 `eplb.window` 步的 token 数重排并在 rank 之间搬权重。`policies.eplb`（`policy.rebalance@1`）只做决定，是从 DeepInfer 搬来的两遍贪心；换重排算法就是换这个单元。都只在 DS 的 worker 里装。默认关（`eplb.interval = 0`），用 `agxinfer serve --set eplb.interval=1000 --set eplb.window=100 --set eplb.extra_experts=32` 打开。重排不改变算出来的结果 |
| `engine/kernels/deepinfer/mega.py` | DeepGEMM 的融合 kernel（2026-11-15）：`ds.hc_step@1`（两个子层之间的一步，`mega_mhc`）和 `ds.moe@1`（一层的 MoE，`bf16_mega_gate` 加带共享专家的 `fp8_fp4_mega_moe`）在后端 `deepinfer` 下的 solution，单 rank 和多 rank 都走它。它持有这个进程在各 rank 共享的 MoE 缓冲区（`SymmBuffer`）里的那一份。模型形状不合适时用 `deepinfer.unfused.*`（同一个组合，不融合的 kernel）。专家和共享专家的权重要按融合 kernel 的行顺序排（`formats.expert_layout`、`shared_layout`，模型按后端就地置换，权重存储里的 `experts.layout` 是标记） |
| `engine/kernels/ds1017_ep.py` | 库（2026-11-15）：专家和 Engram 表分在几个 rank 上时的两个 definition，`ds.ep_experts@1` 和 `ds.engram_fetch@1`，用某个后端自己的专家 kernel 加通信组的两个 collective（`gather`、`reduce`）写成，参照和 `tilelang` 后端共用。每个 rank 把每个 token 的每一项单独送回，token 所在的 rank 按专家的顺序加，所以结果和一个 rank 全算逐位相同 |
| `engine/kernels/tilelang/` | `kernels.tilelang`：参照实现自带的 TileLang kernel（`kernel.py`，原样）和它调用的 torch 写法，作为后端 `tilelang`。用它时引擎的 logits 和参照实现逐位相同。要 CUDA 和 `tilelang` 包，不能录进 graph |
| `engine/kernels/deepinfer/` | `kernels.deepinfer`：DeepInfer 用的 kernel 库（DeepGEMM、FlashMLA、TileKernels、DeepSelect）接成 `ds.*` 的 solution，后端 `deepinfer`，调用方式照 DeepInfer 自己的模块。要 SM 10.0 的卡和这四个库装在跑引擎的环境里（SM100 上 DeepInfer 的环境）。都能录进 graph；路由专家录进 graph 的是固定行数的版本。和官方参照实现算术上不同的三处写在 `ops.py` 的开头。`tilelang_ops/` 是从 DeepInfer 搬来的 TileLang 算子（Apache 2.0，`LICENSE.deepinfer`，原样，去掉了测试）：融合的 norm/RoPE/cast/存缓存、装载、压缩、候选块、`convert_kv_index`。TileLang 的 kernel 按宽度编译，层给它们固定的宽度（`batch.max_context_len`），不然每步重编 |
| `engine/core/state/holder/` | `core.state.holder`（`state.holder@1`）：按 SGLang 的分法定 KV 页数和 GDN 槽数（`mem_fraction_static`、`mamba_full_memory_ratio`、`max_mamba_cache_size`），启动时定，阶段 2 不在线扩容 |
| `engine/worker/batch/` | 持久 batch `worker.batch`（`HybridReqToTokenPool` 的表） |
| `engine/worker/comm/` | `worker.comm`（`comm.group@1`，2026-11-15）：这个 worker 在通信组里的位置。组里的 rank 每一步一起算（DS 1017 的专家和 Engram 表分在各 rank 上）。底下是 `torch.distributed`，照 DeepInfer 的建法：默认组走 gloo，按 `comm.rendezvous` 碰头；`comm.device_backend = nccl` 时另有一个卡上的组，DeepInfer 的 kernel 库要的就是它。`comm.size = 1`（默认）时什么都不建，所以每族模型的 worker 都装着它。一个进程只加入一次：改组的大小要换进程；一个 rank 死了，整个组要重启 |
| `engine/executors/pytorch/` | `executors.pytorch`（`executor@1`）：worker 唯一的 executor，SGLang `ModelRunner` 的流程，在 `torch_eager` 和 `cuda_graph` 上都用它；plan 要求必须有图（`PlanService.required`）时，没有图能装下的 decode 步在状态写入前被拒绝；`spec.py` 是投机解码的一步（草稿、验证、接受、提交、草稿扩展，SGLang 的 `EAGLEWorkerV2` 链式那条路，doc/specs/mtp.md），`spec.*` 配置键也由这个单元声明和检查 |
| `engine/core/plan/` | `core.plan`（`plan@1`）：eager（bucket 0）和 TP1 的 decode graph（`graph.py`：默认的桶、每个桶的固定输入、录制、依赖记录）。没配 `plan.graph_buckets` 时用 SGLang 的默认档位到 `min(batch.max_rows, 64)`；graph 里的 attention 是固定容量的那个（`kernels.torch_graph`）时仍是 2 的幂到 8。单元加载时录制（启动或暂停期），`plan.cuda_graph` 为 auto / off / required；没有 rebuild action，依赖变了由 `core.deps` 重载本单元。启动时让各 backend 按配置固定 kernel 并打一行 `core.plan: {json}`（含录了哪些桶、没录的原因、留下的显存） |
| `engine/core/worker/` | worker 循环 `core.worker`（`proc.loop@1`）：读计划、交给 `executor@1`、按计划顺序回 `Sampled` / `Resync` / `StepFailure`，启动时和结果换了读端时报 `Capacity` |

从 SGLang（`e838360`）搬运的代码在文件头注明出处，各目录带一份 `LICENSE.sglang`。改这些单元时先看 SGLang 的对应实现，能搬运的就搬运后适配，不自己另写。

每个单元目录下有一份 `unit.toml`。将来的单元（调度策略、KV cache、模型等）也放在 `engine/` 下，同样不列进 `framework.toml`。

在四卡 SM100 主机上测 DS1017 prefill 时，将 `RSINFER_MODEL` 设为权重目录。先按[DS1017 评测指南](../../../tests/gpu/ds1017_bench/README_zh.md)安装本分支的 AKA 算子包。从仓库根目录运行：

```bash
./tests/gpu/ds1017_bench/run_rsinfer_official.sh prefill --model "$RSINFER_MODEL" --aka-profile validated 32768 65536
```

结果默认写入 `.runs/results/`。

**两族模型**（2026-11-14）：`models.qwen38` 和 `models.ds1017` 都提供 `model@1`，一个 worker 只能装一族。
- `deploy/agxinfer.py` 按 `config.json` 选配方（`deploy/models/qwen38.py`、`ds1017.py`），在拓扑里把 worker 的单元列出来，去掉另一族独有的（配方的 `WORKER_ONLY`）；`control apply --source` / `check --source` 会按新工作区重算这份名单并随变更带上更新后的拓扑，新增或删除 worker 单元照常生效。绕过 `agxinfer.py` 直接用 `core.control.client` 或自己写拓扑时也要自己维护名单，否则两个模型单元都会被装进去。
- `core.plan` 和 `executors.pytorch` 不再写死 Qwen 的状态种类和层：backend 和存储从模型拿（`model.backends`、`model.stores`）。
- DS 的随机 checkpoint：`deploy/agxinfer.py checkpoint --config <含 config.json 的目录> --out <目录>`，格式和真 checkpoint 一样。

**多个 worker**（2026-11-15，[doc/specs/v1-ds1017.md](../../../doc/specs/v1-ds1017.md) 阶段 3）：`agxinfer serve --gpus 0,1,2,3` 每张卡起一个 worker，`--ranks N` 在一张卡（或 CPU）上起 N 个（本地试多 rank 用）。
- 每个 worker 是一个调度域（`sched.domains`），跑自己的请求。
- 模型在不在各 worker 之间分，由配方定：DS 1017 把每层的路由专家和 Engram 表按 rank 的顺序分开（`comm.size`），每一步所有 rank 一起算；Qwen 不分，每个 worker 是完整的一份。
- 一个变更里，同一个角色的进程是一起启动的（`core.change`）：通信组的 rank 在加载单元时要互相等。
- **改 EP 的卡数（worker 个数、`comm.size`）要整引擎重启**（`stop` 再 `serve`），不在线扩缩（2026-11-16 约定）。协调者先等要重启的旧 rank 起来，再起新增的 rank，而通信组要所有 rank 一起加入，两边会互相等到 `comm.timeout_s`。只改 `comm.size` 不加进程时，已加入的进程在 `worker.comm` 重载时直接报错，变更撤回。
- EPLB 搬过的专家在哪，记在权重存储里（`experts.placed`，2026-11-16）：重载模型或 `state.expert_placement`（改代码、改 `eplb.*` 而槽位数不变）都按它重建映射，只有权重重新装载（或 `refit`）才回到初始布局。关掉 EPLB 而槽位数不变时，kernel 仍按搬过的位置读，只是不再重排。
- 用参照或 `tilelang` 后端时，分开的专家经 `torch.distributed` 交换，录不进 graph，`plan.cuda_graph = auto` 会退回 eager 并在 `core.plan` 那一行里写明是哪个 kernel。
- 没有请求的 rank 每一步也要跑（`model.idle`）：别的 rank 的 token 可能选了它的专家。所以一个 rank 出错，其他 rank 会卡在那一层，直到 `comm.timeout_s`。
- 后端 `deepinfer` 在多 rank 下用 DeepGEMM 的融合 kernel（`kernels/deepinfer/mega.py`），能录进 graph：要显式给 `--cuda-graph auto`（默认 off），桶默认到 8 行，更大的 batch 给 `--graph-buckets 1 2 4 8 16 32 64`。各 rank 同时录、同时回放；融合 MoE kernel 在 kernel 里等别的 rank，等 60 秒是设备断言。所以一个 rank 的一步出错（比如 OOM），其他 rank 的进程 60 秒后退出，之后的 `apply` 是 `change.recovery_required`，只能重启引擎（2026-11-16 碰到过，[doc/rsi/2026-11-16-ds1017-prefill-overhead.md](../../../doc/rsi/2026-11-16-ds1017-prefill-overhead.md)）。
- `sched.token_budget = 32768`（DeepInfer benchmark 的 prefill 预算）配默认的 `--mem-fraction 0.85`，SM100 上显存贴着上限，跑了一百多分钟以后出过 OOM；`16384` 时每张卡至少空 11 GiB。对照 DeepInfer 的 benchmark 时还要 `deepinfer.force_random_gate = true`（它强制随机路由，文本没有意义，量完改回去）。
- 模型层的三个钩子（都是可选的方法，没有就不调）：`model.idle()`（executor 收到空计划时）、`model.stepped()`（每个 forward 之后、graph 之外：EPLB 计数）、backend 的 `fetch(input_ids, positions)`（`core.plan` 在 `prepare` 之后、forward 或回放之前调：多 rank 时取 Engram 的行）。
- 记录：[doc/rsi/2026-11-15-ds1017-several-gpus.md](../../../doc/rsi/2026-11-15-ds1017-several-gpus.md)；没做完的在 specs 的 10.8 节。

### `tests/`

- `tests/__init__.py`：让 `tests` 成为包。所有测试辅助代码都用 `tests.…` 绝对导入。
- `tests/conftest.py`：autouse 的 `_isolate_imports`，对所有测试生效：每个测试后恢复 `sys.path`、`sys.modules` 和各个包的 `__path__`（toy unit 被导入进全进程共享的 `sys.modules`；进程内的控制面会让包先去快照里找模块）。
- `tests/framework/`：框架本体（`engine/framework.toml` 列出的模块）的测试。
  - `conftest.py`：`pkg`、`snapshot` fixture。
  - `support/`：通用工具 `spec`、`manifest_toml`、`write_tree`、`PROC`、`events`、`GENERIC_UNIT`。
  - `unit_runtime/`、`context/`、`ipc/`、`platforms/`、`change/`、`cluster/`：按组件划分。`ipc/test_location_transparency.py` 扫描整个 `engine/`，禁止写死本机地址或用 PID 当地址（P7）。
  - `adversarial/`：对抗与故障边界测试，对应 `doc/evaluations/` 的记录。
  - `formal/`：TLA+ 有限模型（`models/`）、runner（`run.py`）、对真实实现的轨迹回放和契约测试。细节见 `tests/framework/formal/README.md`。
- `tests/units/`：单元的测试（不是框架本体的测试）。`units/core/`：控制面单元 `core.identity`、`core.manifest`、`core.deps`，在小源码树上逐条测规则，不起进程。`units/contracts/`：接口库 `core.contracts`（编码往返、单元能 import 它、每个模块只 import 到本接缝的模块）。
- `tests/units/scheduler/`：调度器、账本和前缀匹配算法的测试。`support.py` 把调度宿主的单元放进进程内的单元运行时，配上线程上的夹具 worker 和充当前端的客户端；`test_processes.py` 用真实控制面起调度宿主和两个夹具 worker 进程。`units/schedfix/` 是测试夹具单元（夹具 worker），真实进程测试把它拷进工作区。
- `tests/units/frontend/`：前端和请求表的测试。`support.py` 提供字节 tokenizer、脚本调度器（真请求表、固定回复，测试夹具）和只回答 Lookup 的假进程管家；`demo.py` 起一个无模型的 HTTP 服务。
- `tests/units/kernels/`：kernel 目录的选择（含回放步的选择）和重载，参照 kernel 对照 HF `qwen3_5`（要 `--extra cpu`），`kernels.torch_graph` 对照参照 kernel，paged attention 的参照对照原来的参照。
- `tests/units/worker/`：worker 循环。先对着假 executor 测循环本身，再把 worker 的全部单元加上循环放到线程上（`support.py` 的 `RealWorker`），接在 `tests/units/scheduler/support.py` 的真调度器下面，贪心生成和 HF 逐 token 相同。
- `tests/units/state/test_expert_placement.py`：EPLB 的机制和策略，一个进程里用线程当几个 rank（`Ranks`，collective 是一道栅栏），每个槽位的行里写着它该是哪个专家，搬错了能看出来。
- `tests/units/models/`、`state/`、`executors/`：模型和权重存储、存储单元和持久 batch、整个 worker 栈对照 HF（逐 token）和几组对比。`tests/units/support/worker.py` 提供：worker 全部单元装进一个 `UnitRuntime`（`Worker`）、小尺寸 HF 参照模型，以及按调度器的方式写每步记录的测试夹具 `Driver`。 `executors/test_paged_attention.py` 在 CPU 上把 attention 切到 paged 一族的参照 operator，对照原来那一族。
- DS 1017 的测试：`tests/units/kernels/test_ds1017.py`（数值格式和参照 kernel）、`tests/units/models/test_ds1017.py`（小配置过整个 worker 栈：分块、多行、graph 桶都和一次性算逐位相同）；`tests/units/support/ds1017.py` 提供小配置、随机 checkpoint 和 DS 的 worker（`DSWorker`），`tests/units/support/graphs.py` 是 CPU 上的 graph 替身。
- `tests/units/support/tokenizer.py`：测试用的真 HF tokenizer（词级、带 Qwen 的 chat 标记和模板）。
- `tests/ci/`：CI 自己的工具（`.github/scripts/`）的测试，比如判断一个 PR 有没有碰到框架本体。它既不是框架本体，也不是单元或整个引擎，所以单独一个目录。
- `tests/gpu/`：要 GPU、手动跑（由 `addopts` 排除）。`test_qwen38_partial.py`：Qwen3.8-27B 真实结构的前 4 层，worker 栈对照 HF；`test_qwen38_serve.py`：用 `deploy/agxinfer.py serve` 起整个引擎，经 HTTP 对照 HF（`QWEN38_MODEL` 指向完整模型就是 SM100 上的端到端验证）；`test_qwen38_overlap.py`：worker 栈重叠一步（下一步在上一步跑完之前提交）和逐步提交采到的 token 相同，步与步之间设备空得更少；`support.py`：生成部分层数的 checkpoint、HF 模型。24 GB 的卡就够。
  - `test_cuda_graph.py`：capture/replay、状态/padding、HF、refit、依赖失效、不可捕获或放不下时 auto 走 eager、状态池容量不变、候选失败清理、overlap、多次重录后显存（allocated、reserved、设备空闲）不变；`test_cuda_graph_apply.py`：真实控制面 kernel apply、失败配置撤回、恢复及冷启动对照，配置变更核对 plan 代数和实际 capture 次数，默认 `mem_fraction` 下连续 10 次 kernel 变更。`test_qwen38_partial.py` 同时覆盖 graph/eager；`tests/units/executors/test_cuda_graph.py` 不需要 GPU：桶、守卫，以及用替身 recorder 在 CPU 上跑回放步的整条路径（固定 buffer、padding、`kernels.torch_graph`）对照 eager。
  - `test_state_rebuild.py`：状态池原地重建（新池比旧池稍大）和撤回时旧权重、旧池重新装回（一次更大的尝试失败以后）都放得下，大小按设备空闲显存定（2026-11-10、2026-11-13 两份 framework-bugs 报告）。
  - `test_speculation.py`：投机解码（doc/specs/mtp.md）：真实结构 4 层、真 kernel、开重叠，fp32 下和不开投机逐 token 相同，bf16 下回放验证 graph 和 eager 相同；草稿来自预言（`tests/units/worker/support.py` 的 `Oracle`），checkpoint 的随机 MTP 权重几乎什么都不接受。
  - `test_ds1017.py`：DS 1017 对照 checkpoint 自带的参照实现（`ds1017_reference/`，原样带进来）：用后端 `tilelang` 时 logits 逐位相同（`cuda_graph=auto` 下也因不可捕获而退回 eager `tilelang`，不会录进别的后端）；参照 kernel 逐个在容差内；参照 kernel 下分块、多行、graph 回放逐位相同。要先在 GPU 环境里装 `tilelang==0.1.12`（不带依赖装，别换掉 torch：`uv pip install --no-deps tilelang==0.1.12 apache-tvm-ffi==0.1.11 torch-c-dlpack-ext==0.1.5 z3-solver==4.15.4.0 cloudpickle ml-dtypes psutil`）。小形状（1024 宽、16 头 × 128）在 4090 D 上能跑；真实形状的 attention kernel 要 SM 10.0。
  - `test_ds1017_deepinfer.py`：`kernels.deepinfer` 的每个 kernel 对参照（真实形状）、一个 token 的结果不随同一步的其他行变、录进 graph 换输入回放；缓存上的组合 definition 的用例在 `support_ds1017_cases.py`（`test_ds1017.py` 拿它们在小形状上对 TileLang 后端）。只能在 SM100 上 DeepInfer 的环境里跑。
  - `check_ds1017_real.py`：SM100 上手动跑的引擎级检查（真实形状的随机权重，或真 checkpoint 的前几层）。`--backends` 每个后端跑一遍（`+graph` 录 graph，`~direct` 把 MoE 换成 DeepInfer 的舍入方式）；`--shadow tilelang` 在每次 kernel 调用旁边用官方 kernel 在相同输入上再算一遍，逐个 definition 报相差多少。
  - `sm100_ds1017.sh`、`sm100_deepinfer_env.sh`：SM100 的任务入口（各步检查、建 DeepInfer 的环境）。
  - `check_ds1017_ranks.py`（2026-11-15）：同一个 checkpoint 在一个 rank 和几个 rank 上各起一次引擎，经 HTTP 问同样的提示词（一个一个问，再一起问），文本要逐字相同；`--no-single` 只起多 rank 的（整个模型一张卡放不下）。本地用 `--random <目录> --ranks 2`（两个 worker 共用一张卡，要把 `--mem-fraction` 调小），SM100 上的入口是 `sm100_ds1017_ranks.sh`。
  - `sm100_deepinfer_baseline.sh`（2026-11-15）：在节点上跑 DeepInfer 自己（EP4/DP4 的离线例子和 benchmark），拿它的文本和吞吐当对照。
  - `sm100_follow.sh`、`sm100_session.sh`：SM100 上的两个长任务，跟着一个分支，推一个提交就跑一次，不用每次排任务和冷启动。前者在 DeepInfer 的环境里跑该提交的 `sm100_follow_check.sh`（做 kernel 的对照用）；后者先起一个引擎，每个提交 `apply` 上去再跑 `sm100_session_check.sh`（热切换后端、量吞吐）。
  - `test_prefix_cache.py`：前缀缓存在 worker 里的部分（2026-11-13）：prefill chunk 中间和末尾存下的状态对照单独算到那里的、graph 回放里 decode 存下的对照 eager、恢复后再算两个 token 对照整个提示词，都带对照组；真 kernel，bf16。
  - `bench_cuda_graph.py`：手动 decode 吞吐 benchmark，在几档 `batch.max_context_len` 下对比 eager（`torch_eager`）和 graph，输出每次测量与环境 JSON；不作为 pytest 的性能阈值断言。
  - `bench_serving.py`：手动的 HTTP 压测，对任何 OpenAI 兼容的服务发同样的负载（本引擎或原生 SGLang），输出每档并发的吞吐、首 token 时间和每 token 间隔。
  - `test_real_kernels.py`：`kernels.triton`、`kernels.flashinfer` 的每个 solution 对照参照实现；`kernels.cuda` 的 KV 写入逐位对照参照实现（含回退路径）；worker 栈（4 层、bf16）上 `auto` 选真 kernel、graph 对 eager（同样行数的桶逐值比，带 padding 行的桶比 token）、graph 显存的计量冷启动和重载一致、被拒绝或只用来自检的 graph 不在 backend 里留东西、KV 和模型 dtype 不同时不选 FlashInfer、kernel 单元带请求重载、按名字点 backend。`conftest.py`：每个测试结束时记下它 import 的第三方模块，下一个测试开始时放回去。`tests/conftest.py` 的隔离会忘掉它们，而 FlashInfer 的原生依赖同一进程里不能 import 两次，CuTe DSL 的子模块重新 import 以后编译不了下一个 kernel。记录见 `doc/evaluations/real-kernels-4090d.md`、`real-kernels-fused.md` 和 `real-kernels-kv-store-indices.md`；SM100 上完整模型的结果在 `real-kernels-sm100.md`。
  - `test_cuda_graph_adversarial.py`：A/B/C 共用的 GPU 验收，检查数值、实际 CUDA replay，以及 catalog 变更后在写状态之前拒绝旧图；实验分支、结果与复现命令见 `doc/evaluations/cuda-graph-abc.md`。
- `tests/scenarios/`：整个引擎（框架本体 + 单元）的测试：modules.html 的 9 个场景、命令行 `serve`、覆盖检查和拆除检查（roadmap 阶段 1 的验收）。
  - `test_m2a.py`：阶段 2 的 M2a，在 CPU 上用 `deploy/agxinfer.py serve` 起整个引擎（前端、调度宿主、worker，小随机模型、真 HF tokenizer），经 HTTP 的文本和 HF 逐字相同；命令行在后台起引擎、`status`、`sleep` / `wake`、`stop`；`deploy.py` 是它和 `tests/gpu/` 共用的起停辅助。
  - `test_no_temporary_checks.py`：`engine/` 里不能留下标着 `# PROBE` 的临时检查（「在引擎里做检查」）。
  - `units/scn/`：场景测试用的单元（测试夹具，写法和真单元一样，但只做场景需要的那一步）；`harness.py` 把它们和本仓库的 `engine/` 一起拷进工作区，但不拷 `engine/` 里控制面以外的单元（前端、请求表、kernel、模型等）。场景拓扑的角色不列单元，那些单元会在每个场景进程里加载；kernel 单元每次启动还要 import torch。
- 测试辅助模块（不以 `test_` 开头，pytest 不收集）：

  | 模块 | 提供 |
  |---|---|
  | `framework/cluster/harness.py` | 测试事件循环里的真实控制面 + 子进程里的节点代理（`Cluster`，`async with cluster(...)`） |
  | `framework/change/engine.py` | 基于 cluster harness 的协调者测试台（`Engine`，`async with engine(...)`） |
  | `framework/change/harness.py` | 进程内 Coordinator + 两个真实 `ProcessHost`，可在传输边界丢请求 / 回执（`Harness`、`Capture`、`OLD`、`harness`） |
  | `framework/adversarial/peer.py` | 单个真实 `ProcessHost`，逐条投递参与者消息（`Peer`、`peer`） |
  | `framework/ipc/peers.py` | IPC 测试在子进程中运行的代码 |
  | `scenarios/harness.py` | 进程内的控制面（带控制面单元）+ 两台假主机，通过控制接口改工作区、`apply`（`Scenario`、`async with scenario(...)`、`build_workspace`、`rungs`） |
  | `scenarios/coverage.py` | 覆盖检查要数的东西：6 种边、动作阶梯 0–6、3 种模块 |

- 被多个测试文件共用的代码放进上面这类辅助模块（同目录或 `support/`），**不要从另一个 `test_*.py` 里导入**。fixture 以 `from … import harness as harness` 的形式在测试文件中重新导出。

### `doc/`

- `doc/modules.html`：**设计真源**。冲突时：`modules.html` > `v0-plan.md` > `discussion-summary.md`。
- `doc/roadmap.md`：阶段和验收。
- `doc/specs/`：框架本体七个组件的规格，以及控制面（`control-plane.md`：控制面进程和四个控制面单元）、变更只作用于空引擎（`empty-apply.md`：除了只改 live 键，每次变更前排空整个引擎，有请求时默认拒绝；取代 `state-changes.md` 的排空与版本迁移）、阶段 2 的单元契约（`stage2-contract.md`：单元清单、接口、分工、里程碑、分支和合并；草案）、投机解码（`mtp.md`：照 SGLang 的 MTP 和 EAGLE 链式验证，2026-11-13）、v1 多卡和多机的讨论记录（`v1-distributed.md`：未定，2026-11-13）、v1 的盘点和计划（`v1-ds1017.md`：DS 1017 的结构、权重和状态的大小、能对照的实现、阶段，2026-11-14）、CUDA Graph 实现方案（`cuda-graph.md`：TP1 decode graph、最小改动与 fail fast、暂停期重录、首版实现选择和验收矩阵）；前缀复用的计划（`prefix-cache.md`：照 SGLang `e838360` 的 radix cache 加 mamba `extra_buffer`，2026-11-13）；full attention 的 kernel 接入（`attn-kernel.md`：两族 kernel 和怎么选、paged operator 的约定、KV 槽列表和写入、decode graph 里的固定 buffer 和 padding lane；描述的是 #51 和 #53 的实现）；`README.md` 是总览和公共约定（标识、清单、信封、结果、测试约定）。
- `doc/v0-plan.md`、`doc/discussion-summary.md`：v0 计划、讨论纪要。
- `doc/evaluations/`：历次修复 / 对抗测试的评估记录，属于历史，不随代码更新。
  - `doc/evaluations/gpu-probes/`：阶段 1′ 的 GPU 探针脚本（`run.sh` 一次跑完）和原始输出 `logs/`，结论在 `doc/evaluations/gpu-probes.md`。探针用完就扔，不进骨架；不在 pyright 和 pytest 的范围内（要 GPU、torch 和第三方 kernel 库）。
- `doc/framework-bugs/`：热改时发现的更新机制 bug 的报告，格式见其中的 `README.md`。
- `doc/rsi/`：agent 用热更新做开发的过程记录（文件名带日期）：一次工作里实际热改了多少、冷启动了多少、为什么，以及专门对更新机制做的试验和结论。属于历史，不随代码更新。发现的 bug 仍然写进 `doc/framework-bugs/`。
- `doc/archive/`：旧版文档，不再维护。
- `doc/figures/`：图片。

## 在跑着的引擎上热改

AGENTS.md 规定默认这样开发：把引擎拉起来，改完用 `apply` 生效。这样开发本身就是对热更新机制（变更协议、单元运行时的重载、`core.deps` 的计划）的测试。碰到机制本身的问题，按 [`doc/framework-bugs/README.md`](../../../doc/framework-bugs/README.md) 写报告；实在绕不过去，就重启引擎，退回每次冷启动的开发方式。热改不代替提交前的 pyright、pytest。

### 起一个引擎

按这次要改、要看的东西起，需要什么就起什么。下面只是几个例子，没有推荐的哪一种：

- 用 `deploy/agxinfer.py`（说明在 `deploy/README.md`）在一台机器上起整个引擎：前端、调度宿主、一个 worker。下面 `A="uv run python deploy/agxinfer.py"`：
  - 真权重，在后台跑：`$A serve <模型目录> --detach`，引擎起来就返回；不加 `--detach` 就在前台跑到 Ctrl-C；
  - 只要前几层、dummy 权重：再加 `--layers N --dummy`；
  - 整个引擎跑在 CPU 上：加 `--device cpu`，模型可以用 `$A checkpoint` 生成的小随机模型（`tests/scenarios/test_m2a.py` 就是这样起的）；
  - 没有专门参数的配置键：`--set 键=值`（值按 JSON 解析，解析不了就当字符串）。
- 直接用控制面，起任意拓扑和单元组合（单机时 `--spawn-agents` 在本机给拓扑里每台主机起节点代理）：

  ```sh
  PYTHONPATH=engine uv run python -m core.control serve --source <工作区> --topology <topology.json> \
      --state-dir <目录> --host <HostId> --bind <地址> [--config <config.json>] --spawn-agents
  ```

- 测试里的起法也可以参考：`tests/scenarios/harness.py`（进程内控制面加两台假主机）、`tests/units/scheduler/test_processes.py`（真实控制面加夹具 worker 进程）。

起来以后：
- **一个运行目录就是一个引擎**（`--run`，默认 `run/agxinfer`），后面的命令都靠它找到引擎：
  - `$A status --run <目录>`：在不在跑、URL、控制面地址、身份、各进程的状态；
  - `$A stop --run <目录>`：停引擎，等它的进程全部退出，到时限（默认 60 秒）还没退的 kill 掉；
  - 同一个运行目录里已经有引擎在跑时，`serve` 会拒绝。
- 控制面地址：`serve` 打印的 `engine up: ... (control <地址>)`，也写在 `<run>/engine.json` 和 `<run>/control/control.json` 里。用 `$A control` 就不用自己找。
- 引擎的源码是 `--source` 指向的工作区里的 `engine/`（默认是仓库本身），各进程从按身份物化的快照目录加载它。
- 各进程的日志：`<run>/host-a/snapshots/.run/<角色>-<编号>-<n>/log`；控制面和节点代理的日志在 `serve` 自己的输出里（`--detach` 时在 `<run>/engine.log`）。
- 想用这张卡跑 GPU 测试又不想停引擎：`$A sleep`，见下面「让出显存」。

### 改一次

在工作区里改文件，然后：

```sh
A="uv run python deploy/agxinfer.py"   # 加 --run <目录> 指定引擎，默认 run/agxinfer
$A control check --source <工作区>              # 静态检查：清单、服务、跨单元 import、配置键；不排队、不碰引擎
$A control apply --source <工作区> --dry-run    # 影响分析：哪些进程重载哪些单元、要不要重启引擎、notes
$A control apply --source <工作区> --intent "为什么改"
$A control apply --config '{"sched.overlap": true}'   # 只改配置，不改代码
$A control status                               # 也可以是 introspect（单元、配置、谁读了哪个键）、logs（变更日志）
```

- `$A control ...` 就是 `python -m core.control.client --control <地址> ...`，地址从运行目录里找。
- `apply` 默认打印摘要（状态、身份、notes、每个进程动了什么、诊断），`--json` 打印完整结果；退出码 0 表示提交了（`--dry-run` 时是规划成功）。
- 变更只作用于空引擎（`doc/specs/empty-apply.md`）：除了只改 live 键，引擎里有请求时 `apply` 被拒，诊断里写着有多少请求。接着：
  - `--wait-empty [秒]`：前端关门（新请求回 503），等里面的跑完再 apply，默认最多 3600 秒，到了就失败、什么都不中止；
  - `--force`：立刻结束里面的请求（客户端收到 503），再 apply。
- 改完用请求验证：`$A chat "..."`、`$A complete "..."`，或直接对前端发 HTTP；需要时对照冷启动的结果。

### 让出显存（sleep / wake）

2026-11-11 加的，记录见 `doc/evaluations/recovery-sleep-cli.md`。

- `$A sleep`：一次变更（`holder.sleep` 设成 true）。worker 把权重换成不占内存的 meta 张量，KV 和 GDN 状态池缩到每种一项，不录 graph；调度器拒绝所有请求，错误信息说引擎睡着了。和 vLLM 的 sleep mode、SGLang 的 `release_memory_occupation` 是一类东西。
- `$A wake`：改回去。从 checkpoint 重新读权重，重建状态池，重录 graph。`refit` 换过的权重会回到 checkpoint。
- 4090 D 上 4 层的引擎：21,482 MiB 降到 962 MiB（剩下的是 CUDA 上下文和 kernel 库的工作区），睡 2 秒，醒 3.6 秒，醒来输出和睡前逐字相同。睡着的时候 `tests/gpu/` 八个文件全部能跑。
- 和别的变更一样要求引擎是空的，`--wait-empty`、`--force` 照常可用。
- 第一次醒来的身份和睡之前不一样：`holder.sleep` 这时显式写进了配置。之后每次醒来都是同一个身份。
- 醒的时候读权重在暂停期里，受执行时限 `change.execute_timeout_s`（默认 3600 秒；2026-11-12 以前是 `supervisor.rpc_timeout_s` 的 30 秒，64 层模型在 SM100 上不够）约束。完整 64 层模型上还没试过。

### 换权重（2026-11-13）

在 4090 D 上实际试过，记录见 `doc/rsi/2026-11-13-weight-swap-on-a-live-engine.md`。都是普通的变更：引擎要是空的（`--wait-empty`、`--force` 照常可用），前缀缓存在提交时丢掉，graph 在暂停期内重录。换完照常用请求验证，需要时对照用新配置冷启动的输出。

- **换一份 checkpoint**：`$A control apply --config '{"weights.path": "<目录>"}'`。
  - worker 重载 `core.state.weights` 和用它的单元，在暂停期内读新的 checkpoint。旧权重先释放再读新的，不用放得下两份。
  - 4090 D 上 4 层（7.6 GiB）整个 apply 3.2 秒，输出和用这份 checkpoint 冷启动逐字相同。大模型读权重要几分钟，受 `change.execute_timeout_s`（默认 3600 秒）约束。
  - 只换内容：新 checkpoint 的结构要和现在的模型一样（模型结构是 `model.hf_config` 等别的键）。只试过同结构的两份随机 checkpoint。
  - 目录里没有 `*.safetensors`：`aborted`，旧权重从原来的 checkpoint 读回来，照常服务。所以撤回也要重读一遍，大模型同样要几分钟。
- **换 dtype**：`$A control apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'`（`serve --dtype`、`--conv-dtype` 对应的键）。
  - 权重按新 dtype 从 checkpoint 重读。holder 按新权重的大小重新给状态定数量，KV 池和 GDN 池跟着重新申请，和用这个 dtype 冷启动一样：4090 D 上 4 层，bf16 是 656 个槽、107,020 页，f32 是 250 个槽、20,934 页。输出和冷启动逐字相同，改回来也一样。
  - 池子大小显式定死了的（`state.kv.pages`、`state.gdn.slots` 不为 0），不跟着变，可能放不下：`aborted`，旧版本照常服务。
  - 2026-11-13 以前这个变更在 24 GB 的卡上 OOM、撤回也失败，只能重启（`doc/framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md`）。
- **`refit`**：`core.state.weights` 的动作，把 checkpoint 读进同一组张量，不重录 graph，撤回时读回原来的内容。现在只有引擎内部发起的变更能请求它（`ChangeSet.state`），控制接口和 `$A control` 还没有入口。从外面换权重用 `weights.path`。
- `$A wake` 从 `weights.path` 重读，`refit` 换过的内容回到 checkpoint。

### 投机解码（MTP，2026-11-13）

做法和数见 `doc/specs/mtp.md`，过程见 `doc/rsi/2026-11-13-mtp-on-a-live-engine.md`。

- 打开：冷启动 `serve --speculative-algorithm EAGLE`（还有 `--speculative-num-steps` 等，和 SGLang 同名，默认 3、1、4），或者在跑着的引擎上 `$A control apply --config '{"spec.algorithm": "eagle"}'`：调度宿主重载 `core.sched`、`core.state.alloc`，worker 重载模型、GDN 存储和后端、executor，几秒。关掉是把它改回 `""`。checkpoint 里要有 `mtp.*`；`$A checkpoint` 生成的有（随机的）。
- 打开以后正常的现象：KV 页变少（草稿模型的那层 attention 也占页）、GDN 槽变少（验证要的中间状态先从预算里扣）；`core.plan` 录的是验证步的 graph（日志里 `"verify": true`），不录 decode 的；还录草稿模型的 decode forward 和验证步之后的草稿扩展（`"draft": true`）。草稿扩展用验证步 prepare 好的 attention metadata，只跟在回放过的验证步后面回放。
- 随机权重的 MTP 几乎什么都不接受，所以本机上开投机更慢，这是对的：4 层时 LM head 占满带宽（mtp.md §6.4）。要看接受多个 token 的路径，测试里用 `Oracle` 换掉 `models.qwen38` 实例的 `draft_forward`。录了 graph 的，换掉以后还要重载 `core.plan`（`tests/gpu/test_speculation.py` 的 `draft_with`），否则回放的还是录 graph 时的草稿模型；`Oracle` 只用张量运算，graph 录得进去。
- **投机的 graph 有没有真在用**：用 `plan.cuda_graph = required`（`--cuda-graph required`）冷启动。graph 过期时，第一个投机步就报 `StaleGraph`（草稿、验证、草稿扩展都这样），不会悄悄退回 eager；`tests/gpu/test_speculation.py` 也这样测。2026-11-13 的第一版里，graph 一录完就过期，`auto` 下全退回了 eager。我在 4090 D 上没查出来，是 Codex 在 SM100 上发现的（`doc/evaluations/mtp-sm100.md`），原因见下一节第 11 条。
- 开投机前后文本不同时，先分清是 kernel 的数值还是 bug：用 fp32，并且两边都 `--set gdn.backend=reference --set attn.backend=reference`，再比。2026-11-13 就是这样查出开重叠时清掉刚 prefill 完的 GDN 状态那个 bug（只在 GPU 上、请求一个一个来时出现）。

### 撤回没成功、引擎要求恢复的时候

- 表现：`apply` 的结果是 `failed`，诊断里有 `escalate.restart_process`（"reload rolled back, but units did not come back: `<单元>` is FAILED: `<原因>`"）、`change.abort_unacknowledged`、`change.recovery_required`。之后每次 `apply` 都被拒。
- **这时 `status` 里各进程的单元是过期的**：那是进程上次检入、确认提交或确认撤回时报的，撤回没得到确认就不更新，可能还显示 ACTIVE、旧的代数，其实是 FAILED。看真实状态用 `$A control introspect`，它现问每个进程。请求发过去可能一直没有回应。
- 处理：v0 没有在线恢复，`$A stop` 再 `serve`，重启引擎。先从诊断里的原因和 worker 日志找为什么没回来；撤回本该成功的，按 `doc/framework-bugs/README.md` 写报告。
- 2026-11-13 以前，这种时候诊断里还有一句 "reload rolled back; the previous units are serving"，是错的。现在只有单元全都回来了才这么说。
- 撤回时 OOM，一个原因是 PyTorch 缓存的碎片：诊断里 "reserved by PyTorch but unallocated" 很大，加上设备空闲其实够。释放了旧的块以后，新的一批大申请会从旧块里切，把它们钉住，下一块大的就放不下了。要大块显存的单元，先放掉旧的、`torch.cuda.empty_cache()`，再申请。权重存储和两个状态存储都这样做了，新写的这类单元也照做。

### 在引擎里做检查（排查和验证）

2026-11-11 在 4090 D 上实际试过，记录见 `doc/rsi/2026-11-11-real-kernels-hot-apply.md`。

**推荐的路径**：要排查一个问题，或者验证一个改动算得对不对、跑得快不快，先把检查临时写进单元的代码里，apply，发请求，读 worker 的日志；查完把临时代码去掉再 apply。不要一上来就去写 pytest、在测试进程里另搭一套 worker 栈。最后要留下来的回归测试再写成 pytest。

一轮是这样的：

1. 临时检查写在**另一份工作区**里，不写进仓库。`apply --source` 收任何带 `engine/` 的目录：

   ```sh
   WS=<临时目录>
   rm -rf $WS/engine && cp -r engine $WS/   # 每轮之前同步一次（先删，仓库里删掉的文件也就没了）
   # 在 $WS/engine 里改单元的代码，临时的几行前面加一行 `# PROBE` 注释
   ```

   仓库里的代码改了，要再同步一次，否则 apply 上去的是旧的。万一写进了仓库，`tests/scenarios/test_no_temporary_checks.py` 会因为 `# PROBE` 失败。
2. `$A control apply --source $WS --dry-run` 看计划，`$A control apply --source $WS`。
3. 发请求（`$A chat`、`$A complete`，或直接发 HTTP）。
4. 读结果：
   - `print(..., file=sys.stderr, flush=True)` 的输出在 worker 的日志里：`<run>/host-a/snapshots/.run/worker-<i>-<n>/log`，`core.plan: {...}` 也在这个文件里。日志只追加，先记下行数或用 `grep -c` 取增量。
   - 大的东西（张量、profiler 的 trace）存成文件。worker 进程的当前目录就是日志所在的目录，`os.getcwd()` 下的文件落在那里。
5. `$A control apply --source .`（仓库本身），检查就撤掉了；改过的配置也改回去。用 `$A status` 确认身份回到了原值。

试过的四种检查：

| 要看什么 | 加在哪 | 怎么做 |
|---|---|---|
| 一个 kernel 和参照实现在真实激活上差多少 | 调这个 kernel 的层（`layers/*/unit.py`） | 参照实现从目录里取：`catalog.resolve("reference.<definition>")`。有状态的 kernel 先把这几行的状态拷出来（`pool[idx].clone()`），参照实现在拷贝上按 0..n-1 的下标跑，再比输出和状态 |
| graph 回放和 eager 的 logits 是否相同 | `executors/pytorch/unit.py` 的 `submit`，`chosen.run(...)` 之后 | 每步把 logits 存文件。同一个请求在 `plan.cuda_graph` 为 `off` 和 `auto` 时各跑一遍（`apply --config`），离线比 |
| 每步发了哪些 kernel、各花多少时间 | `submit` 开头 | 在实例上数步数，第 N 步进 `torch.profiler.profile(...)`，第 N+20 步 `torch.cuda.synchronize()`、退出、`export_chrome_trace` 到文件。Kineto 看得到 graph 回放里的 kernel |
| 显存到哪去了（重载或撤回时 OOM） | 单元加载、卸载的前后；要看卸载和撤回之间的，加在 `unit.runtime` 的 `_reload`、`_rollback` 里（框架本体，只能冷启动带进去，2026-11-13 这样查过） | `torch.cuda.memory_allocated()`、`memory_reserved()`；`gc.get_objects()` 里的 CUDA 张量按 `untyped_storage().data_ptr()` 去重求和；`gc.set_debug(gc.DEBUG_SAVEALL)` 后 `gc.collect()`，`gc.garbage` 里的就是循环引用里还没回收的，看完 `gc.garbage.clear()`、`gc.set_debug(0)`，状态和查之前一样 |

**好处**：

- **快**。一轮 apply 两三秒；冷启动光启动那次变更就要 10 秒左右，还要重读权重。
- **看到的是真的**：真实请求的激活、真实的调度路径。单测里的随机张量和测试夹具都只是近似。
- **引擎不用停**。一张卡放不下引擎和 GPU 测试同时跑时，在 pytest 里查就得先停引擎，回来又是冷启动。
- **顺带测了热更新机制**，这是 AGENTS.md 要的。
- **多数写错的情况是安全的**：加载时或录 graph 时出错的改动会被撤回，引擎照常服务旧代码。处理请求时才出错的，见下面第 3 条。

**问题和注意点**：

1. **回放的 graph 里不跑 Python。** 检查只在 eager 步和录 graph 时执行，回放的 decode 步里什么都不会打出来。要看 decode 步，先 `apply --config '{"plan.cuda_graph": "off"}'`，查完改回 `auto`。
2. **被录进 graph 的前向里不能在 host 上读张量的值**（`.item()`、`float(t)`、`.cpu()`、打印张量）。capture 会失效，apply 被撤回，诊断是 `AcceleratorError ... during capture`。
   - 在层里做检查时按 backend 的元数据跳过 graph 桶（`layers.attn`、`layers.gdn` 里是 `m.graph`）；
   - 或者把检查放在 graph 之外：executor 的 `submit` 里、`chosen.run(...)` 之后；
   - 也别把参照实现的计算留在被录的前向里，它会被录进 graph。
3. **检查代码在处理请求时抛异常，这一步失败，引擎里的请求立刻全部回错误**（2026-11-11 起；以前在途的请求一直挂着，改好再 apply 也不恢复，见 `doc/framework-bugs/2026-11-11-a-repaired-worker-stays-stopped.md`）。
   - 之后的新请求回 400，错误信息里有异常的文字，直到下一次变更提交；
   - 把代码改好（或撤掉检查）apply 就恢复服务，不用 `--force`，也不用重启。真的 CUDA OOM（比如 `holder.mem_fraction` 调得太大）也一样：把配置改回来 apply；
   - apply 救不回来的：CUDA 上下文坏了（非法访存、device 端 assert）、进程死了、某一步卡住。这些要重启引擎；
   - 预防：可能失败的部分包在 `try/except` 里，只打印不抛。检查写在层或模型里时，录 graph 的引擎（`auto` / `required`）在 apply 时会重录 graph，先跑一步 eager 的 padding 行，检查的输出这时就会出现在日志里，可以先看它对不对，再发请求。写在 executor 里的检查要等第一个请求才会跑到。
4. **读张量的值要等设备**，重叠（`sched.overlap`）就没用了，吞吐会掉。带着这种检查量出来的时间不能用；量时间用 profiler，或者用不带检查的引擎。
5. **静态检查照常管着临时代码**：
   - 单元之间不能互相 import（`manifest.cross_unit_import`）。别的单元的东西通过服务拿，参照 kernel 通过 `kernels.catalog@1`；
   - 模块级不能放可变对象（`manifest.mutable_global`）。计数器、profiler 对象放在实例上（`self._probe_n`）。
6. **实例上的状态会清零**：单元被重载时清，它依赖的单元被重载时也清。例如重载 `core.plan` 会连带重建 `executors.pytorch`。
7. **有状态的 kernel 原地改状态池**，对照时别在真的池上再跑一遍，用拷贝。
8. **每次 apply 都要求引擎是空的**。临时代码和临时配置会改变引擎的身份，查完要改回去。
9. **这条路径不代替测试。** 能复现的问题最后写成 pytest（`tests/gpu/` 或 `tests/units/`），提交前照常跑 pyright 和 pytest。
10. 只试过 worker 里的单元（层、executor）。前端和调度宿主里没有这样试过。
11. **检查本身可能让问题消失**（2026-11-13）。
    - 检查写进哪个单元，apply 时那个单元和依赖它的单元就重载。
    - 热改时状态单元保留着上次建的东西，所以重载走的初始化顺序和冷启动不一样。
    - 例子：为了看验证步的 graph 有没有回放，我把打印写进了 `core.plan`。它重载时 batch 里投机用的表已经在了，重录的 graph 是好的，打印出来"在用"；冷启动时 graph 其实一直是过期的。
    - 所以查"冷启动以后是什么样"，检查要写在不会让被查的单元重载的地方，或者带着检查冷启动。

### 哪些是正常的，哪些可能是 bug

正常，按提示做就行：
- 引擎里有请求时被拒；
- worker 某一步出错：引擎里的请求立刻全部回错误，新请求回 400，直到下一次变更提交，之后照常服务；
- 引擎睡着（`holder.sleep`）时请求回 400，说引擎睡着了；
- 变更放不下（比如 OOM）：`aborted`，诊断最后是 "reload rolled back; the previous units are serving"，旧版本照常服务，不要求恢复；
- 改了框架本体（`engine/framework.toml` 列的模块）或控制面单元的原生扩展：`escalate.restart_engine`，只能重启引擎（AGENTS.md「框架改动」）；
- 改了原生扩展、拓扑、放置或平台：变更自动重启对应的进程；
- 改了权重存储 `core.state.weights` 的代码：在暂停期内从 checkpoint 重新读，大模型要几分钟；`refit` 换过的权重回到 checkpoint；
- 静态检查不过（`check` 或 `apply` 的诊断写明了哪条规则）；
- 改了 `kernels.cuda` 的 C++ 源码（`kv_store_cu.py`）直接 apply：它的构建步骤（`[unit] build`）在准备阶段另起一个进程编译，4090 D 约 55 秒，SM100 / CUDA 13 约 123 秒；这段时间引擎照常服务，暂停期内的重载从缓存加载（2026-11-11 起，见 `doc/evaluations/build-steps-and-stream-liveness.md`）。准备阶段的时限是 `change.prepare_timeout_s`（默认 600 秒，`deploy/agxinfer.py` 设成 `--timeout`）。编译失败时变更在暂停之前就失败，诊断是 `change.build_failed`，编译器的输出在 worker 的日志里。以前要先在引擎外编译（`PYTHONPATH=engine uv run python -m kernels.cuda.ops`），现在不用了，这样做也还行。

可能是更新机制的 bug，写报告：
- 提交了，可跑的还是旧代码，或者新旧代码混着跑；
- 提交以后请求出错、引擎停止服务，或者输出和用同一份代码冷启动的不一样；
- 计划不对：`--dry-run` 漏了该重载的单元、重载了不相干的、该要求重启的没要求（或反过来）；
- 撤回（`aborted` / `failed`）以后行为和变更之前不一样，或者引擎进入要求恢复的状态（怎么处理见上面「撤回没成功、引擎要求恢复的时候」）；
- 同一个改动 apply 两次结果不同，或者 `--dry-run` 的计划和实际执行的不一样；
- 进程、显存、文件句柄这类资源在多次 apply 之后一直涨。

### 踩过的坑

- 停引擎用 `$A stop --run <目录>`。别自己按命令行找进程杀：`pkill -f` 会匹配到发命令的 shell 自己；`ps | grep` 可能先匹配到外层启动 `serve` 的那个进程，杀了它，控制面和它起的进程都还在，显存也不放。`stop` 按进程参数精确地找：控制面的 `--state-dir <run>/control`，节点代理和进程宿主参数里 `<run>/host-a/` 下的路径（快照、spec）；运行目录下面另一个目录里的引擎不算它的。
- GPU 上的第一步可能很慢；超时都按 `--timeout` 放长了。
- `$A checkpoint` 生成的 checkpoint 很大（27B 结构的 4 层约 7.6 GiB），用完删掉；快照目录是只读的，删之前先 `chmod -R u+w`。
