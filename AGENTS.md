# AGENTS.md

给在本仓库工作的 agent：规矩、怎么验证。项目是什么见 [README.md](README.md)。

**动手前先读 [`.claude/skills/rsinfer/SKILL.md`](.claude/skills/rsinfer/SKILL.md)**：东西在哪（目录地图），以及怎么在跑着的引擎上热改（Claude Code 会把它当 skill 自动加载）。

## 框架改动

- 做单元、模型、kernel、部署或文档任务时，优先在对应实现中完成。框架本体的边界以 `engine/framework.toml` 为准；不要顺手修改框架里的注释、诊断文案、格式或做无关重构。
- 确实需要修框架 bug 或补框架能力时，保持最小改动，在 PR 中说明具体故障、为何单元侧无法解决和验证结果。与当前任务无关的框架改进另开 PR，避免夹带。
- 框架改动会要求引擎重启，并触发框架测试；`tests/framework/`、共用测试设置、依赖和 CI 改动也会触发框架测试。

## 开发方式：在跑着的引擎上热改（2026-11-10）

- 默认这样开发：按这次需要把引擎拉起来，改完用 `apply` 生效，而不是每次改完都冷启动。开发本身也就是在测热更新机制（变更协议、单元运行时的重载、`core.deps` 的计划）。怎么起、怎么改、哪些行为是正常的，见 skill「在跑着的引擎上热改」。
- 排查和验证也先在引擎上做（2026-11-11）：把检查临时写进单元的代码里（写在另一份工作区里，不写进仓库；对照参照 kernel、存 logits、包 profiler），apply，发请求，读 worker 的日志；查完去掉再 apply。不要一上来就在 pytest 里另搭一套 worker 栈。做法、好处和注意点见 skill「在引擎里做检查」；最后要留下来的回归测试仍然写成 pytest。
- 发现更新机制本身的问题（热改以后跑的不是新代码、计划不对、撤回不干净、热改和冷启动结果不同等），在 `doc/framework-bugs/` 写一份报告，格式见那里的 `README.md`。
- 实在绕不过去，就重启引擎，退回每次冷启动的开发方式，并在报告里写明。
- 热改不代替提交前的测试：pyright、pytest 和 formal 照下面「常用命令」跑。

## 目录结构规则（最重要）

- **如无必要，尽量不要新增、删除、重命名或移动顶层目录**（`engine/`、`tests/`、`doc/`、`.github/`），除非用户明确要求。
- **新增或移动任何目录之前，先判断它是不是某个已有目录的子集。** 是子集就放进那个目录，不要和它并列。
  - 例：formal 测试首先是测试，而且测的是框架本体，所以在 `tests/framework/formal/`，不是顶层的 `formal-tests/`。
  - 例：将来给单元写的测试是测试但不是框架本体的测试，所以放在 `tests/` 下、与 `tests/framework/` 并列（如 `tests/units/`），不要混进 `tests/framework/`。
- 优先使用已有目录；只有确认已有目录都不合适时才新建，并在说明里写清楚理由。
- 移动、重命名、拆分、合并目录前，先把方案告诉用户并得到确认。改完要同步：
  - 用 `git mv`，保留历史；
  - 绝对导入 `tests.framework.…`，包括写在字符串里的代码（toy unit 源码、子进程的 `python -c "..."`）；
  - 用 `Path(__file__).parents[n]` 算出的路径：`tests/framework/cluster/harness.py` 的 `REPO`、`ipc/test_shm_ring_wrap.py`、`ipc/test_location_transparency.py`、`formal/run.py` 和 `formal/test_*.py`；
  - `pyproject.toml`（`testpaths`、`addopts`、pyright `include`）和 `.github/workflows/ci.yml`；
  - `.claude/skills/rsinfer/SKILL.md` 的目录地图。其他文档不必逐个追改（见下一节）。

## 文档是开发笔记

- `doc/` 下的文档和各目录的 README 是带日期的开发笔记，记录的是写下当时的想法，不是必须一直保持正确的规范。
- 改代码时**不要**连带更新所有引用到它的文档，也不要为了让旧文档保持最新花大量精力。需要记录新的结论时，写一份带日期的新文档；新文档可以取代旧文档。
- 只有两类文档需要和代码保持一致：这次改动直接对应的文档（例如你正在写的设计或评估记录），以及本文件和 `.claude/skills/rsinfer/SKILL.md`（目录地图、热改的做法）。

## 常用命令

```sh
uv sync --python 3.12 --extra cpu    # CPU 版 torch（torch_eager、模型单元要）；SM100 上用 --extra gpu，两者互斥
uv run pyright                       # strict；检查 engine/ 和 tests/（含 formal）
uv run pytest                        # 默认测试；不收集 tests/framework/formal；串行
uv run pytest -n 8                   # 全量并行（pytest-xdist）：约 30 秒，串行约 165 秒
uv run pytest tests/framework/ipc -q # 只跑某个目录
# 起引擎、热改、停引擎：uv run python deploy/agxinfer.py serve|control|status|stop|sleep|wake ...
#   见 .claude/skills/rsinfer/SKILL.md「在跑着的引擎上热改」和 deploy/README.md
# formal 全流程：需要 Java 11+；首次 --bootstrap 下载并校验 TLC 1.7.4
uv run python tests/framework/formal/run.py --bootstrap --replay [--java /path/to/java]
```

- 每次提交前跑 `uv run pyright` 和 `uv run pytest`（全量用 `-n 8` 并行；只跑一两个文件时别加 `-n`，起 worker 要 5 秒左右）。改了 `core.change`、`core.supervisor`、`core.nodeagent`、`unit.runtime.host` 或 `tests/framework/formal/` 时，也跑 formal 全流程。
- `uv run pytest` 跳过 formal 靠的是 `pyproject.toml` 里 `addopts` 的 `--ignore=tests/framework/formal`；显式给出路径（`uv run pytest tests/framework/formal`）不受影响。`tests/gpu/`（要 GPU，手动跑）同样由 `addopts` 排除。
- `pyproject.toml` 的 `timeout = 120` 是 pytest-timeout 给**单个测试**的上限，不是整个套件的。不要给 pytest 或 CI job 加总时长上限。
- 生成物不进仓库：`tests/framework/formal/.tools/`、`.runs/` 已被忽略。

已知与环境有关的失败：`ipc/test_streams.py::test_slow_consumer_blocks_the_producer[tcp]` 在 macOS 上偶发失败，Linux 和 CI 上正常。

## 约定

- 文档用中文；代码、注释、docstring 用英文。
- 测试不用 `xfail`，也不按平台或缺少工具自动 `skip`。
- 测试用 `@pytest.mark.covers(edges=..., ladder=..., kinds=...)` 声明自己走到了哪些边、动作阶梯的哪几级、哪些模块种类，并在测试里断言；`tests/scenarios/test_coverage.py` 要求每一项至少有一个测试覆盖。
- 框架代码不写死本机地址、路径或 PID（P7）；超时、心跳之类的可调参数一律做成带默认值的配置项（`doc/specs/README.md`）。
- 框架本体对外返回 `Ok` / `Failed`，不靠异常跨进程边界。
- 改框架本体要克制，能不改就不改，只改注释也算改：改了框架本体的变更，`apply` 直接以 `escalate.restart_engine` 失败，只能重启引擎；CI 也会因此多跑一遍 `tests/framework/`。单元需要的说明写在单元自己的代码和文档里。
- 改 `engine/core/contracts/`（理由见 `doc/specs/stage2-contract.md` §3）：
  - 只共享真正跨单元、跨进程交接的数据和必要调用。内部对象字段、算法、队列和辅助类型留在实现里，现有多余定义也应收缩；
  - 同一个 RSI agent 可以同时修改接口与全部使用者。沿用交接方式的算法/kernel/策略改动直接改实现；交接内容或语义变化时才更新共享定义，需要旧新版本并存时才新增版本；
  - 原地改消息定义时，接收方必须兼容在途旧消息的格式和语义；否则先停止该流生产并排空，或使用新分段/版本。步循环的 `quiesce` 不替代前端流的处理；
  - 两位负责人审核仅为当前团队 PR 协作流程，不是运行时的人类审批要求；
  - 模块按实际交接边界隔离，只 import 所需定义；`__init__.py` 只留 docstring；
  - 新加模块或模块之间新加 import，要同时改 `tests/units/contracts/test_contracts.py` 的 `REACHES`。
