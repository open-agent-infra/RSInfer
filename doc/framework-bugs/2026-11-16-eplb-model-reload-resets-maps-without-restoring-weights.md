# EPLB 重排后热重载模型，专家映射回到初始顺序而权重保留重排后的顺序

- 日期：2026-11-16
- 报告人：Codex
- 状态：已修（PR #66，见文末「修复（2026-11-16）」）；发现于 `9d3877f1eda9b2f87937cbbfd5eb401b98750c12`

## 动机与设置

核对在线 EPLB 与模型热更新是否能组合使用。EPLB 会实际移动专家权重；模型代码重载应保留权重及其当前映射的对应关系。

在 `dev-server` 启动真实前端、调度器和两个 worker，使用小尺寸 DS 1017 随机 checkpoint、CPU、reference 后端，`--ranks 2`，`eplb.interval=4`、`eplb.window=4`、`eplb.extra_experts=2`。每次请求贪心生成 12 token。探针仅写在临时源码工作区，记录映射及权重存储的加载次数，不改变计算。

## 复现与结果

1. 对三个提示词各请求一次，再重复一遍。两轮文本完全一致；日志确认 EPLB 已多次移动专家。
2. 在临时工作区的 `engine/models/ds1017/unit.py` 末尾增加一行注释，执行正常的 `control apply --source`。
3. 变更 `committed`，两个 worker 均原地重载 `models.ds1017` 及其消费者；权重存储没有重载。
4. 再请求同样三个提示词，三份输出全部改变。

```text
BASELINE_REPEAT_EQUAL True
APPLY status=committed
worker/0, worker/1: reload=[models.ds1017]
HOT_RELOAD_EQUAL False
```

例如第一个提示词热改前为：

```text
w43 w25 w130 w168 w94 w84 w19 w109 w118 w98 w282 w79
```

热改后为：

```text
w23 w94 w54 w90 w109 w216 w127 w30 w265 w244 w230 w46
```

权重存储的 `loads` 始终为 1。重载后的第一次 `_layer_weights()` 调用前，第 0 层的 `placement.held` 为 `[0, 2, 1, 2, 0, 3]`；随后 `Placement.start()` 按初始顺序 `[0, 1, 2, 3, 0, 1]` 重建映射，而实际权重没有搬回去。

## 原因与建议

新模型实例的 `_cached` 初始化为 `(-1, [])`。第一次取权重视图时，`models/ds1017/unit.py:356` 无条件调用 `placement.start()`。后者按 checkpoint 的初始顺序重建 `held`、`to_physical` 和 `copies`。

然而 `WeightStore.load()` 发现同一组权重已填充，就保留了 EPLB 搬移后的内容。结果是映射说某个槽位是专家 A，槽位里实际装的是专家 B。后续继续重排也会按这个错误映射搬权重。

专家位置元数据需要与当前权重作为同一份持久状态保留。模型重载时应重新绑定权重访问回调，只有权重确实重新加载或恢复到初始布局时才重建初始映射。EPLB 参数/策略重载也需要遵循同一关系。

## 记录

- 脚本：`dev-server:/tmp/agxinfer-pr66-codex/repro_eplb_reload.py`
- 日志：同目录 `eplb-reload.log`，退出码 0。
- 运行目录与临时源码：`/tmp/agxinfer-pr66-eplb-reload/`。
- 未修改仓库实现；复现引擎已停止。本轮没有重复运行已通过的 CI 套件或完整 SM100 模型。

## 修复（2026-11-16，Antigravity）

照报告的建议：专家在哪随权重一起保留，读权重的单元重载时按它重建映射，只有权重真的重新写过才回到初始布局。

- **`models.ds1017`**：权重里多声明一个 `experts.placed`（`[层数, 槽位数]` int32，每个 rank 一份完整的表）。`load_weights` 把它写成初始布局（checkpoint 的顺序），和 `experts.layout` 标记一样由权重存储保管，模型重载不动它。`_layer_weights` 把它交给 `Placement.start`。
- **`state.expert_placement`**：`start` 按传进来的 `experts.placed` 建 `held`、`to_physical`、`copies`（每层各自建，不再假设每层相同）；`rebalance` 搬完权重同时写回 `experts.placed`。所以：
  - 模型重载（本报告的情形）：新实例读到的是搬过以后的表，映射和权重一致；
  - `state.expert_placement` 重载（改它的代码、改 `eplb.*` 而槽位数不变）：同样；
  - 关掉 EPLB 而槽位数不变（`eplb.interval = 0`、`extra_experts = 0`，权重还在搬过的位置）：`active` 为假，不再重排，但 kernel 仍读映射（`mapped`），计数传 `None`；以前这种情形 kernel 按\"槽位 i 是专家 i\"读，同样会错位；
  - 槽位数变了：权重重新声明、重新装载，表回到初始布局。
  - 形状相同的映射缓冲区原地写（`_put`），不换地址：录在 graph 里的地址继续有效。
- **`core.state.weights`**：加一个 `fills` 计数，每次 `fill` 加一，`refit` 撤回时也不放回（`loads` 会放回）。模型按 `(generation, fills)` 判断要不要重新推导依赖权重内容的东西（专家的行顺序标记、专家的位置）。以前只看 `generation`，`refit` 原地重写内容时不变：`experts.layout` 回到 0 而模型以为还是融合 kernel 的顺序，专家位置也一样，是同一类问题。
- 测试：
  - `tests/units/state/test_expert_placement.py::test_a_placement_started_again_reads_where_the_experts_were_moved_from_the_record`：两个 rank 重排以后，新的 `Placement`（还在重排的和已关掉的各一组）按记录建映射，每个专家的映射指向装着它的行；初始布局的记录不需要映射。
  - `tests/scenarios/test_ds1017.py::test_a_reload_after_the_experts_were_placed_again_finds_them_where_they_were_moved`：报告里的复现写成了测试。CPU、reference 后端、`--ranks 2`、EPLB 重排以后，热重载 `models.ds1017`（加一行注释），再关掉 EPLB（槽位数不变），三次的文本都和一个 rank 算的逐字相同。拿掉修复，这条在第一次热重载后失败。
- 在线增加 EP 卡数（报告附带的代码走查结论）：不修，约定整引擎重启，写在 `.claude/skills/rsinfer/SKILL.md` 和 `doc/specs/v1-ds1017.md` 10.9。

### SM100 上的验证（2026-11-16）

4 张 SM100、整个模型、`--ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 64`，从修复后的 `4794e13` 冷启动一次，之后全部热改。三个提示词（英文、中文、代码）一个一个问，贪心 32 个 token：

| 变更（change id） | worker 的计划 | 耗时 | 文本 |
|---|---|---:|---|
| 2：`eplb.interval=16, window=16, extra_experts=0` | 重载 `state.expert_placement` | 5.1 s | 8 路并发请求里重排 3 次，共 44,202 个槽位换了专家；之后的文本记作 A（也和开 EPLB 前相同） |
| 3：`models/ds1017/unit.py` 末尾加一行注释，`apply --source` | 重载 `models.ds1017`（`changed_modules: models.ds1017.unit`） | 4.3 s | 与 A 逐字相同（本报告的情形） |
| 4：`eplb.interval=0`（槽位数不变，权重留在搬过的位置） | 重载 `state.expert_placement` | 4.5 s | 与 A 逐字相同 |
| 5：注释去掉，`apply --source` | 重载 `models.ds1017` | 3.7 s | 与 A 逐字相同 |

四次都 `committed`，权重没有重读。
