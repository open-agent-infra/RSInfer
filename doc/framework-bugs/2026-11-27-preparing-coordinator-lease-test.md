# 协调者消失测试可能完成全部 prepare 分片才回滚

- 日期：2026-11-27。
- 报告人：Codex。
- 状态：未修；原始基线可复现，本次缓存/PD 工作不修改框架。
- 引擎：现有 `tests/framework/change/test_in_doubt.py` 的三参与者 fake topology；Ubuntu 22.04、Python 3.12.16、pytest 9.2.0、容器 10 CPU。不是模型服务的 apply 故障。
- 变更：测试向参与者发送 200 个 prepare 分片（每片 0.05 秒），注入 `vanish_during_prepare`，租约为 1 秒。不涉及新增缓存或 PD Unit。
- 现象：变更最终 aborted、参与者回滚，但 `len(e.slices(p)) == 200`，不满足现有测试的 `0 < len(...) < 200`。单项执行约 12.9 秒；在最初全量运行中曾通过，后续独立执行和全量执行出现失败。
- 期望：依据该测试与 change.md §4.7，协调者消失时参与者应在准备阶段退出，而非完成全部分片后才发现。
- 复现：在原始基线 `bd26c7e1554211a578e733f7e2fc59feb3a21452` 的干净 archive 中，用同一个锁定的 CPU 环境执行：

  ```sh
  OMP_NUM_THREADS=1 uv run --no-sync pytest \
    tests/framework/change/test_in_doubt.py::test_a_participant_still_preparing_withdraws_when_the_coordinator_vanishes -q
  ```

- 观察到的代码线索：`core.change.participant._advance` 将每片工作的耗时加回 deadline；200 个 0.05 秒的工作本身不消耗租约，只有片间时间消耗。片间累计不足 1 秒时可能跑完全部分片。`tick` 在 `_advance` 后检查期限，而准备结束会续租。这解释了速度/调度变化可能影响断言；这里只记录线索，不把它当作已验证的完整根因修复。
- 绕过：未跳过、放宽或修改该测试；继续执行功能回归和真实 CPU/GPU 集成，并把基线同样失败作为交付报告中的已知问题。缓存/PD 常驻引擎的 apply/撤回与请求对照已单独通过。
