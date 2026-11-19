"""Decode graphs without a GPU: the bucket policy, the dependency guard, and the path of a captured
step. A stand-in recorder replays a "graph" by running the recorded forward again, so everything
but CUDA's capture itself runs here: the backends' fixed buffers, the padding of the rows, and the
kernels of a captured step (``kernels.torch_graph``). The real capture is in ``tests/gpu``."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from core.common.ids import InstanceId, UnitId
from core.contracts.columns import Column
from core.contracts.plan import PlanKey
from core.contracts.records import BatchPlan, StateCopy, StateTrack
from core.plan.graph import Dependencies, GraphPlan, buckets
from core.plan.unit import Plans
from executors.pytorch.unit import check_plan
from platforms.base import StaleGraph
from tests.framework.support import PROC
from tests.units.support.graphs import with_graphs
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config
from unit.runtime.proxy import Provision, ServiceProxy


@pytest.mark.parametrize("values", [[0], [-1], [2, 2], [9], [True], [1.5]])
def test_bad_buckets_fail(values: list[int]) -> None:
    with pytest.raises(ValueError, match="graph_buckets"):
        buckets("auto", values, 8, 1024, fixed_extent=False)


def test_default_buckets_and_required_coverage() -> None:
    """Over fixed extents (a captured step runs a stand-in that costs by the context length): powers
    of two up to 8, larger steps run eagerly unless listed. Otherwise SGLang's default sizes up to
    ``batch.max_rows`` or 64, and that size itself, so that ``required`` is covered up to 64 rows."""
    assert buckets("auto", [], 6, 1024, fixed_extent=True) == [1, 2, 4]
    assert buckets("auto", [], 64, 1024, fixed_extent=True) == [1, 2, 4, 8]
    assert buckets("auto", [], 6, 1024, fixed_extent=False) == [1, 2, 4, 6]
    assert buckets("auto", [], 64, 1024, fixed_extent=False) == [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    assert buckets("auto", [], 100, 1024, fixed_extent=False)[-3:] == [48, 56, 64]
    assert buckets("required", [], 6, 1024, fixed_extent=False) == [1, 2, 4, 6]
    for fixed_extent in (True, False):
        assert buckets("auto", [64, 16], 64, 1024, fixed_extent=fixed_extent) == [16, 64]
        assert buckets("required", [6, 1, 4], 6, 1024, fixed_extent=fixed_extent) == [1, 4, 6]
        with pytest.raises(ValueError, match="cuda_graph"):
            buckets("silent_fallback", [], 4, 1024, fixed_extent=fixed_extent)
    for rows, fixed_extent in ((6, True), (64, True), (100, False)):
        with pytest.raises(ValueError, match="cover"):
            buckets("required", [], rows, 1024, fixed_extent=fixed_extent)


def test_decode_length_is_validated_before_state_changes() -> None:
    plan = BatchPlan(Column.of("i32", [0]), Column.of("i32", [2]), Column.of("i32", [0]),
                     Column.of("i64", [4, 5]), Column.of("u8", [0]), "decode", 0)
    with pytest.raises(ValueError, match="exactly one"):
        check_plan(plan)


def test_tokens_in_another_type_than_i64_are_refused_before_their_bytes_are_read() -> None:
    plan = BatchPlan(Column.of("i32", [0]), Column.of("i32", [2]), Column.of("i32", [0]),
                     Column.of("i32", [4, 5]), Column.of("u8", [0]), "prefill", 0)
    with pytest.raises(ValueError, match="i64"):
        check_plan(plan)


def test_the_guard_follows_the_provider_its_generation_and_the_catalog() -> None:
    tensor = torch.zeros(2, 3)
    state = SimpleNamespace(generation=0, buffers=lambda: {"x": tensor})
    catalog = SimpleNamespace(version=0)
    old = Provision("test@1", InstanceId(PROC, UnitId("test"), 1), state)
    proxy = ServiceProxy(old)
    guard = Dependencies([proxy], [lambda: proxy], catalog)
    assert guard.valid()
    tensor.add_(1)  # new contents at the same address: a refit keeps the graphs
    assert guard.valid()
    state.generation += 1
    assert not guard.valid()
    state.generation = 0
    catalog.version += 1
    assert not guard.valid()
    catalog.version = 0
    assert guard.valid()
    old.revoked = True  # the provider was reloaded: another instance, whatever its generation
    replacement = Provision("test@1", InstanceId(PROC, UnitId("test"), 2), state)
    assert not guard.valid()
    assert Dependencies([ServiceProxy(replacement)], [lambda: state], catalog).valid()


def test_a_step_checks_the_dependencies_once_for_all_buckets() -> None:
    checks: list[bool] = []
    holds = True

    class Deps:
        def valid(self) -> bool:
            checks.append(holds)
            return holds

    plans: Any = Plans(None, [])
    plans.graphs = {size: GraphPlan(size, torch.device("cpu"), []) for size in (1, 2, 4)}
    plans.deps = Deps()
    assert plans.buckets("decode") == [1, 2, 4] and plans.buckets("prefill") == []
    assert plans.get(PlanKey("decode", 2)) is plans.graphs[2] and plans.get(PlanKey("decode", 0)) is plans.eager
    assert checks == [True]
    holds = False
    assert plans.buckets("decode") == [] and checks == [True, False]
    with pytest.raises(LookupError):
        plans.get(PlanKey("decode", 8))


def test_a_speculative_step_replays_each_forward_where_it_can_and_its_draft_extend_after_its_verify() -> None:
    """doc/specs/mtp.md: the draft model's decode forwards and the verify step replay in their smallest
    bucket while the dependencies hold; the draft extend replays only right after a verify step that
    replayed, on that step's metadata, else eagerly; a prefill's extend always eagerly. ``required``
    refuses a forward no graph holds instead of running it eagerly."""
    from executors.pytorch.spec import view

    calls: list[tuple[str, int]] = []
    holds = True

    class Deps:
        def valid(self) -> bool:
            return holds

    def graph(name: str, size: int) -> Any:
        def run(*_: Any) -> tuple[str, str]:
            calls.append((name, size))
            return name, name
        return SimpleNamespace(run=run, size=size)

    def eager(name: str) -> Callable[..., tuple[str, str]]:
        def run(*_: Any) -> tuple[str, str]:
            calls.append((name, 0))
            return name, name
        return run

    model = SimpleNamespace(forward_hidden=eager("verify"), draft_forward=eager("draft"))
    plans: Any = Plans(model, [])
    plans.verify_graphs = {size: graph("verify", size) for size in (2, 4)}
    for v in plans.verify_graphs.values():
        v.extend = graph("extend", v.size)
    plans.draft_graphs = {size: graph("draft", size) for size in (2, 4)}
    plans.deps = Deps()
    rows, tokens = [0, 1, 2], torch.zeros(12, dtype=torch.int64)
    decode, verify = view(rows, [1] * 3, [5, 6, 7], "decode"), view(rows, [4] * 3, [5, 6, 7], "decode", 4)

    def step() -> None:
        plans.draft(None, decode, tokens[:3], tokens[:3], tokens[:3], tokens[:3])
        plans.verify(None, verify, tokens, tokens)
        plans.draft(None, verify, tokens, tokens, tokens, tokens[:3])

    step()
    assert calls == [("draft", 4), ("verify", 4), ("extend", 4)]
    calls.clear()
    plans.draft(None, verify, tokens, tokens, tokens, tokens[:3])  # no verify replayed just before it
    plans.draft(None, view(rows, [4] * 3, [0, 0, 0], "prefill"), tokens, tokens, tokens, tokens[:3])
    assert calls == [("draft", 0), ("draft", 0)]
    calls.clear()
    holds = False
    step()
    assert calls == [("draft", 0), ("verify", 0), ("draft", 0)]
    plans.required = True
    with pytest.raises(StaleGraph, match="no ready draft graph"):
        plans.draft(None, decode, tokens[:3], tokens[:3], tokens[:3], tokens[:3])
    with pytest.raises(StaleGraph, match="not replayed"):
        plans.draft(None, verify, tokens, tokens, tokens, tokens[:3])


def test_required_on_cpu_fails_at_start() -> None:
    with pytest.raises(Exception, match="platform has no CUDA graph support"):
        Worker(tiny_config().to_dict(), **{"plan.cuda_graph": "required"})


def test_auto_without_a_graph_platform_runs_eagerly_and_says_why() -> None:
    """The kernels of a captured step are fixed and logged all the same; an eager decode step keeps
    the reference, not the fixed-extent kernel of the graphs."""
    w = Worker(tiny_config().to_dict())
    try:
        plans = w.service("plan@1")
        assert plans.resolved["cuda_graph"] == {"mode": "auto", "buckets": [],
                                                "reason": "platform has no CUDA graph support"}
        assert plans.resolved["decode"]["layers.attn@1"]["attn.decode@1"] == "reference.attn.decode@1 (fallback)"
        assert plans.resolved["graph"] == {
            "layers.attn@1": {"backend": "auto", "kv.store@1": "reference.kv.store@1 (fallback)",
                              "attn.decode@1": "torch_graph.attn.decode@1"},
            "layers.gdn@1": {"backend": "auto",
                             "gdn.qkvzba_conv1d_update@1": "reference.gdn.qkvzba_conv1d_update@1 (fallback)",
                             "gdn.packed_decode@1": "reference.gdn.packed_decode@1 (fallback)",
                             "gdn.track_states@1": "reference.gdn.track_states@1 (fallback)"}}
        assert not plans.required and plans.buckets("decode") == []
    finally:
        w.close()


def test_a_backend_without_a_capturable_decode_kernel_is_named() -> None:
    w = Worker(tiny_config().to_dict(), **{"attn.decode_backend": "reference"})
    try:
        assert w.service("plan@1").resolved["graph"]["layers.attn@1"]["attn.decode@1"] == ""
    finally:
        w.close()


# ---------------------------------------------------------------- a captured step, on the CPU

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_checkpoint(hf_model(tiny_config()), tmp_path_factory.mktemp("graph-weights"))


@pytest.fixture
def pair(checkpoint: Path) -> Iterator[tuple[Worker, Worker]]:
    eager, graph = (Worker(tiny_config().to_dict(), **{"weights.path": str(checkpoint), "batch.max_rows": 4})
                    for _ in range(2))
    yield eager, graph
    graph.close()
    eager.close()


def state(w: Worker) -> list[torch.Tensor]:
    """Every item the ledger can hand out: KV pages 1.., GDN slots 1.. (index 0 is padding)."""
    kv, gdn = w.service("state.kv.paged@1"), w.service("state.gdn.slots@1")
    return [*(kv.bind(layer).buffers()[name][kv.page_size:].clone() for layer in kv.k_buffer for name in ("k", "v")),
            gdn.conv[:, 1:].clone(), gdn.temporal[:, 1:].clone()]


def test_a_captured_step_pads_with_the_padding_row_and_equals_the_eager_step(pair: tuple[Worker, Worker]) -> None:
    """Rows fewer than the bucket, in another order, removed and added again, with state copies:
    the logits, the tokens and every request's state are the eager worker's, which ran the
    reference attention; the padded lanes wrote the padding items only."""
    eager, graph = pair
    replays = with_graphs(graph, [1, 2, 4])
    assert graph.service("plan@1").buckets("decode") == [1, 2, 4]
    assert all(torch.count_nonzero(t).item() == 0 for t in state(graph))  # recording touched no request item
    a, b = Driver(eager), Driver(graph)
    for d in (a, b):
        for row, prompt in ((0, [5, 17, 3]), (1, [9, 2, 8, 4, 7]), (2, [6, 1])):
            d.add(row, prompt)
    rounds = [{0: 3, 1: 5, 2: 2}, {2: 1, 0: 1, 1: 1}, {1: 1}, {0: 1, 1: 1}, {1: 1, 2: 1, 0: 1}, {2: 1}, {0: 1, 2: 1}]
    for i, chunks in enumerate(rounds):
        copies = ([StateCopy("gdn.slots", "after", Column.of("i32", [0]), Column.of("i32", [4]))] if i == 1 else
                  [StateCopy("gdn.slots", "before", Column.of("i32", [4]), Column.of("i32", [0]))] if i == 4 else [])
        x, y = a.step(chunks, copies), b.step(chunks, copies)
        torch.testing.assert_close(y.logits, x.logits, rtol=2e-4, atol=2e-4)
        assert y.finish() == x.finish()
        for u, v in zip(state(eager), state(graph), strict=True):
            torch.testing.assert_close(u, v, rtol=2e-4, atol=2e-4)
    assert replays == [4, 1, 2, 4, 1, 2]  # the prefill ran eagerly; each decode step in its smallest bucket
    for d in (a, b):  # an eager prefill after replays, then a padded step with the row's new items
        d.remove(1)
        d.add(1, [8, 3, 7, 6])
    for chunks in ({1: 4, 0: 1}, {1: 1, 0: 1, 2: 1}):
        x, y = a.step(chunks), b.step(chunks)
        torch.testing.assert_close(y.logits, x.logits, rtol=2e-4, atol=2e-4)
    assert replays[-1] == 4


def test_a_captured_step_keeps_states_for_the_prefix_cache_as_the_eager_one(pair: tuple[Worker, Worker]) -> None:
    """``StateTrack`` in decode steps that replay (the mask and destinations are fixed buffers of the
    bucket, the copy recorded with the step): the kept states are the eager worker's, and a bucket
    that kept states in its last replay keeps none in the next one, which tracks nothing."""
    eager, graph = pair
    replays = with_graphs(graph, [1, 2, 4])
    a, b = Driver(eager), Driver(graph)
    for d in (a, b):
        for row, prompt in ((0, [5, 17, 3]), (1, [9, 2, 8, 4, 7]), (2, [6, 1])):
            d.add(row, prompt)

    def track(rows: list[int], dst: list[int]) -> list[StateTrack]:
        return [StateTrack("gdn.slots", Column.of("i32", rows), Column.of("i32", dst), Column.of("i32", [0] * len(rows)))]

    rounds = [({0: 3, 1: 5, 2: 2}, []), ({1: 1}, track([1], [4])), ({2: 1}, []), ({0: 1, 2: 1}, track([2, 0], [5, 4])),
              ({2: 1, 0: 1}, []), ({0: 1, 1: 1, 2: 1}, track([1], [5]))]
    for chunks, tracked in rounds:
        x, y = a.step(chunks, track=tracked), b.step(chunks, track=tracked)
        torch.testing.assert_close(y.logits, x.logits, rtol=2e-4, atol=2e-4)
        for u, v in zip(state(eager), state(graph), strict=True):
            torch.testing.assert_close(u, v, rtol=2e-4, atol=2e-4)
    assert replays == [1, 1, 2, 2, 4]  # every decode step replayed


def test_graphs_whose_dependencies_moved_are_not_listed_and_required_refuses_the_step(pair: tuple[Worker, Worker]) -> None:
    _, graph = pair
    replays = with_graphs(graph, [1, 2])
    plans, batch = graph.service("plan@1"), graph.service("worker.batch@1")
    d = Driver(graph)
    d.add(0, [5, 17, 3])
    d.step({0: 3})
    d.step({0: 1})
    assert replays == [1]
    batch.generation += 1  # a table moved
    assert plans.buckets("decode") == []
    d.step({0: 1})  # auto: the step runs eagerly
    assert replays == [1]
    plans.required = True
    before = state(graph)
    delta, plan, ops = d.records({0: 1})
    with pytest.raises(StaleGraph, match="no state was changed"):
        graph.service("executor@1").submit(100, delta, plan, ops)
    assert all(torch.equal(x, y) for x, y in zip(before, state(graph), strict=True))
    plans.required = False


def test_idle_ep_replay_obeys_graph_dependencies_and_advances_placement_once() -> None:
    calls: list[str] = []
    model = SimpleNamespace(prepare_idle=lambda: calls.append("fetch"), stepped=lambda: calls.append("placement"))
    catalog = SimpleNamespace(version=1)
    plans = Plans(model, [])
    plans.idle_graph = SimpleNamespace(replay=lambda: calls.append("experts"))
    assert not plans.replay_idle()
    plans.deps = Dependencies([], [], catalog)
    assert plans.replay_idle() and calls == ["fetch", "experts", "placement"]
    catalog.version += 1
    assert not plans.replay_idle() and calls == ["fetch", "experts", "placement"]


def test_graph_plan_dirty_tail_zero_copy_and_mark_recorded(pair: tuple[Worker, Worker]) -> None:
    eager, graph = pair
    replays = with_graphs(graph, [4])
    d_ref = Driver(eager)
    d_g = Driver(graph)
    for d in (d_ref, d_g):
        for r, prompt in enumerate(([5, 17, 3], [9, 2, 11], [4, 8, 12])):
            d.add(r, prompt)
        d.step({0: 3, 1: 3, 2: 3})
    # Run 3 rows in bucket 4 (dirty_from becomes 3), then shrink to 1 row (only rows [1:3] need zeroing),
    # then grow back to 2 rows (verifying zero-copy fast path + padding isolation).
    for counts in ({0: 1, 1: 1, 2: 1}, {0: 1}, {0: 1, 2: 1}):
        s_ref = d_ref.step(counts)
        s_g = d_g.step(counts)
        torch.testing.assert_close(s_g.logits, s_ref.logits, rtol=2e-4, atol=2e-4)
    assert replays == [4, 4, 4]
    batch = graph.service("worker.batch@1")
    batch.rows[1].has_previous = False
    batch.mark_recorded([1])
    assert batch.rows[1].has_previous is True
