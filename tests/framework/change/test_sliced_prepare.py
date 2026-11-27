"""change.md §4.6 and §7 tests 14–15: prepare in slices while the old version serves."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import msgspec
import pytest

from core.change.coordinator import PlanInput
from core.change.protocol import Execute, Pause, Prepare
from core.change.types import ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import Identity, ProcId, UnitId
from core.common.result import Ok, Result
from core.ipc.rpc import Deferred
from core.supervisor.protocol import ProcSpec
from tests.framework.change.engine import ALL, SCHED, W1, Engine, engine, reload_everywhere, run, wait_until
from tests.framework.change.harness import Capture
from tests.framework.support import PROC, events, spec, write_tree
from unit.runtime import ProcessPlan, ReloadPlan
from unit.runtime.host import ProcessHost

def op_everywhere(args: dict[ProcId, dict[str, Any]]) -> Any:
    def script(inp: PlanInput) -> ActionPlan:
        return ActionPlan(procs=[
            ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op", args.get(p, {}))],
                        leads_epochs=p == SCHED)
            for p in ALL])

    return script


def has(e: Engine, p: ProcId, line: str) -> bool:
    return line in e.c.log_of(p)


def test_sliced_prepare_keeps_serving_until_everything_is_prepared(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            sliced = {"slices": 20, "slice_s": 0.05}
            e.planner.script = op_everywhere(dict.fromkeys(ALL, sliced))
            change = asyncio.ensure_future(e.coord.submit(ChangeSet(base=e.coord.identity)))
            await wait_until(lambda: has(e, W1, "op slice 2"))
            first = await e.status(W1)
            await wait_until(lambda: has(e, W1, "op slice 12"))
            later = await e.status(W1)
            assert (later.epoch or 0) > (first.epoch or 0) and not later.paused  # the old version serves
            assert not change.done()

            r = await change
            assert r.status == "committed", r.diagnostics
            for p in ALL:
                log = e.c.log_of(p)
                assert log.index("op slice 19") < log.index("op execute")  # paused only once prepared
            assert not [d for d in r.diagnostics if d.code == "change.prepare_one_shot"]

    run(main())


def test_a_failing_slice_aborts_the_change_without_leftovers(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            sliced = {"slices": 50, "slice_s": 0.02}
            e.planner.script = op_everywhere({**dict.fromkeys(ALL, sliced), W1: {**sliced, "fail_slice": 3}})
            r = await e.coord.submit(ChangeSet(base=old))
            assert r.status == "aborted"
            assert any("injected failure in slice 3" in d.reason for d in r.diagnostics)
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op abort"], p
                st = await e.status(p)
                assert st.identity == old and not st.paused and not st.diagnostics
                assert not any(u.unit not in ("toyc.loop", "toyc.tunable", "toyc.versioned") for u in st.units)
            sliced_so_far = {p: len(e.slices(p)) for p in ALL}
            await asyncio.sleep(0.3)
            assert {p: len(e.slices(p)) for p in ALL} == sliced_so_far  # nothing keeps preparing
            assert sliced_so_far[W1] == 4

            e.planner.script = reload_everywhere()
            assert (await e.coord.submit(e.code_change())).status == "committed"

    run(main())


def test_a_one_shot_prepare_reports_how_long_serving_stopped(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            e.planner.script = op_everywhere({W1: {"prepare_s": 0.5}})
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            stalls = {d.subject: d for d in r.diagnostics if d.code == "change.prepare_one_shot"}
            assert sorted(stalls) == sorted(str(p) for p in ALL)
            m = re.search(r"serving stopped for ([0-9.]+)s", stalls[str(W1)].reason)
            assert m is not None and float(m.group(1)) >= 0.5
            assert stalls[str(W1)].severity == "info"

    run(main())


SLOW_KERNEL = """
import time

from toyc.logutil import log


class Kernel:
    def __init__(self, ctx, config):
        time.sleep(0.5)  # an expensive constructor; the margin for the timing below
        log(config, ctx, f"built {ctx.id}")
"""


def test_new_units_are_staged_one_per_safe_point_while_the_old_version_serves(tmp_path: Path) -> None:
    """change.md §4.6: staging is a sliced prepare too. Each new unit is built at its own safe point,
    and the process keeps stepping and answering in between, instead of stopping until all are built.
    A request sent once unit i is built is answered at the next safe point, so before unit i + 2."""

    async def main() -> None:
        async with engine(tmp_path) as e:
            kernels = [spec(f"toyc.k{i}", "toyc.slow:Kernel", process=["worker"], reads=["toyc.log_dir"])
                       for i in range(5)]
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, stage=kernels)])
            files: dict[str, str | None] = {"engine/toyc/slow.py": e.c.blobs.put(SLOW_KERNEL.encode())}
            change = asyncio.ensure_future(e.coord.submit(ChangeSet(base=e.coord.identity, files=files)))
            replies = []
            for built, not_yet in (("built toyc.k0", "built toyc.k2"), ("built toyc.k2", "built toyc.k4")):
                await wait_until(lambda built=built: has(e, W1, built))
                replies.append(await e.status(W1))
                assert not has(e, W1, not_yet), f"answered only after {not_yet!r}: staging did not slice"
            assert not replies[0].paused and not replies[1].paused  # the old version serves
            assert (replies[1].epoch or 0) > (replies[0].epoch or 0)  # and it stepped in between
            r = await change
            assert r.status == "committed", r.diagnostics
            assert not [d for d in r.diagnostics if d.code == "change.prepare_one_shot"]
            assert {f"toyc.k{i}" for i in range(5)} <= {u.unit for u in (await e.status(W1)).units}

    run(main())


def in_process_host(plan: ProcessPlan, monkeypatch: pytest.MonkeyPatch) -> ProcessHost:
    """A process host without its main loop: ``deliver`` hands it messages and runs its safe points."""
    host = ProcessHost(ProcSpec(PROC, Identity("v1"), plan, "unused", "127.0.0.1"))
    monkeypatch.setattr(host.rpc, "defer", Capture)
    host.runtime.load_process_plan(plan).unwrap()
    return host


def deliver(host: ProcessHost, msg: msgspec.Struct, safe_points: int = 20) -> Result[msgspec.Struct]:
    result = host.handlers[type(msg)](host.registry.wrap(msg, host.me), msg)
    if isinstance(result, Capture):
        for _ in range(safe_points):  # back to back, as a main loop with instant steps would run them
            if result.result is not None:
                return result.result
            host.participant.tick()
        raise AssertionError("no reply within the safe points")
    assert not isinstance(result, Deferred)
    return result


CONSUMER = """
from tests.framework.support import events


class Consumer:
    def __init__(self, ctx, config):
        self.instance = ctx.instance
        self.provider = ctx.use("review.provider@1")
        self.ready = False
        ctx.on_action("work", self)

    def prepare(self, args):
        events.append(("prepare", str(self.instance)))
        self.ready = True

    def execute(self, args):
        events.append(("execute", str(self.instance)))
        assert self.ready, "execute ran on a new instance that never prepared"

    def abort(self):
        self.ready = False

    def commit(self):
        pass
"""
PROVIDER = """
class Provider:
    def __init__(self, ctx, config):
        ctx.provide("review.provider@1", self)
"""


def test_an_action_of_a_staged_unit_whose_provider_reloads_is_prepared_on_its_new_instance(
        tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A change stages B, which uses A, and reloads A: the reload rebuilds B with A. Which actions
    wait for the reload is decided once staging built B, so B's action is prepared on the instance
    that executes it, not on the staged one the reload replaces (found in review of #15)."""
    provider = spec(f"{pkg}.provider", f"{pkg}.provider:Provider", provides=["review.provider@1"])
    consumer = spec(f"{pkg}.consumer", f"{pkg}.consumer:Consumer", requires=["review.provider@1"])
    v1 = write_tree(tmp_path / "snapshots" / "v1" / "engine", {f"{pkg}/provider.py": PROVIDER})
    write_tree(tmp_path / "snapshots" / "v2" / "engine", {
        f"{pkg}/provider.py": PROVIDER + "\n# the new provider version\n", f"{pkg}/consumer.py": CONSUMER})
    host = in_process_host(ProcessPlan(PROC, Identity("v1"), [provider], code_root=str(v1)), monkeypatch)

    def send(msg: msgspec.Struct) -> Result[msgspec.Struct]:
        return deliver(host, msg)

    try:
        actions = ProcActions(PROC, stage=[consumer], actions=[UnitAction(consumer.id, "work")],
                              reload=ReloadPlan(changed_modules=[f"{pkg}.provider"], units_to_reload=[provider]))
        assert isinstance(send(Prepare(1, Identity("v2"), actions, 30)), Ok)
        assert isinstance(send(Pause(1, None)), Ok)
        result = send(Execute(1))
        assert isinstance(result, Ok), result
        prepared = [inst for kind, inst in events if kind == "prepare"]
        executed = [inst for kind, inst in events if kind == "execute"]
        assert prepared and executed and prepared[-1] == executed[-1], events
    finally:
        host.shutdown()


SLOW_SLICES = """
import time

from tests.framework.support import events


class Unit:
    def __init__(self, ctx, config):
        ctx.on_action("work", self)

    def prepare(self, args):
        for i in range(2):
            events.append(("slice", i))
            time.sleep(0.4)  # a slice longer than the coordinator's whole lease
            yield

    def execute(self, args):
        pass

    def abort(self):
        pass

    def commit(self):
        pass
"""


def test_a_slice_longer_than_the_lease_survives_when_keepalives_arrive(tmp_path: Path) -> None:
    """While a slice runs, the coordinator's keepalives wait in the socket until the next safe point.
    Checking the lease before advancing the next slice lets rpc.poll() drain those keepalives first,
    so a slice longer than the lease does not make the participant think the coordinator is gone."""

    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=0.3, keepalive_interval_s=0.1) as e:
            e.planner.script = op_everywhere(dict.fromkeys(ALL, {"slices": 2, "slice_s": 0.4}))
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            for p in ALL:
                st = await e.status(p)
                assert not [d for d in st.diagnostics if d.code == "change.coordinator_lost"], st.diagnostics
                assert len(e.slices(p)) == 2

    run(main())


def test_a_slice_longer_than_the_lease_withdraws_when_no_keepalive_arrives(
        tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slice work does not extend the lease indefinitely: if no message from the coordinator
    arrives during a slow slice, the next safe point withdraws instead of starting another slice."""
    unit = spec(f"{pkg}.slow", f"{pkg}.slow:Unit")
    v1 = write_tree(tmp_path / "snapshots" / "v1" / "engine", {f"{pkg}/slow.py": SLOW_SLICES})
    host = in_process_host(ProcessPlan(PROC, Identity("v1"), [unit], code_root=str(v1)), monkeypatch)
    try:
        r = deliver(host, Prepare(1, Identity("v1"), ProcActions(PROC, actions=[UnitAction(unit.id, "work")]), 0.3))
        assert not isinstance(r, Ok), r
        assert events == [("slice", 0)], events
        assert [d for d in host.participant.diagnostics if d.code == "change.coordinator_lost"]
    finally:
        host.shutdown()
