"""A unit's build step (``[unit] build``, change.md §4.6): run while the engine serves, before
anything pauses, in a process of its own from the change's code; its failure fails the change
before the pause."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import msgspec
import pytest

from core.change.coordinator import PlanInput
from core.change.types import ActionPlan, ProcActions
from tests.framework.change.engine import ALL, SCHED, W1, engine, run, wait_until
from tests.framework.cluster.harness import ALL_ROLES
from tests.framework.support import spec
from unit.runtime import ReloadPlan

BUILT = spec("toyc.versioned", "toyc.versioned:Versioned", process=ALL_ROLES, reads=["toyc.log_dir"],
             build="toyc.builder:make")


def reload_with_build(inp: PlanInput) -> ActionPlan:
    return ActionPlan(procs=[ProcActions(p, reload=ReloadPlan(changed_modules=["toyc.versioned"],
                                                              units_to_reload=[BUILT]), leads_epochs=p == SCHED)
                             for p in ALL])


def builds(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_a_build_step_runs_from_the_new_code_while_the_old_version_serves(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))
    monkeypatch.setenv("TOYC_BUILD_S", "1.5")

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            e.planner.script = reload_with_build
            change = asyncio.ensure_future(e.coord.submit(e.code_change()))
            await asyncio.sleep(0.5)  # the steps run (1.5 s each, all processes at once)
            assert not change.done() and not builds(log)
            for p in ALL:  # the old version serves meanwhile, nothing is paused
                st = await e.status(p)
                assert not st.paused and e.versioned(p) == "versioned v1"
                assert await e.epochs_advance(p)
            r = await change
            assert r.status == "committed", r.diagnostics
            assert r.new_identity is not None
            lines = builds(log)
            assert len(lines) == len(ALL), lines  # one per process that reloads the unit
            assert all(r.new_identity in line for line in lines), lines  # from the change's snapshot
            for p in ALL:
                assert e.versioned(p) == "versioned v2"
                assert any("build step toyc.builder:make done" in line for line in e.c.output_of(p)), p
            assert r.timings["prepare"] >= 1.5

    run(main())


def test_a_failing_build_step_fails_the_change_before_anything_paused(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))
    monkeypatch.setenv("TOYC_BUILD_FAIL", "1")

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            old = e.coord.identity
            e.planner.script = reload_with_build
            r = await e.coord.submit(e.code_change())
            assert r.status == "aborted", r
            codes = {d.code for d in r.diagnostics}
            assert "change.build_failed" in codes, r.diagnostics
            reason = next(d.reason for d in r.diagnostics if d.code == "change.build_failed")
            assert "toyc.versioned" in reason and "exited with 3" in reason, reason
            assert "pause" not in r.timings, r.timings  # it never got that far
            assert e.coord.identity == old
            for p in ALL:
                st = await e.status(p)
                assert st.identity == old and not st.paused and e.versioned(p) == "versioned v1"
                assert await e.epochs_advance(p)
            assert builds(log)  # the steps did run

    run(main())


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def test_an_undone_build_step_leaves_no_compiler_and_no_lock_and_the_source_builds_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The change fails while the step builds (here: past ``change.prepare_timeout_s``). Before,
    the undo SIGKILLed the step's process alone: the compilers it started went on, and its lock on
    the build directory stayed, so every later build of the source waited on it (Codex's review of
    #55). Now the step's whole session gets SIGTERM, its ``finally`` deletes the lock, and what does
    not stop on SIGTERM is killed: the session, not the step's process group, since ninja runs each
    compiler in a group of its own (seen on the 4090 D: an ``nvcc`` went on after the undo)."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "compile_s").write_text("60")
    monkeypatch.setenv("TOYC_BUILD_LOG", str(tmp_path / "builds.log"))
    monkeypatch.setenv("TOYC_BUILD_WORK", str(work))

    def compilers() -> list[int]:
        return [int(x) for x in (work / "pids").read_text().split()] if (work / "pids").exists() else []

    async def main() -> None:
        async with engine(tmp_path / "engine", prepare_timeout_s=4.0) as e:
            old = e.coord.identity
            e.planner.script = reload_with_build
            r = await e.coord.submit(e.code_change())
            assert r.status == "aborted" and e.coord.identity == old, r
            started = compilers()
            assert len(started) == 2, started  # one step compiled (ninja, nvcc), the others waited on the lock
            deadline = time.monotonic() + 10
            while any(alive(p) for p in started) and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            assert not [p for p in started if alive(p)]
            assert not (work / "lock").exists()
            for p in ALL:
                st = await e.status(p)
                assert st.identity == old and not st.paused and e.versioned(p) == "versioned v1"

            (work / "compile_s").write_text("0.2")
            e.coord.tun = msgspec.structs.replace(e.coord.tun, prepare_timeout_s=30.0)
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert all(e.versioned(p) == "versioned v2" for p in ALL)

    run(main())


@pytest.mark.parametrize("parent_still_visible", [False, True], ids=["host-killed", "term-before-reparenting"])
def test_a_build_step_whose_process_was_killed_takes_its_compilers_with_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_still_visible: bool) -> None:
    """SIGKILL to the process whose step compiles: nobody is left to stop the step's session from
    outside. Before, the step got SIGTERM from the kernel, let its lock go and killed the ``ninja``
    it waited for, and that was all: the compiler ``ninja`` had started went on, beside the next
    build of the same source (Codex's second review of #55). Now a step whose process is gone ends
    the rest of its session itself."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "compile_s").write_text("60")
    monkeypatch.setenv("TOYC_BUILD_LOG", str(tmp_path / "builds.log"))
    monkeypatch.setenv("TOYC_BUILD_WORK", str(work))

    def compilers() -> list[int]:
        return [int(x) for x in (work / "pids").read_text().split()] if (work / "pids").exists() else []

    async def main() -> None:
        async with engine(tmp_path / "engine", procs={"change.build_stop_grace_s": 1.0}, prepare_timeout_s=6.0) as e:
            e.planner.script = reload_with_build
            change = asyncio.ensure_future(e.coord.submit(e.code_change()))
            await wait_until(lambda: len(compilers()) == 2)
            its = compilers()  # ninja and nvcc of the one step that compiles; the others wait on the lock
            host = int((work / "host").read_text())
            assert host in [e.c.sup.local_pid(p) for p in ALL]
            if parent_still_visible:
                # Deliver the parent-death signal to the build leader while getppid() still returns
                # the host PID, as can happen while a multithreaded host is exiting on Linux.
                os.kill(os.getsid(its[0]), signal.SIGTERM)
            else:
                os.kill(host, signal.SIGKILL)
            await wait_until(lambda: not any(alive(p) for p in its), timeout=10)  # a 1 s grace: they ignore SIGTERM
            r = await change
            assert r.status != "committed", r
            # the other processes' steps, which got the lock meanwhile, were undone as before
            await wait_until(lambda: not any(alive(p) for p in compilers()), timeout=10)
            assert not (work / "lock").exists()

    run(main())


def test_a_change_without_build_steps_starts_no_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other units' reloads, and every unit's load at a process's start, run no step."""
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            plain = msgspec.structs.replace(BUILT, manifest=msgspec.structs.replace(
                BUILT.manifest, unit=msgspec.structs.replace(BUILT.manifest.unit, build="")))

            def script(inp: PlanInput) -> ActionPlan:
                return ActionPlan(procs=[ProcActions(p, reload=ReloadPlan(changed_modules=["toyc.versioned"],
                                                                          units_to_reload=[plain]),
                                                     leads_epochs=p == SCHED) for p in ALL])

            e.planner.script = script
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert not builds(log)
            assert e.versioned(W1) == "versioned v2"

    run(main())


def test_a_bad_build_entry_is_refused_by_the_manifest_check() -> None:
    from core.common.manifest import parse_manifest
    from core.common.result import Failed
    from tests.framework.support import manifest_toml

    bad = parse_manifest(manifest_toml("toyc.x", "toyc.x:X", build="not an entry"))
    assert isinstance(bad, Failed) and "bad build" in str(bad.diagnostics), bad
