"""The change coordinator (doc/specs/change.md §4): every change goes through it.

RECEIVED → CHECKED → PLANNED → PREPARING → PREPARED → PAUSING → EXECUTING → COMMITTED,
and from any of them to ABORTING → ABORTED. One change at a time; the rest wait in
the queue. It does not decide what to touch (``core.deps`` plans) and executes
nothing itself (participants do); it sequences phases and tells the truth about
the outcome.

Participants that voted ``Prepared`` cannot decide alone (§4.7), so the
coordinator renews their lease with ``KeepAlive`` while a change runs, writes
its decision to the event log before sending ``Commit``/``Abort``, answers
``QueryDecision`` from the log, and retries Commit/Resume within bounded limits.
No participant resumes before every Commit is confirmed. A missing acknowledgement
requires engine recovery, even when the failure detector reports a process lost.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Protocol

import msgspec

from core.change.config import ChangeTunables
from core.change.eventlog import EventLog
from core.change.faults import Faults
from core.change.protocol import (
    Abort,
    Commit,
    Committed,
    Decision,
    DecisionKind,
    Drain,
    Drained,
    Execute,
    Executed,
    KeepAlive,
    Pause,
    Paused,
    Prepare,
    Prepared,
    QueryDecision,
    Resume,
    Resumed,
)
from core.change.queue import ChangeQueue, Queued
from core.change.state import IdentityFn, IdentityManifest, apply_changeset, identity_of
from core.change.types import EMPTY, ActionPlan, ChangeResult, ChangeSet, ChangeStatus, ProcActions
from core.common.ids import HostId, Identity, ProcId
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail
from core.supervisor.supervisor import ProcStatus, Supervisor
from core.supervisor.topology import Topology
from unit.runtime.types import ReloadReport

COMPONENT = "change"


class PlanInput(msgspec.Struct, frozen=True):
    cs: ChangeSet
    base_identity: Identity
    base: IdentityManifest
    new: IdentityManifest
    procs: list[ProcStatus]


class Checker(Protocol):
    """``core.manifest``: static checks; nothing runs if they fail."""

    def check(self, inp: PlanInput) -> Awaitable[Result[None]]: ...


class Planner(Protocol):
    """``core.deps``: what each process has to do."""

    def plan(self, inp: PlanInput) -> Awaitable[Result[ActionPlan]]: ...


class LocalParticipant(Protocol):
    """The control-plane process itself, for ``ActionPlan.control`` (its control-plane units).
    The coordinator runs one change at a time, so between two of its steps the control
    process is at a safe point; there is nothing to pause."""

    def prepare(self, change_id: int, identity: Identity, actions: ProcActions,
                files: dict[str, str]) -> Awaitable[Result[None]]: ...

    def execute(self, change_id: int) -> Result[None]: ...

    def commit(self, change_id: int) -> None: ...

    def abort(self, change_id: int) -> list[Diagnostic]:
        """Undo the change here; an error among the diagnostics means the undo did not finish."""
        ...


class AcceptAll:
    async def check(self, inp: PlanInput) -> Result[None]:
        return Ok(None)


class _Abort(Exception):
    def __init__(self, diags: list[Diagnostic]) -> None:
        super().__init__("; ".join(str(d) for d in diags))
        self.diags = diags


def _blobs_of(cs: ChangeSet) -> set[str]:
    """Blob hashes a change-set refers to: file contents and artifacts."""
    return {*(h for h in cs.files.values() if h is not None), *cs.artifacts.values()}


class _Vanish(Exception):
    """Injected: the coordinator goes silent in the middle of a change."""


@dataclass
class _Run:
    """Book-keeping of one change in flight, for the abort path."""

    id: int
    cs: ChangeSet
    plan: ActionPlan
    identity: Identity
    asked: list[ProcActions] = field(default_factory=list[ProcActions])
    """Participants sent ``Prepare``: they get ``Abort`` if the change fails."""
    prepared: list[ProcActions] = field(default_factory=list[ProcActions])
    restarted: list[ProcId] = field(default_factory=list[ProcId])
    spawned: list[ProcId] = field(default_factory=list[ProcId])
    stopped: list[ProcId] = field(default_factory=list[ProcId])
    """Stopped by the plan; the supervisor keeps their specs until commit, so abort can restart them."""
    old_topology: Topology | None = None
    """The topology when the change started; abort puts it back."""
    timings: dict[str, float] = field(default_factory=dict[str, float])
    reports: dict[str, ReloadReport] = field(default_factory=dict[str, ReloadReport])
    diags: list[Diagnostic] = field(default_factory=list[Diagnostic])
    """Reported by participants along the way (e.g. one-shot prepares)."""
    drain_outcomes: dict[str, int] = field(default_factory=dict[str, int])
    executing: bool = False
    """``Execute`` went out: a participant may still be in it, or undo a reload, when ``Abort`` comes."""
    _t: float = field(default_factory=time.monotonic)

    def lap(self, phase: str) -> None:
        now = time.monotonic()
        self.timings[phase] = now - self._t
        self._t = now


class Coordinator:
    def __init__(
        self, supervisor: Supervisor, log: EventLog, *, planner: Planner, checker: Checker | None = None,
        identity_fn: IdentityFn = identity_of, tun: ChangeTunables | None = None,
        local: LocalParticipant | None = None,
    ) -> None:
        self.sup = supervisor
        self.log = log
        self.planner = planner
        self.checker = checker or AcceptAll()
        self.identity_fn = identity_fn
        self.local = local
        self.tun = tun or ChangeTunables()
        if self.tun.commit_attempts < 1 or self.tun.commit_retry_interval_s < 0 \
                or not 0 < self.tun.commit_ack_timeout_s < float("inf"):
            raise ValueError("commit limits must be positive and finite; retry interval must be nonnegative")
        if self.tun.blob_prune_batch < 1 or self.tun.blob_prune_interval_s < 0:
            raise ValueError("change.blob_prune_batch must be positive, change.blob_prune_interval_s nonnegative")
        self.recovery_required = False
        self.identity: Identity = EMPTY
        self.manifest = IdentityManifest()
        self.state_version = 0
        self.queue = ChangeQueue()
        self._prune_due_at = 0.0
        self.faults = Faults()
        """Fault points for tests (``core.change.faults.POINTS``)."""
        self.busy = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._running: _Run | None = None
        self._background: set[asyncio.Task[Any]] = set()
        self.on_result: list[Callable[[ChangeResult], None]] = []

    # ------------------------------------------------------------ queue

    async def start(self) -> None:
        self.sup.handlers[QueryDecision] = self._on_query
        self._task = asyncio.create_task(self._worker())

    def close(self) -> None:
        self.sup.handlers.pop(QueryDecision, None)
        for t in [*self._background, *([self._task] if self._task else [])]:
            t.cancel()

    def enqueue(self, cs: ChangeSet, *, dry_run: bool = False) -> asyncio.Future[ChangeResult]:
        fut: asyncio.Future[ChangeResult] = asyncio.get_running_loop().create_future()
        invalid = self._invalid(cs)
        if invalid:  # rejected as soon as it is received (§6)
            cid = self.log.begin(cs)
            self._deliver(fut, self._done(cid, self._result("failed", cid, cs, None, None, invalid)))
            return fut
        self.queue.push(Queued(cs, dry_run, data={"future": fut}))
        self._wake.set()
        return fut

    def _invalid(self, cs: ChangeSet) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        if cs.source == "internal" and cs.computed_under is None:
            out.append(diag(COMPONENT, "receive", "an internal change must say which identity it was computed under",
                            code="change.missing_computed_under", hint="set computed_under"))
        if cs.state_only:
            touched = [name for name, changed in [
                ("files", bool(cs.files)), ("config", bool(cs.config)), ("deps", bool(cs.deps)),
                ("topology", cs.topology is not None), ("artifacts", bool(cs.artifacts))] if changed]
            if touched:
                out.append(diag(COMPONENT, "receive",
                                f"a state-only change cannot change {touched}: the identity would not say what runs",
                                code="change.state_only_with_content",
                                hint="submit it as a change that produces a new identity"))
        return out

    def _deliver(self, fut: object, result: ChangeResult) -> None:
        assert isinstance(fut, asyncio.Future)
        if not fut.done():
            fut.set_result(result)  # type: ignore[union-attr]
        for cb in self.on_result:
            cb(result)

    async def submit(self, cs: ChangeSet, *, dry_run: bool = False) -> ChangeResult:
        return await self.enqueue(cs, dry_run=dry_run)

    async def _worker(self) -> None:
        while True:
            item, dropped = self.queue.pop(self.identity)
            for d in dropped:
                self._finish_dropped(d)
            if item is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            self.busy = True
            try:
                result = await self._run(item)
            except Exception as e:  # noqa: BLE001 - the coordinator must outlive any change
                result = self._result("failed", -1, item.cs, None, None,
                                      [diag(COMPONENT, "internal", f"{type(e).__name__}: {e}",
                                            code="change.internal_error")])
            finally:
                self.busy = False
            self._deliver(item.data["future"], result)
            try:
                await self.prune_if_due()
            except Exception as e:  # noqa: BLE001 - retention must not stop the queue
                self.log.event(None, "prune_failed", {"error": f"{type(e).__name__}: {e}"})

    def _finish_dropped(self, item: Queued) -> None:
        cid = self.log.begin(item.cs)
        result = self._result("dropped", cid, item.cs, None, None, [diag(
            COMPONENT, "queue", f"computed under {item.cs.computed_under}, current is {self.identity}",
            code="change.stale_internal", severity="info", hint="the unit recomputes it next cycle")])
        self.log.finish(result)
        self.log.event(cid, "dropped_stale_internal", {"computed_under": item.cs.computed_under})
        self._deliver(item.data["future"], result)

    def _result(self, status: ChangeStatus, cid: int, cs: ChangeSet, new: Identity | None,
                run: _Run | None, diags: list[Diagnostic], plan: ActionPlan | None = None,
                old: Identity | None = None) -> ChangeResult:
        """``old``: the running identity before the change (internal changes do not check ``cs.base``)."""
        return ChangeResult(
            status=status, change_id=cid, old_identity=old if old is not None else self.identity,
            new_identity=new, plan=run.plan if run else plan, timings=dict(run.timings) if run else {},
            drain_outcomes=dict(run.drain_outcomes) if run else {},
            restarted=list(run.restarted) if run else [], reports=dict(run.reports) if run else {},
            state_version=self.state_version, diagnostics=diags,
        )

    def _fault(self, point: str) -> None:
        if self.faults.take(point):
            raise _Abort([diag(COMPONENT, point, f"fault injected at {point}", code="change.injected_fault")])

    def _spawn(self, coro: Coroutine[Any, Any, object]) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # ------------------------------------------------------- decisions (§4.7)

    def decision_of(self, change_id: int) -> DecisionKind:
        decided = self.log.decision(change_id)
        if decided is not None:
            return decided
        if self._running is not None and self._running.id == change_id:
            return "pending"
        return "abort" if self.log.known(change_id) else "unknown"  # never decided commit: presumed abort

    async def _on_query(self, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        assert isinstance(msg, QueryDecision)
        if self.faults.armed("no_decisions") or self.faults.silent(msg.change_id):
            return fail(COMPONENT, "query", "the coordinator does not answer (injected)", code="change.injected_fault")
        return Ok(Decision(msg.change_id, self.decision_of(msg.change_id)))

    async def _keepalive(self, run: _Run) -> None:
        """Renew every participant's lease, also through long phases such as restarts."""
        procs = [pa.proc for pa in run.plan.participants()]
        interval = self.tun.keepalive_interval_s
        while True:
            await asyncio.sleep(interval)
            if self.faults.silent(run.id) or self.faults.quiet:
                continue
            for p in procs:
                self._spawn(self._call(p, KeepAlive(run.id), timeout=interval))

    def _vanish(self, run: _Run, phase: str) -> ChangeResult:
        self.faults.silence(run.id)
        return self._done(run.id, self._result("aborted", run.id, run.cs, None, run, [diag(
            COMPONENT, phase, "coordinator vanished (injected)", code="change.injected_fault")]))

    # ------------------------------------------------------------- one change

    async def _run(self, item: Queued) -> ChangeResult:
        cid = self.log.begin(item.cs)
        try:
            return await self._run_begun(item, cid)
        except Exception as e:  # noqa: BLE001 - a begun change always ends with a durable outcome
            diags = [diag(COMPONENT, "internal", f"{type(e).__name__}: {e}", code="change.internal_error")]
            if self.recovery_required:
                diags.append(diag(COMPONENT, "internal", "the change failed inside its transaction; restart the "
                                  "engine", code="change.recovery_required"))
            return self._done(cid, self._result("failed", cid, item.cs, None, None, diags))

    async def _run_begun(self, item: Queued, cid: int) -> ChangeResult:
        cs = item.cs
        old = self.identity
        t0 = time.monotonic()
        if self.recovery_required:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", "a previous change is unresolved; restart the engine",
                code="change.recovery_required")]))
        # RECEIVED -> CHECKED
        if cs.source != "internal" and cs.base != old:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", f"base {cs.base!r} is not the running identity {self.identity!r}",
                code="change.stale_base", hint="rebase the change on the current identity")]))
        new_manifest = apply_changeset(self.manifest, cs)
        missing = self.sup.blobs.missing(_blobs_of(cs))
        if missing:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", f"{len(missing)} file or artifact contents are not in the blob store",
                code="change.missing_blobs", hint="upload them before submitting")]))
        inp = PlanInput(cs, self.identity, self.manifest, new_manifest, self.sup.status())
        try:
            checked = await self.checker.check(inp)
        except Exception as e:  # noqa: BLE001 - the checks come from a unit (core.manifest)
            checked = fail(COMPONENT, "check", f"the static checks raised {type(e).__name__}: {e}",
                           code="change.check_crashed")
        if isinstance(checked, Failed):
            return self._done(cid, self._result("failed", cid, cs, None, None, checked.diagnostics))
        # CHECKED -> PLANNED
        try:
            planned = await self.planner.plan(inp)
        except Exception as e:  # noqa: BLE001 - the plan comes from a unit (core.deps)
            planned = fail(COMPONENT, "plan", f"the impact analysis raised {type(e).__name__}: {e}",
                           code="change.plan_crashed")
        if isinstance(planned, Failed):
            return self._done(cid, self._result("failed", cid, cs, None, None, planned.diagnostics))
        plan = planned.value
        notes = [*checked.diagnostics, *planned.diagnostics]  # warnings found on the way
        try:
            new_identity = old if cs.state_only else self.identity_fn(new_manifest)
        except Exception as e:  # noqa: BLE001 - the identity comes from a unit (core.identity)
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "identity", f"cannot compute the identity: {type(e).__name__}: {e}",
                code="change.no_identity")], plan=plan))
        if item.dry_run:
            return self._done(cid, self._result("planned", cid, cs, new_identity, None, notes, plan=plan))
        if plan.restart_engine:
            why = "; ".join(plan.notes) or "the change touches the framework core"
            return self._done(cid, self._result("failed", cid, cs, None, None, [*notes, diag(
                COMPONENT, "plan", f"the engine must restart: {why}", code="escalate.restart_engine",
                hint="restart the engine from the new source")], plan=plan))
        run = _Run(cid, cs, plan, new_identity, old_topology=self.sup.topology)
        run.diags += notes
        run.timings["check+plan"] = time.monotonic() - t0
        self.log.record_identity(new_identity, new_manifest, old if old != EMPTY else None)
        self._running = run
        keepalive = asyncio.create_task(self._keepalive(run))
        try:
            return await self._transact(run, old, new_manifest)
        except Exception:
            self.recovery_required = True  # unexpected inside the transaction: participants may hold part of it
            raise
        finally:
            keepalive.cancel()
            self._running = None

    async def _transact(self, run: _Run, old: Identity, new_manifest: IdentityManifest) -> ChangeResult:
        cid, cs = run.id, run.cs
        try:
            self._fault("after_check")
            await self._prepare(run, new_manifest)
            if self.faults.take("vanish_after_prepare"):
                raise _Vanish("prepare")
            self._fault("after_prepare")
            await self._pause(run)
            self._fault("after_pause")
            await self._execute(run, new_manifest)
            self._fault("after_execute")
        except _Abort as e:
            undo = await self._abort(run)
            status = "failed" if self.recovery_required else "aborted"
            return self._done(cid, self._result(status, cid, cs, None, run, [*run.diags, *e.diags, *undo], old=old))
        except _Vanish as e:
            return self._vanish(run, str(e))
        status, diags = await self._commit(run)
        # The decision is durable and the other participants serve the new identity,
        # so it is current even when a participant could not be brought along.
        self.identity = run.identity
        self.manifest = new_manifest
        if cs.state_only:
            self.state_version += 1
        return self._done(cid, self._result(status, cid, cs, run.identity, run, [*run.diags, *diags], old=old))

    def _done(self, cid: int, result: ChangeResult) -> ChangeResult:
        self.log.finish(result)
        return result

    async def _call(self, proc: ProcId, msg: msgspec.Struct, timeout: float | None = None
                    ) -> Result[msgspec.Struct]:
        return await self.sup.call(proc, msg, timeout=timeout or self.sup.tun.rpc_timeout_s)

    async def _gather(self, calls: list[tuple[ProcId, Awaitable[Result[msgspec.Struct]]]]
                      ) -> dict[ProcId, msgspec.Struct]:
        results = await asyncio.gather(*(c for _, c in calls))
        errors = [d for r in results if isinstance(r, Failed) for d in r.diagnostics]
        if errors:
            raise _Abort(errors)
        return {p: r.unwrap() for (p, _), r in zip(calls, results, strict=True)}

    def _hosts(self, run: _Run) -> set[HostId]:
        hosts = {p.proc.host for p in run.plan.procs} | {p.proc.host for p in run.plan.spawn}
        return hosts

    # ---------------------------------------------------------------- phases

    async def _prepare(self, run: _Run, manifest: IdentityManifest) -> None:
        hosts = sorted(self._hosts(run))
        mats = await asyncio.gather(*(self.sup.materialize(h, run.identity, manifest.files) for h in hosts))
        errors = [d for m in mats if isinstance(m, Failed) for d in m.diagnostics]
        if errors:
            raise _Abort(errors)
        participants = run.plan.participants()
        run.asked = list(participants)
        calls = [self._call(pa.proc, Prepare(run.id, run.identity, pa, self.tun.coordinator_timeout_s),
                            timeout=self.tun.prepare_timeout_s) for pa in participants]
        if self.faults.take("vanish_during_prepare"):
            self.faults.silence(run.id)
            await self.faults.send_then_vanish(calls)
            raise _Vanish("prepare")
        results = await asyncio.gather(*calls)
        for pa, r in zip(participants, results, strict=True):
            if isinstance(r, Ok):
                run.prepared.append(pa)
                if isinstance(r.value, Prepared):
                    run.diags += r.value.diagnostics
        errors = [d for r in results if isinstance(r, Failed) for d in r.diagnostics]
        if errors:
            raise _Abort(errors)
        if run.plan.control is not None:
            if self.local is None:
                raise _Abort([diag(COMPONENT, "prepare", "the plan changes the control plane, which has no "
                                   "local participant here", code="change.no_control_plane")])
            local = await self.local.prepare(run.id, run.identity, run.plan.control, manifest.files)
            if isinstance(local, Failed):
                raise _Abort(local.diagnostics)
        run.lap("prepare")

    async def _pause(self, run: _Run) -> None:
        drain_timeout = run.plan.drain_timeout_s if run.plan.drain_timeout_s is not None else self.tun.drain_timeout_s
        draining = [pa for pa in run.prepared if pa.drain]
        if draining:
            drained = await self._gather([
                (pa.proc, self._call(pa.proc, Drain(run.id, drain_timeout),
                                     timeout=drain_timeout + self.sup.tun.rpc_timeout_s))
                for pa in draining])
            for d in drained.values():
                if isinstance(d, Drained):
                    for outcome, n in d.outcomes.items():
                        run.drain_outcomes[outcome] = run.drain_outcomes.get(outcome, 0) + n
            run.lap("drain")
        leaders = [pa for pa in run.prepared if pa.leads_epochs]
        followers = [pa for pa in run.prepared if not pa.leads_epochs]
        paused = await self._gather([(pa.proc, self._call(pa.proc, Pause(run.id, None))) for pa in leaders])
        epochs = [p.epoch for p in paused.values() if isinstance(p, Paused) and p.epoch is not None]
        after = max(epochs) if epochs else None
        await self._gather([(pa.proc, self._call(pa.proc, Pause(run.id, after))) for pa in followers])
        run.lap("pause")

    async def _execute(self, run: _Run, manifest: IdentityManifest) -> None:
        run.executing = True
        executed = await self._gather([(pa.proc, self._call(pa.proc, Execute(run.id), timeout=self.tun.execute_timeout_s))
                                       for pa in run.prepared])
        for proc, e in executed.items():
            if isinstance(e, Executed):
                run.diags += e.diagnostics
                if e.report is not None:
                    run.reports[str(proc)] = e.report
        topo = run.plan.topology or self.sup.topology
        for proc in run.plan.stop:
            r = await self.sup.stop(proc)
            if isinstance(r, Failed):
                self.recovery_required = True
                raise _Abort(r.diagnostics)
            run.stopped.append(proc)
        # The processes of one role start together, role after role: units of ranks that compute
        # together wait for each other while they load (a communication group's rendezvous).
        restarts = [pa.restart for pa in run.plan.procs if pa.restart is not None]
        for _, plans in groupby(restarts, key=lambda plan: plan.proc.role):
            assert topo is not None
            together = list(plans)
            run.restarted += [plan.proc for plan in together]
            await self._started(self.sup.relaunch(plan, run.identity, topo, start_paused=True) for plan in together)
        for _, plans in groupby(run.plan.spawn, key=lambda plan: plan.proc.role):
            assert topo is not None
            together = list(plans)
            run.spawned += [plan.proc for plan in together]
            await self._started(self.sup.launch(plan, run.identity, topo, start_paused=True) for plan in together)
        if run.plan.topology is not None:
            self.sup.topology = run.plan.topology
        if run.plan.control is not None and self.local is not None:
            local = self.local.execute(run.id)
            if isinstance(local, Failed):
                raise _Abort(local.diagnostics)
        run.lap("execute")

    @staticmethod
    async def _started(starts: Iterable[Awaitable[Result[None]]]) -> None:
        failed = [d for r in await asyncio.gather(*starts) if isinstance(r, Failed) for d in r.diagnostics]
        if failed:
            raise _Abort(failed)

    async def _confirm(self, run: _Run, procs: list[ProcId], *, resume: bool,
                       deadline: float) -> tuple[list[Diagnostic], dict[ProcId, Resumed]]:
        """Commit (or Resume) until every process confirms, within the attempts and the deadline.
        Returns what is still unconfirmed, and the ``Resumed`` acknowledgements."""
        pending = list(procs)
        phase = "resume" if resume else "commit"
        resumed: dict[ProcId, Resumed] = {}

        async def call(proc: ProcId, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return fail(COMPONENT, phase, "confirmation deadline expired", code="change.commit_ack_timeout")
            try:
                return await asyncio.wait_for(
                    self._call(proc, msg, timeout=min(remaining, self.sup.tun.rpc_timeout_s)), remaining)
            except TimeoutError:
                return fail(COMPONENT, phase, "confirmation deadline expired", code="change.commit_ack_timeout")

        for attempt in range(self.tun.commit_attempts):
            if not pending:
                return [], resumed
            if time.monotonic() >= deadline:
                break
            msg = Resume(run.id, run.identity) if resume else Commit(run.id)
            results = await asyncio.gather(*(call(p, msg) for p in pending))
            retry: list[ProcId] = []
            for p, r in zip(pending, results, strict=True):
                ack = r.value if isinstance(r, Ok) else None
                valid = (isinstance(ack, Resumed) and not ack.rolled_back) if resume else isinstance(ack, Committed)
                if not valid or not isinstance(ack, (Committed, Resumed)) \
                        or ack.change_id != run.id or ack.identity != run.identity:
                    retry.append(p)
                    self.log.event(run.id, f"{phase}_unacknowledged", {"proc": str(p), "attempt": attempt + 1})
                elif isinstance(ack, Resumed):
                    resumed[p] = ack
            pending = retry
            if pending and attempt + 1 < self.tun.commit_attempts:
                await asyncio.sleep(min(self.tun.commit_retry_interval_s, max(0.0, deadline - time.monotonic())))
        return [diag(COMPONENT, phase,
                     f"{phase} not confirmed for committed identity {run.identity}; restart the engine",
                     subject=str(p), code="change.recovery_required") for p in pending], resumed

    async def _commit(self, run: _Run) -> tuple[ChangeStatus, list[Diagnostic]]:
        self.recovery_required = True
        self.log.decide(run.id, "commit")  # QueryDecision must see the same durable decision.
        if run.plan.control is not None and self.local is not None:
            self.local.commit(run.id)
        self.log.event(run.id, "commit_decided", {"identity": run.identity,
                       "participants": [str(pa.proc) for pa in run.prepared]})
        if self.faults.take("delay_commit"):
            await self.faults.hold(2 * self.tun.coordinator_timeout_s)
        for proc in run.stopped:  # decided: they are not coming back
            self.sup.forget(proc)
        deadline = time.monotonic() + self.tun.commit_ack_timeout_s
        errors, _ = await self._confirm(run, [pa.proc for pa in run.prepared], resume=False, deadline=deadline)
        if not errors:
            for pa in run.prepared:
                self.sup.committed(pa.proc, run.identity, pa.plan_after)
            for proc in [*run.restarted, *run.spawned]:
                self.sup.committed(proc, run.identity, None)
            errors, resumed = await self._confirm(
                run, [pa.proc for pa in run.prepared] + run.restarted + run.spawned, resume=True, deadline=deadline)
            for proc, ack in resumed.items():
                self.sup.uses_snapshots(proc, ack.snapshots)
                self.sup.runs_units(proc, ack.units)
        run.lap("commit")
        if errors:  # never infer that a lost process has stopped serving
            unconfirmed = sorted({d.subject for d in errors})
            return "failed", [*errors, diag(
                COMPONENT, "commit", f"decided to commit identity {run.identity}, but {', '.join(unconfirmed)} did "
                "not confirm it; the engine needs recovery and refuses further changes until then",
                code="change.recovery_required", hint="restart the engine (v0 has no online recovery yet)")]
        self.recovery_required = False
        return "committed", []

    async def _abort(self, run: _Run) -> list[Diagnostic]:
        self.log.decide(run.id, "abort")  # before any Abort leaves
        out: list[Diagnostic] = []
        if run.plan.control is not None and self.local is not None:
            local = self.local.abort(run.id)
            if any(d.severity == "error" for d in local):  # the old control-plane units are not all back
                self.recovery_required = True
            out += local
        if self.faults.take("skip_abort"):
            self.recovery_required = True
            out.append(diag(COMPONENT, "abort", "Abort not sent (injected)", code="change.injected_fault"))
        else:
            # After Execute, a participant reads Abort once its Execute is done, then reloads back.
            wait = self.tun.execute_timeout_s if run.executing else None
            results = await asyncio.gather(*(self._call(pa.proc, Abort(run.id), timeout=wait) for pa in run.asked))
            for pa, r in zip(run.asked, results, strict=True):
                if isinstance(r, Ok) and isinstance(r.value, Resumed) \
                        and r.value.change_id == run.id and r.value.rolled_back:
                    self.sup.uses_snapshots(pa.proc, r.value.snapshots)
                    self.sup.runs_units(pa.proc, r.value.units)
                else:
                    self.recovery_required = True
                    if isinstance(r, Failed):
                        out += r.diagnostics
                    out.append(diag(COMPONENT, "abort", f"rollback not confirmed: {r}", subject=str(pa.proc),
                                    code="change.abort_unacknowledged",
                                    hint="the participant asks for the decision when its lease runs out"))
        for proc in run.spawned:
            r = await self.sup.stop(proc)
            if isinstance(r, Failed):
                self.recovery_required = True
                out += r.diagnostics
            else:
                self.sup.forget(proc)
        for proc in run.restarted:
            r = await self.sup.relaunch_previous(proc)
            if isinstance(r, Failed):
                self.recovery_required = True
                out += r.diagnostics
        for proc in run.stopped:  # their state is gone; they come back from the spec they had
            r = await self.sup.restart([proc])
            if isinstance(r, Failed):
                self.recovery_required = True
            out += r.diagnostics if isinstance(r, Failed) else [diag(
                COMPONENT, "abort", "was stopped by the change; started again", subject=str(proc),
                code="change.restarted_after_abort", severity="info")]
        if run.plan.topology is not None:
            self.sup.topology = run.old_topology
        if self.recovery_required:
            out.append(diag(COMPONENT, "abort", "rollback or process stop is unconfirmed; restart the engine",
                            code="change.recovery_required"))
        run.lap("abort")
        return out

    # ------------------------------------------------------------ retention

    def prune_blobs(self, limit: int | None = None) -> int:
        """Blob retention (§5.1): keeps the current identity, the one being prepared, the
        ``change.blob_retention`` most recent ones, the contents of queued changes, and
        every blob younger than ``change.blob_grace_s``. Deletes at most ``limit`` blobs;
        returns how many it deleted."""
        keep = [self.identity, *([self._running.identity] if self._running else [])]
        queued = {h for item in self.queue.items() for h in _blobs_of(item.cs)}
        return self.log.prune(keep, self.tun.blob_retention, queued, grace_s=self.tun.blob_grace_s, limit=limit)

    async def prune_if_due(self) -> int:
        """Retention between changes, without holding up the control plane: at most every
        ``change.blob_prune_interval_s``, in batches of ``change.blob_prune_batch`` with the event
        loop free in between (RPCs and heartbeats go on). A queued change goes first; the rest
        is pruned after it. Each batch decides afresh what to keep."""
        if time.monotonic() < self._prune_due_at:
            return 0
        removed = 0
        while True:
            n = self.prune_blobs(limit=self.tun.blob_prune_batch)
            removed += n
            if n < self.tun.blob_prune_batch:
                self._prune_due_at = time.monotonic() + self.tun.blob_prune_interval_s
                return removed
            await asyncio.sleep(0)
            if len(self.queue):
                return removed  # still due: the next pass carries on

    # ------------------------------------------------------------- startup

    async def startup(self, files: dict[str, str], topology: Topology, *, config: dict[str, object] | None = None
                      ) -> ChangeResult:
        """From the empty system to an identity: also a change (modules.html scenario ②)."""
        return await self.submit(ChangeSet(base=EMPTY, files=dict(files), config=dict(config or {}),
                                           topology=topology, source="startup"))

