"""The participant side of a change, inside every process host (change.md §4).

Requests are served at safe points only. Replies that must wait (prepared,
drained, paused at the epoch boundary) are deferred and sent from a later safe
point.

Prepare runs in slices (§4.6): a unit's ``prepare(args)`` may return an
iterator, and every safe point advances each unfinished one by one ``next()``,
so the old version keeps serving in between. A ``prepare`` that returns
``None`` was done in one shot; how long serving stopped for it is reported.

A unit may declare a build step (``UnitSection.build``): what is slow to make for it, made ahead of
its load. A change that reloads the unit, or adds it to a process that already serves, runs the
step first, in a process of its own from the change's code, polled once per slice: the old version
keeps serving while it runs, and the load at the pause finds what it made.

Any coordinator message renews the participant's lease (§4.7). When the lease
runs out while still preparing, the participant withdraws by itself: it has
promised nothing yet. After ``Prepared`` it must not decide alone: it asks the
coordinator. After execution, an unavailable or unknown decision cannot justify
rollback: stay paused. Commit fixes the decision, but only Resume releases the
pause after the coordinator has confirmed every participant.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from core.change.build import signal_session
from core.change.config import ChangeTunables
from core.change.protocol import (
    Abort,
    Alive,
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
from core.change.types import ProcActions, UnitAction
from core.common.ids import Identity, UnitId
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, Severity, diag, fail
from core.context.config import ConfigCommit, ConfigEntry
from core.ipc.rpc import Deferred
from unit.runtime.errors import StagingFailed
from unit.runtime.types import ReloadPlan, ReloadReport

if TYPE_CHECKING:
    from unit.runtime.host import ProcessHost
    from unit.runtime.runtime import UnitRuntime

COMPONENT = "change.participant"


def _exited(pid: int) -> bool:
    """Whether child ``pid`` exited, without reaping it."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _chain(*parts: Iterator[None]) -> Iterator[None]:
    """``itertools.chain`` that ``close`` reaches: closing it closes the part it is in (a build
    step's ``finally`` stops the step's process)."""
    for part in parts:
        yield from part


Phase = Literal["preparing", "prepared", "draining", "drained", "pausing", "paused", "executed", "undone",
                "committed"]
type _Slices = tuple[str, Iterator[object]]
"""What is being prepared (for diagnostics), and the iterator that does one slice per ``next()``."""


@dataclass
class _Undo:
    """What execute changed, so that abort can put it back."""

    reverse_reload: ReloadPlan | None = None
    old_config: dict[str, Any] = field(default_factory=dict[str, Any])
    candidate_root: str | None = None
    """The snapshot execute switched to; modules first imported from it go when it is undone."""
    modules_before: set[str] = field(default_factory=set[str])


@dataclass
class _Change:
    id: int
    identity: Identity
    actions: ProcActions
    deadline: float
    timeout_s: float
    phase: Phase = "preparing"
    staged: bool = False
    prepared_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    """``prepare()`` was called on these (even if it raised): one ``abort()`` each undoes
    prepare and execute, on the instance that did them."""
    config_now: dict[str, Any] = field(default_factory=dict[str, Any])
    """Values of keys that exist already: taken at the epoch boundary."""
    config_new: dict[str, ConfigEntry] = field(default_factory=dict[str, ConfigEntry])
    """Values of keys owned by units added at the safe point: declared with them."""
    deferred: list[UnitAction] = field(default_factory=list[UnitAction])
    """Actions of units this change reloads: prepared at the safe point, on the new instance."""
    slicing: list[_Slices] = field(default_factory=list[_Slices])
    """Sliced preparations not finished yet."""
    prepare_reply: Deferred | None = None
    prepare_diags: list[Diagnostic] = field(default_factory=list[Diagnostic])
    execute_diags: list[Diagnostic] = field(default_factory=list[Diagnostic])
    undo_diags: list[Diagnostic] = field(default_factory=list[Diagnostic])
    undo: _Undo = field(default_factory=_Undo)
    report: ReloadReport | None = None
    drain_reply: Deferred | None = None
    drain_deadline: float = 0.0
    drain_outcomes: dict[str, int] = field(default_factory=dict[str, int])
    draining_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    """``start_drain()`` was called on these (even if it raised): each hears ``commit()`` or
    ``abort()``, so it takes requests again (state-changes.md §5.2)."""
    pause_reply: Deferred | None = None
    pause_after: int | None = None

    def renew(self) -> None:
        self.deadline = time.monotonic() + self.timeout_s


class Participant:
    def __init__(self, host: ProcessHost, tun: ChangeTunables) -> None:
        self.host = host
        self.rt = host.runtime
        self.tun = tun
        self.change: _Change | None = None
        self.finished: dict[int, Resumed] = {}
        """Replies to Commit/Abort of finished changes, for duplicate messages."""
        self._initial_resume_pending = host.spec.start_paused
        self.diagnostics: list[Diagnostic] = []
        for type_, handler in [(Prepare, self.on_prepare), (Drain, self.on_drain), (Pause, self.on_pause),
                               (Execute, self.on_execute), (Commit, self.on_commit), (Abort, self.on_abort),
                               (KeepAlive, self.on_keepalive), (Resume, self.on_resume)]:
            host.handlers[type_] = handler
        host.safe_point_hooks.append(self.tick)

    # ----------------------------------------------------------- helpers

    def _serving_handler(self, a: UnitAction) -> Any:
        """The handler of the instance serving now, or ``None`` once the change removed the unit."""
        return self.rt.action_handler(a.unit, a.action)

    def _handler(self, a: UnitAction) -> Any:
        h = self.rt.action_handler(a.unit, a.action)
        if h is None:
            raise LookupError(f"{a.unit} has no handler for action {a.action!r}")
        return h

    def _code_root_for(self, identity: Identity) -> str | None:
        """Snapshots of one host are siblings: ``<root>/<identity>/engine``."""
        root = self.rt.code_root
        if root is None:
            return None
        candidate = Path(root).parent.parent / identity / "engine"
        return str(candidate) if candidate.is_dir() else None

    def _current(self, change_id: int) -> _Change | Failed:
        c = self.change
        if c is None or c.id != change_id:
            return fail(COMPONENT, "protocol", f"change {change_id} is not in progress here",
                        subject=str(self.host.spec.proc), code="change.unknown_change")
        c.renew()  # the coordinator is alive
        return c

    def _fail(self, phase: str, e: BaseException | str) -> Failed:
        reason = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
        return fail(COMPONENT, phase, reason, subject=str(self.host.spec.proc), code=f"change.{phase}_failed")

    def _note(self, phase: str, reason: str, code: str, severity: Severity = "warning") -> None:
        self.diagnostics.append(diag(COMPONENT, phase, reason, subject=str(self.host.spec.proc), code=code,
                                     severity=severity))

    def on_keepalive(self, env: Envelope, msg: KeepAlive) -> Result[msgspec.Struct]:
        c = self._current(msg.change_id)
        return c if isinstance(c, Failed) else Ok(Alive(c.id))

    # ----------------------------------------------------------- prepare

    def on_prepare(self, env: Envelope, msg: Prepare) -> Result[msgspec.Struct] | Deferred:
        c = self.change
        if c is not None:
            if c.id != msg.change_id:
                return fail(COMPONENT, "prepare", f"busy with change {c.id}", code="change.busy")
            c.renew()
            if c.phase == "undone":
                return self._fail("prepare", "change was already undone; waiting for Abort")
            if c.phase != "preparing":
                return Ok(Prepared(c.id, list(c.prepare_diags)))
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(fail(COMPONENT, "prepare", "superseded by a repeated Prepare",
                                             code="change.duplicate"))
            reply = c.prepare_reply = self.host.rpc.defer()
            return reply
        if msg.change_id in self.finished:
            return fail(COMPONENT, "prepare", "change already finished", code="change.finished")
        c = _Change(msg.change_id, msg.identity, msg.actions, 0.0, msg.coordinator_timeout_s)
        a = msg.actions
        builds = self._builds(a, self._code_root_for(msg.identity))
        removed = set(a.reload.units_to_remove) if a.reload else set[UnitId]()
        orphans = sorted({ua.unit for ua in a.actions if ua.unit in removed})
        if orphans:
            return fail(COMPONENT, "prepare", f"actions of units this change removes: {orphans}",
                        subject=str(self.host.spec.proc), code="change.action_on_removed_unit")
        c.undo.modules_before = set(sys.modules)  # before staging loads anything from the candidate
        if (candidate := self._code_root_for(msg.identity)) != self.rt.code_root:
            c.undo.candidate_root = candidate
        try:
            # Keys of units this change brings are declared with those units, so they start with
            # the values asked for; the other values are taken at the epoch boundary.
            for_new, c.config_now = new_unit_config(a)
            staged_ids = {u.id for u in a.stage}
            c.config_new = {k: e for k, e in for_new.items() if e.owner not in staged_ids}
            if c.config_now:
                r = self.rt.config.stage(c.config_now)
                if isinstance(r, Failed):
                    return r
            if a.stage:  # one new unit per slice, the old version serving in between (§4.6)
                changed = a.reload.changed_modules if a.reload else []
                staging = self.rt.stage_in_slices(a.stage, code_root=self._code_root_for(msg.identity),
                                                  changed_modules=changed,
                                                  config={k: e for k, e in for_new.items() if e.owner in staged_ids})
                c.staged = True  # an undo discards whatever it built
                then = self._staging_then(c, staging, list(a.actions))
                c.slicing.append((f"staging {len(a.stage)} new unit(s)",
                                  _chain(builds, then) if builds is not None else then))
            elif builds is not None:  # the builds first: preparing an action may load what they make
                c.slicing.append(("build steps", self._builds_then(c, builds, list(a.actions))))
            else:
                self._prepare_or_defer(c, a.actions)
        except Exception as e:  # noqa: BLE001 - a failing unit fails the change, not the process
            self._undo(c)
            return self._fail("prepare", e)
        self.change = c
        c.renew()  # work done for the coordinator does not count against its lease
        if not c.slicing:
            c.phase = "prepared"
            return Ok(Prepared(c.id, list(c.prepare_diags)))
        reply = c.prepare_reply = self.host.rpc.defer()
        return reply

    def _start_prepare(self, c: _Change, ua: UnitAction) -> None:
        c.prepared_actions.append(ua)  # before the call: abort() also undoes a prepare that raised
        t0 = time.monotonic()
        out = self._handler(ua).prepare(ua.args)
        if out is None:
            c.prepare_diags.append(self._one_shot(
                f"{ua.unit}.{ua.action}", t0,
                hint="return an iterator from prepare() to prepare in slices between steps"))
        else:
            c.slicing.append((f"{ua.unit}.{ua.action}", iter(out)))

    def _builds(self, a: ProcActions, code_root: str | None) -> Iterator[None] | None:
        """The ``build`` steps (``UnitSection.build``) of the units this change loads here: those it
        reloads, and those it adds to a process that already serves. At a process's start nothing is
        served yet, and the load itself builds."""
        specs = [*(a.reload.units_to_reload if a.reload else []), *(a.reload.units_to_add if a.reload else [])]
        if self.rt.snapshot().units:
            specs += a.stage
        steps = dict.fromkeys((s.id, s.manifest.unit.build) for s in specs if s.manifest.unit.build)
        if not steps or code_root is None:
            return None
        return self._run_builds(list(steps), code_root)

    def _run_builds(self, steps: list[tuple[UnitId, str]], code_root: str) -> Iterator[None]:
        """Each step in a process of its own, from the change's code (``code_root``), one after
        another, polled once per slice: this process keeps serving, and holds no lock and no GIL
        the build needs. Its output goes where this process's goes. A step that exits non-zero
        fails the change before anything paused; an undo (``close``) stops the one running (``_stop_build``)."""
        env = {**os.environ, "PYTHONPATH": code_root}
        for unit, entry in steps:
            t0 = time.monotonic()
            # Its own session: an undo stops it and whatever it started (ninja, the compilers); it
            # does so itself should this process be gone by then (``core.change.build``).
            proc = subprocess.Popen([sys.executable, "-P", "-s", "-m", "core.change.build", entry,
                                     str(self.tun.build_stop_grace_s)], env=env,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                while proc.poll() is None:
                    yield
            finally:
                if proc.returncode is None:
                    self._stop_build(proc)
            took = time.monotonic() - t0
            if proc.returncode != 0:
                raise StagingFailed(fail(COMPONENT, "prepare", f"{unit}: its build step {entry} exited with "
                                         f"{proc.returncode} after {took:.1f}s; its output is in this process's log",
                                         subject=str(unit), code="change.build_failed"))
            print(f"core.change: {unit}: build step {entry} done in {took:.1f}s", file=sys.stderr, flush=True)

    def _stop_build(self, proc: subprocess.Popen[bytes]) -> None:
        """Stop a build step that still runs, and its session (the session, not the process group:
        ``ninja`` runs each compiler in a group of its own): SIGTERM first, so that its cleanup runs
        (``core.change.build``); what is left of the session after ``change.build_stop_grace_s`` is
        killed. The step is reaped only after that: until then its id, its session's id, is not reused."""
        signal_session(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.tun.build_stop_grace_s
        while not _exited(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        for _ in range(10):  # until nothing of it is left: a process may fork meanwhile
            if not signal_session(proc.pid, signal.SIGKILL):
                break
            time.sleep(0.02)
        proc.wait()

    def close(self) -> None:
        """This process is stopping: end what the change in progress still runs here (a build
        step's process). Nothing else is undone: the process unloads every unit next."""
        c = self.change
        if c is None:
            return
        for what, it in c.slicing:
            close = getattr(it, "close", None)
            try:
                if close is not None:
                    close()
            except Exception as e:  # noqa: BLE001
                print(f"core.change: closing {what}: {e}", file=sys.stderr, flush=True)
        c.slicing.clear()

    def _builds_then(self, c: _Change, builds: Iterator[None], actions: list[UnitAction]) -> Iterator[None]:
        yield from builds
        self._prepare_or_defer(c, actions)

    def _staging_then(self, c: _Change, staging: Iterator[None], actions: list[UnitAction]) -> Iterator[None]:
        """The staging slices; then the preparations start, each in its own slices: they may use
        what the new units register (a graph rebuilt with the new kernel), the new units' own actions
        need their instances, and which actions wait for the reload depends on the staged units too."""
        yield from staging
        self._prepare_or_defer(c, actions)

    def _prepare_or_defer(self, c: _Change, actions: list[UnitAction]) -> None:
        """Prepare each action now, unless the reload at the safe point replaces its unit's instance:
        then it is prepared on the new instance there. Decided on the units present now, so after
        staging a new unit that uses a reloaded provider counts (the reload rebuilds it with it)."""
        a = c.actions
        touched = self.rt.units_touched_by(a.reload) if a.reload else set[UnitId]()
        for ua in actions:
            if ua.unit in touched:
                c.deferred.append(ua)
            else:
                self._start_prepare(c, ua)

    def _one_shot(self, what: str, since: float, hint: str = "") -> Diagnostic:
        took = time.monotonic() - since
        return diag(COMPONENT, "prepare", f"{what} prepared in one shot: serving stopped for {took:.3f}s",
                    subject=str(self.host.spec.proc), code="change.prepare_one_shot", hint=hint, severity="info")

    def _advance(self, c: _Change) -> None:
        """One slice of every unfinished preparation; ``Prepared`` once all are done."""
        try:
            for item in list(c.slicing):
                try:
                    next(item[1])
                except StopIteration:
                    c.slicing.remove(item)
        except Exception as e:  # noqa: BLE001
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(e.failed if isinstance(e, StagingFailed) else self._fail("prepare", e))
            self._rollback(c)
            self._resume(c, rolled_back=True)
            return
        if not c.slicing:
            c.phase = "prepared"
            c.renew()
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(Ok(Prepared(c.id, list(c.prepare_diags))))
                c.prepare_reply = None

    def _undo(self, c: _Change) -> None:
        """Put back everything this change did here, newest first. Calling it again does nothing."""
        if c.undo_diags:
            return  # a failed undo needs recovery, not a second attempt reported as success
        before = len(self.diagnostics)
        try:
            self._undo_actions(c)
        except Exception as e:  # noqa: BLE001 - cleanup failures also prevent resuming
            self.diagnostics += self._fail("abort", e).diagnostics
        c.undo_diags = self.diagnostics[before:]
        if c.undo_diags:
            self.change = c
            c.phase = "undone"
            quiesced = self.host.quiesce()
            if isinstance(quiesced, Failed):
                c.undo_diags += quiesced.diagnostics
                self.diagnostics += quiesced.diagnostics
            self.host.paused = True  # even so: this process must stop stepping until recovered

    def _undo_actions(self, c: _Change) -> None:
        for what, it in c.slicing:
            close = getattr(it, "close", None)
            try:
                if close is not None:
                    close()
            except Exception as e:  # noqa: BLE001
                self.diagnostics.append(diag(COMPONENT, "abort", f"{what}: {e}", code="change.abort_failed"))
        c.slicing.clear()
        for ua in reversed(c.prepared_actions):  # before a reverse reload replaces the instances
            try:
                self._handler(ua).abort()
            except Exception as e:  # noqa: BLE001
                self._abort_failed(ua, e)
        c.prepared_actions.clear()
        if c.undo.old_config:  # before the old units are built again: they read these values
            restored = self.rt.config.stage(c.undo.old_config)
            if isinstance(restored, Failed):
                self.diagnostics += restored.diagnostics
            else:
                self.diagnostics += self.rt.config.commit().hook_failures
            c.undo.old_config = {}
        if c.undo.reverse_reload is not None:
            r = self.rt.reload(self._with_fresh_modules(c.undo))
            if isinstance(r, Failed):
                self.diagnostics += r.diagnostics
            c.undo.reverse_reload = None
        self.rt.config.discard_staged()
        if c.staged:
            self.rt.discard_staged()
            c.staged = False
        for ua in reversed(c.draining_actions):  # last: the instance serving again hears it
            try:
                handler = self._serving_handler(ua)
                if handler is not None:
                    handler.abort()
            except Exception as e:  # noqa: BLE001
                self._abort_failed(ua, e)
        c.draining_actions.clear()

    def _with_fresh_modules(self, undo: _Undo) -> ReloadPlan:
        assert undo.reverse_reload is not None
        return with_fresh_modules(self.rt, undo.reverse_reload, undo.candidate_root, undo.modules_before)

    def _abort_failed(self, ua: UnitAction, e: Exception) -> None:
        self.diagnostics.append(diag(COMPONENT, "abort", f"{ua.unit}.{ua.action}: {e}", code="change.abort_failed"))

    # ------------------------------------------------------- drain/pause

    def on_drain(self, env: Envelope, msg: Drain) -> Result[msgspec.Struct] | Deferred:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "preparing":
            return self._fail("drain", "not prepared yet")
        if c.phase == "draining":  # a repeated Drain: the newest request gets the answer
            if c.drain_reply is not None:
                c.drain_reply.resolve(fail(COMPONENT, "drain", "superseded by a repeated Drain",
                                           code="change.duplicate"))
            reply = c.drain_reply = self.host.rpc.defer()
            return reply
        if c.phase != "prepared":
            return Ok(Drained(c.id, dict(c.drain_outcomes)))
        try:
            for ua in c.actions.drain:
                c.draining_actions.append(ua)  # before the call: abort() also ends a drain that raised
                self._handler(ua).start_drain(ua.args)
        except Exception as e:  # noqa: BLE001
            return self._fail("drain", e)
        c.phase = "draining"
        c.drain_deadline = time.monotonic() + msg.timeout_s
        reply = c.drain_reply = self.host.rpc.defer()
        self._check_drain(c)  # may already resolve it
        c.renew()  # work done inside this RPC does not expire the lease in the same safe point
        return reply

    def _check_drain(self, c: _Change) -> None:
        try:
            done = all(self._handler(ua).drained() for ua in c.actions.drain)
            if not done and time.monotonic() >= c.drain_deadline:
                # The deadline passed: each unit settles what is left by its own policy (end them,
                # send them back to the queue, ...) and says how many went which way.
                for ua in c.actions.drain:
                    for outcome, n in (self._handler(ua).drain_deadline() or {}).items():
                        c.drain_outcomes[outcome] = c.drain_outcomes.get(outcome, 0) + int(n)
                done = True
        except Exception as e:  # noqa: BLE001
            c.phase = "prepared"  # still serving; the coordinator aborts (drained() is not asked again)
            if c.drain_reply is not None:
                c.drain_reply.resolve(self._fail("drain", e))
                c.drain_reply = None
            return
        if done and c.drain_reply is not None:
            c.phase = "drained"
            c.drain_reply.resolve(Ok(Drained(c.id, dict(c.drain_outcomes))))
            c.drain_reply = None

    def on_pause(self, env: Envelope, msg: Pause) -> Result[msgspec.Struct] | Deferred:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase in ("paused", "executed", "committed"):
            return Ok(Paused(c.id, self.host.epoch))
        if c.phase == "undone":
            return self._fail("pause", "change was already undone; waiting for Abort")
        if c.phase == "preparing":
            return self._fail("pause", "not prepared yet")
        if c.phase == "pausing" and c.pause_reply is not None:
            c.pause_reply.resolve(fail(COMPONENT, "pause", "superseded by a repeated Pause", code="change.duplicate"))
        c.phase = "pausing"
        c.pause_after = msg.after_epoch
        reply = c.pause_reply = self.host.rpc.defer()
        self._check_pause(c)  # may already resolve it
        c.renew()  # work done inside this RPC does not expire the lease in the same safe point
        return reply

    def _check_pause(self, c: _Change) -> None:
        # The steps the loop has started (device work, results not taken yet) finish first:
        # what Execute unloads must not still be in use, and finishing them may be what reaches
        # the agreed epoch (the last step of a follower whose leader has stopped).
        quiesced = self.host.quiesce()
        if isinstance(quiesced, Failed):
            c.phase = "prepared"  # still serving; the coordinator aborts
            if c.pause_reply is not None:
                c.pause_reply.resolve(quiesced)
                c.pause_reply = None
            return
        if not self._reached(c.pause_after):
            return  # keep stepping until the agreed epoch is done, then finish again
        self.host.paused = True
        c.phase = "paused"
        if c.pause_reply is not None:
            c.pause_reply.resolve(Ok(Paused(c.id, self.host.epoch)))
            c.pause_reply = None

    def _reached(self, after: int | None) -> bool:
        """Whether this process has finished the epoch a Pause waits for. No epoch finished yet
        is not there; a process without a loop numbers no epochs and is there at once."""
        if after is None or not self.host.has_loop():
            return True
        return self.host.epoch is not None and self.host.epoch >= after

    # ----------------------------------------------------------- execute

    def on_execute(self, env: Envelope, msg: Execute) -> Result[msgspec.Struct]:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "executed":
            return Ok(Executed(c.id, c.report, list(c.execute_diags)))
        if c.phase != "paused":
            return fail(COMPONENT, "execute", f"not paused (phase {c.phase})", code="change.not_paused")
        a = c.actions
        try:
            plan = self._reload_plan(c)
            # the way back is taken before the new config takes effect, so it holds the old values
            reverse = self._reverse(plan) if plan is not None else None
            if c.config_now:
                c.undo.old_config = {k: self.rt.config.entries()[k].value for k in c.config_now}
            if plan is None:
                if c.config_now:
                    self._run_hooks(self.rt.config.commit())
            else:
                # the values change inside the reload: if it fails, the old units come back with the
                # old values; the units it did not rebuild hear about them only once it succeeded
                self.rt.config.discard_staged()
                c.undo.candidate_root = plan.new_code_root or c.undo.candidate_root
                r = self.rt.reload(plan, values=c.config_now)
                if isinstance(r, Failed):
                    raise _Failed(r)
                c.report = r.value
                c.undo.reverse_reload = reverse
                changed = {k: v for k, v in c.config_now.items() if c.undo.old_config.get(k) != v}
                self._run_hooks(self.rt.config.run_hooks(changed, skip_units=set(r.value.loaded)))
            for ua in c.deferred:  # the new instances of reloaded units prepare now, all at once
                c.prepared_actions.append(ua)
                t0 = time.monotonic()
                for _ in self._handler(ua).prepare(ua.args) or ():
                    pass
                c.execute_diags.append(self._one_shot(f"{ua.unit}.{ua.action} (its unit was reloaded)", t0))
            for ua in a.actions:
                self._handler(ua).execute(ua.args)
        except Exception as e:  # noqa: BLE001
            self._undo(c)  # Abort acknowledges the undo, or reports that recovery is required
            if isinstance(e, _Failed):
                # reload() can fail to restore its old instances before returning Failed.
                # Finish the participant's cleanup, then keep that rollback failure sticky.
                rollback_errors = [d for d in e.failed.diagnostics if d.phase == "rollback" and d.severity == "error"]
                self.diagnostics += rollback_errors
                c.undo_diags += rollback_errors
            c.phase = "undone"
            return e.failed if isinstance(e, _Failed) else self._fail("execute", e)
        c.phase = "executed"
        c.renew()
        return Ok(Executed(c.id, c.report, list(c.execute_diags)))

    @staticmethod
    def _run_hooks(commit: ConfigCommit) -> None:
        if commit.hook_failures:
            raise RuntimeError("; ".join(d.reason for d in commit.hook_failures))

    def _reload_plan(self, c: _Change) -> ReloadPlan | None:
        """The reload execute does: the planned one, moved to the new identity's snapshot. Without
        one, still move there if the code root differs or staged units wait: after the change the
        process loads code from the snapshot of the identity it serves, and staged units are its own."""
        plan = c.actions.reload
        root = self._code_root_for(c.identity)
        if root is not None and root == self.rt.code_root:
            root = None
        staged = any(u.staged for u in self.rt.snapshot().units)
        if plan is None:
            return ReloadPlan(new_code_root=root) if root is not None or staged else None
        if plan.new_code_root is None and root is not None:
            plan = msgspec.structs.replace(plan, new_code_root=root)
        if c.config_new:  # values for keys of the units it adds, declared before they start
            plan = msgspec.structs.replace(plan, config={**plan.config, **c.config_new})
        return plan

    def _reverse(self, plan: ReloadPlan) -> ReloadPlan:
        return self.rt.reverse_plan(plan)

    # ---------------------------------------------------- commit / abort

    def on_commit(self, env: Envelope, msg: Commit) -> Result[msgspec.Struct]:
        done = self.finished.get(msg.change_id)
        if done is not None:
            if done.rolled_back:
                return fail(COMPONENT, "commit", f"rolled back change {msg.change_id} before Commit arrived",
                            subject=str(self.host.spec.proc), code="change.already_rolled_back",
                            hint="the engine needs recovery")
            return Ok(Committed(done.change_id, done.identity))
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "committed":
            return Ok(Committed(c.id, c.identity))
        if c.phase != "executed":
            return fail(COMPONENT, "commit", f"not executed (phase {c.phase})", code="change.not_executed")
        return Ok(self._commit(c))

    def _commit(self, c: _Change) -> Committed:
        for ua in c.actions.actions:
            try:
                self._handler(ua).commit()
            except Exception as e:  # noqa: BLE001 - commit cannot fail any more; report it
                self.diagnostics.append(diag(COMPONENT, "commit", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.commit_hook_failed"))
        for ua in c.draining_actions:  # the instance serving now: a reloaded unit's new one
            try:
                handler = self._serving_handler(ua)
                if handler is not None:
                    handler.commit()
            except Exception as e:  # noqa: BLE001 - commit cannot fail any more; report it
                self.diagnostics.append(diag(COMPONENT, "commit", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.commit_hook_failed"))
        c.draining_actions.clear()
        self.host.identity = c.identity
        c.phase = "committed"
        c.renew()
        return Committed(c.id, c.identity)

    def on_resume(self, env: Envelope, msg: Resume) -> Result[msgspec.Struct]:
        done = self.finished.get(msg.change_id)
        if done is not None:
            if done.rolled_back or done.identity != msg.identity:
                return fail(COMPONENT, "resume", "change did not commit this identity", code="change.not_committed")
            return Ok(done)
        c = self.change
        if c is None and self._initial_resume_pending and self.host.identity == msg.identity:
            self.host.paused = False
            self._initial_resume_pending = False
            done = Resumed(msg.change_id, msg.identity, snapshots=self._snapshots(), units=self.host.unit_briefs())
            self.finished[msg.change_id] = done
            return Ok(done)
        if c is None or c.id != msg.change_id or c.phase != "committed" or c.identity != msg.identity:
            return fail(COMPONENT, "resume", "change has not committed", code="change.not_committed")
        return self._resume(c, rolled_back=False)

    def on_abort(self, env: Envelope, msg: Abort) -> Result[msgspec.Struct]:
        if msg.change_id in self.finished:
            return Ok(self.finished[msg.change_id])
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            # Abort can overtake Prepare; remember the decision without touching another change.
            reply = Resumed(msg.change_id, self.host.identity, rolled_back=True, units=self.host.unit_briefs())
            self.finished[msg.change_id] = reply
            return Ok(reply)
        if c.phase == "committed":
            return fail(COMPONENT, "abort", "committed decision cannot be rolled back", code="change.already_committed")
        self._rollback(c)
        return self._resume(c, rolled_back=True)

    def _rollback(self, c: _Change) -> None:
        for d in (c.prepare_reply, c.drain_reply, c.pause_reply):
            if d is not None:
                d.resolve(fail(COMPONENT, "abort", "change aborted", code="change.aborted"))
        self._undo(c)

    def _resume(self, c: _Change, *, rolled_back: bool) -> Result[Resumed]:
        if c.undo_diags:
            return Failed(list(c.undo_diags))
        self.host.paused = False
        self.change = None
        reply = Resumed(c.id, self.host.identity, rolled_back, self._snapshots(), self.host.unit_briefs())
        self.finished[c.id] = reply
        return Ok(reply)

    def _snapshots(self) -> list[Identity]:
        """Snapshots this process still loads code from: modules unchanged by in-place
        changes keep their files in older snapshots, so node agents must keep those."""
        root = self.rt.code_root
        if root is None:
            return []
        current = Path(root).resolve().parent
        out = {current.name}
        for mod in list(sys.modules.values()):
            file = getattr(mod, "__file__", None)
            if not file:
                continue
            path = Path(file).resolve()
            if path.is_relative_to(current.parent):
                out.add(path.relative_to(current.parent).parts[0])
        return sorted(Identity(i) for i in out)

    # ---------------------------------------------------------- safe point

    def tick(self) -> None:
        c = self.change
        if c is None:
            return
        # Check the lease before advancing a slice or quiescing: rpc.poll() has just drained any
        # KeepAlives that arrived during the previous slice, so a slow slice does not false-expire
        # while the coordinator is alive, and a silent coordinator still expires on real wall time.
        if time.monotonic() > c.deadline:
            self._lease_expired(c)
            if self.change is not c:
                return
        if c.phase == "preparing":
            self._advance(c)
        elif c.phase == "draining":
            self._check_drain(c)
        elif c.phase == "pausing":
            self._check_pause(c)

    def _lease_expired(self, c: _Change) -> None:
        silent = f"heard nothing from the coordinator of change {c.id} for {c.timeout_s}s"
        if c.phase == "committed":
            c.renew()
            self._note("timeout", f"{silent}; waiting for Resume", "change.in_doubt")
            return
        if c.phase == "preparing":  # it has promised nothing yet
            self._give_up(c, f"{silent} while preparing; withdrew", "change.coordinator_lost")
            return
        answer = self._query(c)
        if answer == "pending":
            c.renew()
            self._note("timeout", f"{silent}; the coordinator answered 'pending'; waiting", "change.decision_pending",
                       severity="info")
        elif answer == "commit" and c.phase == "executed":
            self._commit(c)
            self._note("timeout", f"{silent}; the coordinator answered 'commit'; committed",
                       "change.committed_on_query")
        elif c.phase == "executed" and answer != "abort":
            c.renew()
            self._note("timeout", f"{silent}; decision unresolved; remaining paused", "change.in_doubt")
        elif answer is None:
            self._give_up(c, f"{silent} and could not reach it; rolled back", "change.coordinator_lost")
        else:
            self._give_up(c, f"{silent}; the coordinator answered {answer!r} (phase {c.phase}); rolled back",
                          "change.rolled_back_on_query")

    def _query(self, c: _Change) -> DecisionKind | None:
        r = self.host.client.call(self.host.spec.control, QueryDecision(c.id), timeout=self.host.ipc.rpc_timeout_s)
        return r.value.decision if isinstance(r, Ok) and isinstance(r.value, Decision) \
            and r.value.change_id == c.id else None

    def _give_up(self, c: _Change, reason: str, code: str) -> None:
        self._note("timeout", reason, code, severity="error" if code == "change.coordinator_lost" else "warning")
        self._rollback(c)
        self._resume(c, rolled_back=True)


def new_unit_config(actions: ProcActions) -> tuple[dict[str, ConfigEntry], dict[str, Any]]:
    """Split a change's values: those of keys owned by the units it adds, declared together with
    those units so they are built with them, and the others, for keys already declared here."""
    new_units = [*actions.stage, *(actions.reload.units_to_add if actions.reload else [])]
    owners = {key: (u.id, decl) for u in new_units for key, decl in u.manifest.config.items()}
    for_new = {k: ConfigEntry(owners[k][0], owners[k][1], v) for k, v in actions.config.items() if k in owners}
    return for_new, {k: v for k, v in actions.config.items() if k not in owners}


def with_fresh_modules(rt: UnitRuntime, reverse: ReloadPlan, candidate_root: str | None,
                       modules_before: set[str]) -> ReloadPlan:
    """The reverse reload, also dropping modules first imported from the candidate snapshot
    during the change; otherwise they stay loaded from a snapshot that is about to go, and a
    later change that adds the same module again would find the aborted version cached."""
    if candidate_root is None:
        return reverse
    candidate = Path(candidate_root).resolve()
    fresh: set[str] = set()
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if name not in modules_before and file and file.endswith(".py") \
                and not rt.framework.contains(name) and Path(file).resolve().is_relative_to(candidate):
            fresh.add(name)
    return msgspec.structs.replace(reverse, changed_modules=sorted({*reverse.changed_modules, *fresh}))


class _Failed(Exception):
    def __init__(self, failed: Failed) -> None:
        super().__init__(str(failed))
        self.failed = failed
