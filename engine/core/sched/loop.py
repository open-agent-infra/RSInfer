"""The scheduler's step loop: requests in, plans out to every rank, results back, tokens out.

The structure follows mini-sglang (9a91cfa, MIT, see LICENSE.mini-sglang): a prefill batch if one
can be formed, else the decode batch; admission in order, stopping at the first request that does
not fit; requests reserve their prompt plus ``max_tokens`` up front (``PrefillAdder``,
``estimated_len``). The two decisions are policy units (``core.contracts.policy``); whether a
request fits is the ledger's answer (``core.contracts.ledger``).

What is ours (stage2-contract.md §4):

- **Steps in flight.** Up to ``depth`` planned steps have not retired: 1 without overlap, 2 with
  ``sched.overlap`` (§4.4). A decode row whose last run is still in flight takes its input token
  from the worker (``from_previous``); a request may finish in a step whose result has not come
  back, and its token of the next step is dropped (``sched.reqs``). Its items are released only
  once its last step retired.
- **Ranks.** Every rank gets the same plan, its own ``rank`` in the record. A step retires once
  every rank answered it; the tokens are rank 0's, and every rank must have sampled the same.
- **Domains** (``sched.domains``, doc/specs/v1-ds1017.md stage 3). The ranks are split into
  scheduling domains, one by default. The ranks of a domain are what the line above says. Domains
  hold different requests (data parallel): each has ``max_batch`` rows of its own, a ledger of its
  own (its ranks' stores), and a view of its own for the compose policy, so one domain may compute
  prompts in a step in which another decodes. A waiting request goes to the domain whose prefix
  cache holds most of it, then to the one with most free rows; the order policy orders each
  domain's waiting requests. All domains step together: a step is one plan to every rank, an empty
  one to a domain with nothing to run, because ranks that share experts compute each layer
  together. A domain that refused a step (``Resync``) did not run it while the others did: its
  requests stay where they were.
- **Resync.** A step every rank answered with ``Resync`` did not run. The steps sent after it are
  refused too, so the next plan waits for them and goes out with nothing in flight, as a full
  delta. Ranks that disagree (one ran it, one did not) cannot be put back together: that is a
  failure.
- **Failures.** ``StepFailure``, a step without an answer for ``sched.step_timeout_s``, or ranks
  that disagree: the worker's state is unknown. Then no more steps go out and every request in
  the engine ends with an error at once (its client is not left waiting, and the engine is empty
  for the change that repairs it); new requests get an error, and ``report()`` says what
  happened. Nothing is frozen: control requests, reloads and config keep working. Once a change
  committed (the identity served is no longer the one that failed), the loop starts over as a
  new instance of this unit would: the steps that never came back are given up, the next plan
  resets the workers' persistent batch, and serving goes on. If the change did not repair it, the
  next step fails the same way. A new instance (a reload of this unit) does the same at once.
- **A replaced worker.** Plans or results of a rank going over a new connection mean a new
  worker process (or a broken connection): what the requests had there is gone. Those requests
  end with an error at once, the steps sent there and not answered are given up, and serving
  goes on; nothing waits for a reload. Which requests: those whose plans went over the old
  connection. The two streams see one replacement at different times, so the second sighting
  is paired with the first, and requests first sent to the new worker are not touched. A step
  already on its way to the new worker is refused there (it waits for a full delta) while the
  other ranks may run it: then it counts as not run. A request with nothing computed starts over
  from position 0 on every rank, which clears its state first (``StateStore.clear``); a request
  further on, whose ranks now differ, ends.
- **Never blocking on a peer.** Plans and tokens are written without waiting; what a reader does
  not take yet stays queued and is tried again next turn.
- **What a row holds.** Admission refuses a request longer than one row of the workers'
  persistent batch (``batch.max_context_len``), and plans use at most ``batch.max_rows`` rows:
  the worker raises on a plan beyond them (stage2-contract.md §4.3). A frontend that takes nothing for
  ``sched.frontend_timeout_s`` loses its requests (they are cancelled), not the scheduler.
- **Asleep** (``holder.sleep``): the workers hold no weights or states; every request is refused
  at once, and no step is planned.
- **Changes.** A change applies to an empty engine (doc/specs/empty-apply.md, ``drain.py``): this
  unit says what it holds, ends it all when the change forces it, and once empty takes no
  requests until the change committed or was undone (``held``).
"""

from __future__ import annotations

import gc
import sys
import time
from array import array
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import msgspec

from core.contracts.columns import Column, ragged
from core.contracts.ledger import Ledger
from core.contracts.policy import Candidate, ComposePolicy, OrderPolicy, StepView, Take
from core.contracts.records import (
    STEP_PLAN,
    STEP_RESULT,
    AddedRow,
    BatchPlan,
    Capacity,
    Resync,
    RowDelta,
    Sampled,
    StateAssign,
    StateCopy,
    StateOps,
    StateTrack,
    StepFailure,
)
from core.contracts.requests import ENDED_BY_CHANGE, REQUESTS, TOKENS, Requests, TokenEvents
from core.contracts.serving import ReturnRoute
from core.ipc.records import OpenRecord, RecordBuilder, decode_record, encode_record, read_sections
from core.ipc.stream import StreamAddress

type Answer = Sampled | Resync | StepFailure
FORCED = ENDED_BY_CHANGE
"""The error of a request a change ended at once (doc/specs/empty-apply.md)."""
type Kind = Literal["waiting", "prefilling", "decoding"]


class Reader(Protocol):
    def read(self, timeout: float | None = None) -> bytes | None: ...
    @property
    def generation(self) -> int: ...


class Writer(Protocol):
    generation: int

    def write(self, data: bytes, timeout: float | None = None) -> bool: ...
    def abandon(self) -> None: ...


class Request(Protocol):
    """What this loop uses of an entry of ``sched.reqs@1`` (it sets ``row``)."""

    row: int | None

    @property
    def input(self) -> Any: ...
    computed: int
    """Tokens computed so far; at admission, the prefix the cache holds (``Admitted.prefix``)."""

    @property
    def output_ids(self) -> Sequence[int]: ...
    @property
    def route(self) -> StreamAddress | None: ...
    @property
    def prompt_tokens(self) -> Sequence[int]: ...
    @property
    def reclaimable(self) -> bool: ...
    def finished(self) -> bool: ...


class Table(Protocol):
    """What this loop uses of ``sched.reqs@1``."""

    @property
    def steps(self) -> Mapping[int, Any]: ...
    @property
    def last_submitted(self) -> int: ...
    @property
    def last_retired(self) -> int: ...

    def entries(self) -> Iterable[Request]: ...
    def get(self, request: int) -> Request | None: ...
    def add(self, request: Any, *, route: StreamAddress | None = None) -> Any: ...
    def start_step(self, epoch: int, rows: Mapping[int, int], counts: Mapping[int, int] | None = None) -> None: ...
    def sample(self, epoch: int, identity: str, sampled: Sampled) -> TokenEvents: ...
    def cancel(self, request: int, identity: str) -> TokenEvents: ...
    def retire(self, epoch: int, ran: bool = True, advanced: Mapping[int, int] | None = None) -> list[int]: ...
    def reap(self, request: int) -> Any: ...


@dataclass(frozen=True)
class Settings:
    max_batch: int = 32
    """Rows plans use: ``sched.max_batch``, at most the workers' ``batch.max_rows``."""
    max_context: int = 0
    """Tokens one row of the workers holds (``batch.max_context_len``); 0: no limit."""
    token_budget: int = 2048
    """Prompt tokens per step; sglang's smallest ``chunked_prefill_size``."""
    step_timeout_s: float = 3600.0
    quiesce_timeout_s: float = 30.0
    frontend_timeout_s: float = 5.0
    idle_wait_s: float = 0.01
    asleep: bool = False
    """``holder.sleep``: the workers gave their memory back; every request is refused."""
    draft_tokens: int = 0
    """Speculation (doc/specs/mtp.md): 0 off, else ``spec.num_draft_tokens`` D. A decode row then
    verifies D tokens and advances by what it accepts; its KV is reserved for the worst case."""


ASLEEP = ("the engine is asleep (holder.sleep): it holds no weights or states. Wake it first "
          "(agxinfer wake, or apply holder.sleep false)")


class Fault(msgspec.Struct, frozen=True):
    step: int
    rank: int | None
    reason: str


class Report(msgspec.Struct, frozen=True):
    """``sched.status@1``: what the scheduler holds, for whoever is about to change the engine."""

    waiting: int
    running: int
    in_flight: list[int]
    """Steps planned and not retired, oldest first."""
    last_retired: int | None
    fault: Fault | None
    policy_error: str | None
    diagnostics: list[str]


class PolicyError(Exception):
    """A policy's decision the loop does not carry out."""


@dataclass
class Flight:
    epoch: int
    rows: dict[int, int]
    """row -> request"""
    counts: dict[int, int]
    samples: frozenset[int]
    """Rows whose chunk ends the prompt or decodes: the rows that sample."""
    since: float
    prefills: frozenset[int] = frozenset()
    """Rows that computed prompt tokens: what they computed enters the prefix cache once the step
    retired (``Ledger.commit``)."""
    speculative: frozenset[int] = frozenset()
    """Decode rows that verify drafts: planned at ``draft_tokens`` tokens (``counts``), the most they
    advance; how far they went is their ``Sampled`` count."""
    answers: dict[int, Answer] = field(default_factory=dict[int, Answer])
    void: set[int] = field(default_factory=set[int])
    """Domains this step was sent to after a step their workers refused (``Resync``): they refuse
    this one too."""
    written: dict[int, int] = field(default_factory=dict[int, int])
    """rank -> the generation of the plan connection it went over."""
    refused_by: set[int] = field(default_factory=set[int])
    """Ranks whose new worker refuses this step: it reached them after a replacement, before the
    full delta. A ``Resync`` from them while other ranks ran it is expected."""
    unhomed: list[list[int]] = field(default_factory=list[list[int]])
    """Per domain, request ids in this step that do not yet have ``home`` recorded on every rank of
    their domain."""


class Outbox:
    """Records for one reader, written in order without waiting. ``since`` is when the oldest
    one was queued; each record has a tag (a plan's epoch)."""

    def __init__(self, writer: Writer) -> None:
        self.writer = writer
        self.queue: deque[tuple[float, bytes, int]] = deque()

    def put(self, data: bytes, now: float, tag: int = 0) -> None:
        self.queue.append((now, data, tag))

    @property
    def since(self) -> float | None:
        return self.queue[0][0] if self.queue else None

    def flush(self) -> list[int]:
        """Write what the reader takes; the tags written. They all went over one connection: a
        connection that breaks stops the flush, and the next one is made by the next write."""
        written: list[int] = []
        while self.queue:
            try:
                if not self.writer.write(self.queue[0][1], timeout=0):
                    break
            except TimeoutError:
                break
            written.append(self.queue.popleft()[2])
        return written

    def drop(self, tags: set[int]) -> None:
        self.queue = deque(e for e in self.queue if e[2] not in tags)


class Scheduler:
    def __init__(self, *, table: Table, order: OrderPolicy, compose: ComposePolicy,
                 requests: Reader, results: Sequence[Reader], plans: Sequence[Writer],
                 tokens_to: Callable[[StreamAddress], Writer], discard: Callable[[StreamAddress], None],
                 identity: Callable[[], str], ledger: Ledger | None = None, ledgers: Sequence[Ledger] | None = None,
                 depth: Callable[[], int] = lambda: 1,
                 overlap_lead_s: Callable[[], float] = lambda: 0.0,
                 prefill_coalesce_s: Callable[[], float] = lambda: 0.0,
                 prefill_interval: Callable[[], int] = lambda: 0,
                 prefill_occupancy_target: Callable[[], int] = lambda: 0,
                 balance_prefill: Callable[[], bool] = lambda: False,
                 settings: Settings = Settings(), clock: Callable[[], float] = time.monotonic) -> None:
        if not plans or len(plans) != len(results):
            raise ValueError("one plan writer and one result reader per rank")
        if settings.max_batch < 1 or settings.token_budget < 1:
            raise ValueError("sched.max_batch and sched.token_budget must be positive")
        if (ledger is None) == (ledgers is None):
            raise ValueError("one ledger, or one per scheduling domain")
        self.ledgers: Sequence[Ledger] = ledgers if ledgers is not None else [cast(Ledger, ledger)]
        """One per scheduling domain (``ledger.domains@1``, read through its proxy every time)."""
        if not len(self.ledgers) or len(plans) % len(self.ledgers):
            raise ValueError(f"{len(plans)} rank(s) do not split evenly into {len(self.ledgers)} scheduling "
                             "domain(s) (sched.domains)")
        per = len(plans) // len(self.ledgers)
        self.domains = [list(range(d * per, (d + 1) * per)) for d in range(len(self.ledgers))]
        """The ranks of each domain, in order."""
        align = self.ledgers[0].align
        if settings.token_budget % align:
            # A chunk that does not end its prompt ends on a page (_carry_out): a smaller budget
            # never computes one. SGLang: chunked_prefill_size must be divisible by page_size.
            raise ValueError(f"sched.token_budget = {settings.token_budget} must be a multiple of the prefix "
                             f"cache's page, state.kv.page_size = {align} (as SGLang's "
                             "chunked_prefill_size), or alloc.radix_cache = false")
        self.table, self.order, self.compose = table, order, compose
        self.requests, self.results = requests, list(results)
        self.plans = [Outbox(w) for w in plans]
        self.tokens_to, self.discard, self.identity = tokens_to, discard, identity
        self.depth, self.settings, self.clock = depth, settings, clock
        self.overlap_lead_s = overlap_lead_s
        self.prefill_coalesce_s = prefill_coalesce_s
        self.prefill_interval = prefill_interval
        self.prefill_occupancy_target = prefill_occupancy_target
        self.balance_prefill = balance_prefill
        self._last_retire_at: float = 0.0
        self._decode_step_s: float = 0.012
        self._prefill_tok_s: float = 56000.0
        self._decode_streak: int = 0
        self._coalesce_since: float | None = None
        self._defer_until: float | None = None
        self.flights: deque[Flight] = deque()
        self.full = True
        """The next plan resets the workers' persistent batch: it cannot know what an earlier
        instance of this unit sent them."""
        self.removed: list[int] = []
        self.capacity: dict[int, dict[str, int]] = {}
        self.fault: Fault | None = None
        self.fault_identity = ""
        """The identity served when the loop failed: a change that commits replaces it (``_recover``)."""
        self.policy_error: str | None = None
        self.diagnostics: deque[str] = deque(maxlen=64)
        self.routes: dict[StreamAddress, Outbox] = {}
        # A new instance takes over the connections as they are (they belong to the host).
        self.result_gens = [r.generation for r in self.results]
        self.plan_gens = [w.generation for w in plans]
        self.home: dict[int, dict[int, int]] = {}
        """request -> rank -> the plan connection its first plan to that rank went over."""
        self.expect_results = [False] * len(plans)
        self.expect_plans = [False] * len(plans)
        """One replacement, seen on one stream already: the other stream's change is the same one."""
        self.adopted = False
        self.drain_mode: str | None = None
        """A change is emptying the engine (``drain.py``): ``refuse``, ``wait`` or ``force``."""
        self.held = False
        """Taking no requests: the engine is empty for a change, until it committed or was undone."""
        self.capacity_wanted: set[int] = set()
        """Ranks whose ``Capacity`` under the identity served now has not come since a change committed:
        their stores may have been rebuilt with another size. No step is planned meanwhile."""
        self.recheck = False
        """A step was answered without a record arriving (a replaced rank's part given up): see
        whether it can retire."""
        self.turned_away: dict[int, int] = {}
        """Waiting request -> the domain whose ledger last had no room for it: it tries another next."""
        self._idle_gc_armed = False

    # ------------------------------------------------------------------ the loop

    def step(self) -> int | None:
        before = self.table.last_retired
        if self.fault is not None and self.identity() != self.fault_identity:
            self._recover()
        if not self.adopted:
            self._adopt()
        busy = self._collect()
        busy = self._ingest() or busy
        self._check_timeout()
        self._check_ledger()
        busy = self._plan() or busy
        self._flush()
        if not busy:
            self._wait()
        return self._epoch() if self.table.last_retired != before else None

    def quiesce(self) -> int | None:
        """Every step sent is answered, every token queued for a frontend written (or its route
        given up). With a failed worker the steps it never answered are not waited for. A step
        without an answer within ``sched.quiesce_timeout_s`` fails the pause: the change is not
        made, and the engine goes on serving."""
        deadline = self.clock() + self.settings.quiesce_timeout_s
        while self.flights and self.fault is None:
            self._flush()
            if not self._collect():
                self._wait_result(min(self.settings.idle_wait_s, max(0.0, deadline - self.clock())))
            self._check_timeout()
            if self.flights and self.fault is None and self.clock() >= deadline:
                head = self.flights[0]
                missing = [r for r in range(len(self.results)) if r not in head.answers]
                raise TimeoutError(f"step {head.epoch} has no answer from rank(s) {missing} after "
                                   f"{self.settings.quiesce_timeout_s} s; the engine goes on serving")
        while self.routes:
            self._flush_routes()
            if self.routes:
                time.sleep(self.settings.idle_wait_s)
        return self._epoch()

    def report(self) -> Report:
        waiting = sum(1 for r in self.table.entries() if r.row is None and not r.finished())
        running = sum(1 for r in self.table.entries() if r.row is not None)
        return Report(waiting, running, [f.epoch for f in self.flights], self._epoch(), self.fault,
                      self.policy_error, list(self.diagnostics))

    def _epoch(self) -> int | None:
        return self.table.last_retired if self.table.last_retired >= 1 else None

    def _domain(self, row: int) -> int:
        """The domain a row belongs to: domain ``d`` has the rows ``d * max_batch`` onwards, which
        its ranks know as ``0 .. max_batch - 1``."""
        return row // self.settings.max_batch

    def _of(self, rank: int) -> int:
        return rank // len(self.domains[0])

    def _ledger(self, request: Request) -> Ledger:
        assert request.row is not None
        return self.ledgers[self._domain(request.row)]

    @property
    def _ready(self) -> bool:
        return all(ledger.ready for ledger in self.ledgers)

    def occupancy(self) -> tuple[int, int, int]:
        """Requests running, requests waiting, steps in flight. A request ended (or cancelled)
        no longer counts, but a step still in flight does: it may write what the request held.
        A failed worker's steps never come back, so they do not count once the loop failed."""
        live = [r for r in self.table.entries() if not r.finished()]
        running = sum(1 for r in live if r.row is not None)
        return running, len(live) - running, 0 if self.fault is not None else len(self.flights)

    def flush_ledgers(self) -> None:
        """What the workers' stores hold may be gone or stale: every domain's prefix cache goes."""
        for ledger in self.ledgers:
            ledger.flush()

    def end_all(self, message: str) -> int:
        """End every request with ``message`` as its error; how many."""
        live = [r for r in self.table.entries() if not r.finished()]
        for r in live:
            self._error(r, message)
        return len(live)

    def _note(self, text: str) -> None:
        if not self.diagnostics or self.diagnostics[-1] != text:
            self.diagnostics.append(text)
            print(f"core.sched: {text}", file=sys.stderr, flush=True)

    def _wait(self) -> None:
        wait_s = self.settings.idle_wait_s
        if self._defer_until is not None:
            wait_s = min(wait_s, max(0.0005, self._defer_until - self.clock()))
        if self.flights and self.fault is None:
            self._wait_result(wait_s)
            return
        if self._idle_gc_armed and self._defer_until is None:
            self._idle_gc_armed = False
            gc.collect(0)
        if self.held:
            time.sleep(wait_s)
            return
        raw = self.requests.read(timeout=wait_s)
        if raw is not None:
            self._take_requests(raw)

    def _wait_result(self, timeout: float) -> None:
        head = self.flights[0] if self.flights else None
        rank = next((r for r in range(len(self.results)) if head is None or r not in head.answers), 0)
        raw = self.results[rank].read(timeout=timeout)
        if raw is not None:
            self._take_result(rank, raw)
            self._retire_ready()

    # ------------------------------------------------------------------ taking over

    def _adopt(self) -> None:
        """Steps the request table knows and this instance did not send were abandoned by an
        earlier instance after a worker failure: their requests end with an error, the rest
        carry on from what retired."""
        self.adopted = True
        for request in self.table.entries():
            if request.row is not None:
                self.home[request.input.request] = {rank: self.plan_gens[rank]
                                                    for rank in self.domains[self._domain(request.row)]}
        for epoch in sorted(self.table.steps):
            rows = dict(cast(Mapping[int, int], self.table.steps[epoch].rows))
            for rid in rows.values():
                request = self.table.get(rid)
                if request is not None and not request.finished() and request.route is not None:
                    self._error(request, f"step {epoch} never came back from the workers; "
                                         "the request's state on the device is unknown")
            for ledger in self.ledgers:
                ledger.retire(epoch, ran=False)
            for rid in self.table.retire(epoch, ran=False):
                self._reap(rid)
            self._note(f"took over: step {epoch} was abandoned, its requests ended")

    # ------------------------------------------------------------------ results

    def _collect(self) -> bool:
        got = False
        for rank, reader in enumerate(self.results):
            for _ in range(8):
                raw = reader.read(timeout=0)
                if raw is None:
                    break
                got = True
                self._take_result(rank, raw)
        if got or self.recheck:
            self.recheck = False
            self._retire_ready()
        return got

    def _take_result(self, rank: int, raw: bytes) -> None:
        reader = self.results[rank]
        if reader.generation != self.result_gens[rank]:
            replaced = self.result_gens[rank] != 0
            self.result_gens[rank] = reader.generation
            if replaced:
                self._results_moved(rank)
        try:
            record = decode_record(raw)
            if record.schema != STEP_RESULT or record.rank != rank:
                raise ValueError(f"{record.schema} for rank {record.rank} on rank {rank}'s results")
            parts = read_sections(record, {}, {Capacity.SECTION: Capacity, Sampled.SECTION: Sampled,
                                               Resync.SECTION: Resync, StepFailure.SECTION: StepFailure}).unwrap()
        except Exception as e:  # noqa: BLE001 - a bad record is dropped, not fatal
            self._note(f"rank {rank}: result dropped: {type(e).__name__}: {e}")
            return
        if Capacity.SECTION in parts:
            capacity = cast(Capacity, parts[Capacity.SECTION])
            self._capacity(rank, capacity.items)
            if capacity.identity == self.identity():
                self.capacity_wanted.discard(rank)
        answer = cast(Answer | None, parts.get(StepFailure.SECTION) or parts.get(Resync.SECTION)
                      or parts.get(Sampled.SECTION))
        if answer is None:
            return
        flight = next((f for f in self.flights if f.epoch == record.epoch), None)
        if flight is None or rank in flight.answers:
            if self.fault is None:
                self._note(f"rank {rank}: answer to step {record.epoch}, which is not waiting for one")
            return
        flight.answers[rank] = answer
        if isinstance(answer, StepFailure):
            self._fail(record.epoch, rank, f"worker failed: {answer.reason}")

    def _capacity(self, rank: int, items: dict[str, int]) -> None:
        """A domain's ledger counts what the smallest of its ranks holds, once they all said."""
        self.capacity[rank] = items
        domain = self._of(rank)
        said = [self.capacity[r] for r in self.domains[domain] if r in self.capacity]
        if len(said) < len(self.domains[domain]):
            return
        kinds = set.intersection(*(set(c) for c in said))
        lowest = {k: min(c[k] for c in said) for k in sorted(kinds)}
        ledger = self.ledgers[domain]
        try:
            ledger.set_capacity(lowest)
        except ValueError as e:
            self._note(f"capacity {lowest} not taken: {e}")
            return
        cap = ledger.max_requests
        if cap is not None and cap < self.settings.max_batch:
            self._note(f"at most {cap} requests run at once, not sched.max_batch = {self.settings.max_batch}: "
                       f"the prefix cache's slots per request (as SGLang's max_running_requests)")

    def _plans_moved(self, rank: int, gen: int) -> None:
        """Plans to ``rank`` now go over a new connection: to a new worker process, or the old
        one's connection broke and records on it may be lost. Requests sent over the old one
        lose what they had there. If results moved first, it is the same replacement, handled
        then: the plan writer was abandoned at once, so no plan went over the old connection
        since, and the plan just written is the full delta for the new worker."""
        if self.expect_plans[rank]:
            self.expect_plans[rank] = False
            return
        self.expect_results[rank] = True
        self._lose(rank, lambda first: first < gen, f"plans to rank {rank} go over a new connection "
                                                   "(a new worker process?)")

    def _results_moved(self, rank: int) -> None:
        """Results of ``rank`` come from a new producer. If plans moved first, it is the same
        replacement. Otherwise the plan writer may still be connected to the old reader: it looks
        the reader up again at the next write, and that move is paired with this one."""
        if self.expect_results[rank]:
            self.expect_results[rank] = False
            return
        self.expect_plans[rank] = True
        self.plans[rank].writer.abandon()
        self._lose(rank, lambda first: True, f"results of rank {rank} come from a new producer "
                                            "(a new worker process?)")

    def _lose(self, rank: int, old: Callable[[int], bool], why: str) -> None:
        """End the requests whose plans went to ``rank`` over the old connection, and give up the
        steps sent there that it did not answer (their answers may never come)."""
        domain = self._of(rank)
        ledger = self.ledgers[domain]
        lost = {rid for rid, first in self.home.items() if rank in first and old(first[rank])}
        void = [f for f in self.flights if rank in f.written and old(f.written[rank]) and rank not in f.answers]
        ledger.flush()  # a new worker: what the old one's stores held is gone
        for f in void:
            lost |= {rid for row, rid in f.rows.items() if self._domain(row) == domain}
        restarted = 0
        for request in self.table.entries():
            # Computed tokens but none of them went over the old connection: they came from the
            # prefix cache, which was the old worker's too. It starts over from 0, as a request
            # with nothing computed does (Antigravity, #59), or ends.
            rid = request.input.request
            if rid in lost or request.row is None or self._domain(request.row) != domain or request.finished() \
                    or request.computed == 0:
                continue
            if ledger.restart(rid):
                request.computed = 0
                restarted += 1
            else:
                lost.add(rid)
        if restarted:
            self._note(f"{restarted} request(s) admitted from the prefix cache start over from position 0")
        self.full = True
        for f in self.flights:  # not given up: on their way to the new worker, which refuses them
            if f not in void and rank not in f.answers:
                f.refused_by.add(rank)
        if len(self.domains) > 1:
            # What the other domains ran of these steps counts: only this rank's answer is given up,
            # as if its worker had refused them (its domain's requests on them end below).
            for f in void:
                f.answers[rank] = Resync(why)
                f.refused_by.add(rank)
            void = []
            self.recheck = True
        if not lost:
            self._note(f"{why}; no request was on it")
            return
        for rid in sorted(lost):
            request = self.table.get(rid)
            if request is not None and not request.finished():
                self._error(request, f"{why}: what the request had there is gone")
        tags = {f.epoch for f in void}
        for f in void:
            self.flights.remove(f)
            ledger.retire(f.epoch, ran=False)
            for rid in self.table.retire(f.epoch, ran=False):
                self._reap(rid)
        for outbox in self.plans:
            outbox.drop(tags)
        self._note(f"{why}: {len(lost)} request(s) ended" + (f", step(s) {sorted(tags)} given up" if tags else ""))

    def _retire_ready(self) -> None:
        while self.fault is None and self.flights and len(self.flights[0].answers) == len(self.results):
            flight = self.flights.popleft()
            try:
                self._retire(flight)
            except Exception as e:  # noqa: BLE001 - half-retired: the state is unknown
                self._fail(flight.epoch, None, f"retiring step {flight.epoch} failed: {type(e).__name__}: {e}")

    def _retire(self, flight: Flight) -> None:
        answers = [flight.answers[r] for r in range(len(self.results))]
        everywhere = set(range(len(self.domains)))
        if all(isinstance(a, Resync) for a in answers):
            for ledger in self.ledgers:
                ledger.retire(flight.epoch, ran=False)
            ready = self.table.retire(flight.epoch, ran=False)
            self.full = True
            for later in self.flights:
                later.void |= everywhere
            self._note(f"step {flight.epoch} not run: {cast(Resync, answers[0]).reason or 'Resync'}")
            for rid in ready:
                self._reap(rid)
            return
        # A domain's ranks ran one plan: they sampled the same, or all refused it (it did not run
        # there), or the new workers of replaced ranks refused it while the others ran it.
        ran: dict[int, tuple[Sequence[int], Sequence[int], Sequence[int]]] = {}
        refused_by: set[int] = set()
        split: set[int] = set()
        for domain, ranks in enumerate(self.domains):
            mine = [answers[r] for r in ranks]
            refused = {r for r in ranks if isinstance(answers[r], Resync)}
            if len(refused) == len(ranks):
                continue
            if refused and refused <= flight.refused_by and all(isinstance(a, (Resync, Sampled)) for a in mine):
                refused_by |= refused
                split.add(domain)
                continue
            if domain in flight.void:
                self.flights.appendleft(flight)
                self._fail(flight.epoch, None, f"step {flight.epoch} ran after the workers refused the step "
                                               "before it, without a full delta")
                return
            if not all(isinstance(a, Sampled) for a in mine):
                kinds = {r: type(answers[r]).__name__ for r in ranks}
                self.flights.appendleft(flight)
                self._fail(flight.epoch, None, f"ranks disagree whether step {flight.epoch} ran: {kinds}")
                return
            sampled = cast(Sampled, mine[0])
            if any(msgspec.msgpack.encode(a) != msgspec.msgpack.encode(sampled) for a in mine[1:]):
                self.flights.appendleft(flight)
                self._fail(flight.epoch, None, f"ranks sampled different tokens in step {flight.epoch}")
                return
            kept = self._kept(flight, domain, sampled)
            if kept is None:
                base = domain * self.settings.max_batch
                self.flights.appendleft(flight)
                self._fail(flight.epoch, ranks[0], f"step {flight.epoch} sampled rows "
                                                   f"{sorted(int(x) + base for x in sampled.rows.values())}, not its rows "
                                                   f"{sorted(row for row in flight.rows if self._domain(row) == domain)}")
                return
            ran[domain] = kept
        m_rows = array("i")
        m_counts = array("i")
        m_tokens = array("q")
        for r_seq, c_seq, t_seq in ran.values():
            m_rows.extend(r_seq)
            m_counts.extend(c_seq)
            m_tokens.extend(t_seq)
        merged = Sampled(Column("i32", m_rows.tobytes()),
                         Column("i32", m_counts.tobytes()),
                         Column("i64", m_tokens.tobytes()))
        advanced = self._advanced(flight, merged, ran)
        if advanced is None:
            self.flights.appendleft(flight)
            self._fail(flight.epoch, 0, f"step {flight.epoch} accepted more than its {self.settings.draft_tokens} "
                                        "draft tokens in a row, or none")
            return
        stayed = everywhere - set(ran)
        now = self.clock()
        started = max(flight.since, self._last_retire_at) if self._last_retire_at > 0 else flight.since
        gpu_s = max(now - started, 1e-4)
        self._last_retire_at = now
        if ran:
            if flight.prefills:
                per_dom = [0] * len(self.domains)
                for row in flight.prefills:
                    dom = self._domain(row)
                    if dom in ran:
                        per_dom[dom] += flight.counts.get(row, 0)
                peak_tok = max(per_dom, default=0)
                if peak_tok >= 512 and gpu_s > 1e-3:
                    rate = min(max(peak_tok / gpu_s, 5000.0), 250000.0)
                    self._prefill_tok_s = 0.7 * self._prefill_tok_s + 0.3 * rate
            else:
                self._decode_step_s = 0.8 * self._decode_step_s + 0.2 * min(gpu_s, 0.2)
        events = self.table.sample(flight.epoch, self.identity(), merged) if ran else None
        for domain, ledger in enumerate(self.ledgers):
            if domain in ran:
                ledger.retire(flight.epoch, advanced=({flight.rows[row]: n for row, n in advanced.items()
                                                       if self._domain(row) == domain} if advanced else None))
            else:
                ledger.retire(flight.epoch, ran=False)
        # A row of a domain that did not run the step computed nothing.
        ret_adv = ({**advanced, **{row: 0 for row in flight.rows if self._domain(row) in stayed}}
                   if (advanced or stayed) else None)
        ready = self.table.retire(flight.epoch, advanced=ret_adv)
        if stayed:
            self._stayed(flight, stayed, split, refused_by)
        for row in flight.prefills:  # checkpoint_kv_cache
            request = self.table.get(flight.rows[row])
            if self._domain(row) in ran and request is not None and not request.finished():
                self._ledger(request).commit(request.input.request, request.prompt_tokens)
        if events is not None:
            self._emit(events, flight.epoch)
        for rid in ready:
            self._reap(rid)

    def _stayed(self, flight: Flight, stayed: set[int], split: set[int], refused: set[int]) -> None:
        """The domains ``stayed`` did not run ``flight`` (the others did): their requests are where
        they were, and they get a full delta. In a domain of ``split`` a replaced rank's new worker
        refused it and the domain's other ranks ran it: a request with nothing computed starts over
        from position 0 on every rank, and starting from 0 clears its state first; a request further
        on now has a different state on different ranks: it ends."""
        for row, rid in flight.rows.items():
            request = self.table.get(rid)
            if self._domain(row) in split and request is not None and not request.finished() and request.computed > 0:
                self._error(request, f"rank(s) {sorted(refused)} were replaced and refused step {flight.epoch}, "
                                     "which the others ran: the request's state differs between ranks")
        self.full = True
        for later in self.flights:
            later.void |= stayed
        if len(self.domains) == 1:
            self._note(f"step {flight.epoch} not run: refused by the new worker(s) of rank(s) {sorted(refused)}")
        else:
            self._note(f"step {flight.epoch} not run by domain(s) {sorted(stayed)}: their workers refused it")

    def _advanced(self, flight: Flight, kept: Sampled, ran: Iterable[int]) -> dict[int, int] | None:
        """Row -> tokens each speculative decode row of the domains ``ran`` accepted: its count, 1 to
        ``draft_tokens``. ``None``: a count out of that range."""
        if not flight.speculative:
            return {}
        counts = dict(zip((int(x) for x in kept.rows.values()), (int(x) for x in kept.counts.values()), strict=True))
        domains = set(ran)
        out = {row: counts.get(row, 0) for row in flight.speculative if self._domain(row) in domains}
        if any(not 1 <= n <= self.settings.draft_tokens for n in out.values()):
            return None
        return out

    def _kept(self, flight: Flight, domain: int, sampled: Sampled
              ) -> tuple[Sequence[int], Sequence[int], Sequence[int]] | None:
        """What a domain sampled, as rows, counts and tokens, the rows as the scheduler numbers them:
        the tokens of the rows that sample. Every row of the plan answers, and a row whose chunk did
        not end its prompt has its token dropped (``Sampled``). ``None``: the rows are not the plan's."""
        base = domain * self.settings.max_batch
        max_b = self.settings.max_batch
        s_rows = cast(Sequence[int], sampled.rows.values())
        s_counts = cast(Sequence[int], sampled.counts.values())
        s_tokens = cast(Sequence[int], sampled.tokens.values())
        n_r = len(s_rows)
        if n_r != len(s_counts):
            return None
        if len(flight.samples) == len(flight.rows) and len(s_tokens) == n_r and all(c == 1 for c in s_counts):
            f_rows = flight.rows
            exp_n = sum(1 for row in f_rows if base <= row < base + max_b)
            if n_r != exp_n:
                return None
            out_r = array("i")
            seen_r: set[int] = set()
            for x in s_rows:
                row = x + base
                if row in seen_r or row not in f_rows or not (base <= row < base + max_b):
                    return None
                seen_r.add(row)
                out_r.append(row)
            return out_r, s_counts, s_tokens
        rows = [int(x) + base for x in s_rows]
        counts = [int(x) for x in s_counts]
        tokens = [int(x) for x in s_tokens]
        planned = sorted(row for row in flight.rows if self._domain(row) == domain)
        if sorted(rows) != planned or len(counts) != len(rows) or sum(counts) != len(tokens):
            return None
        keep_rows: list[int] = []
        keep_counts: list[int] = []
        keep_tokens: list[int] = []
        offset = 0
        for row, count in zip(rows, counts, strict=True):
            if row in flight.samples:
                keep_rows.append(row)
                keep_counts.append(count)
                keep_tokens.extend(tokens[offset:offset + count])
            offset += count
        return keep_rows, keep_counts, keep_tokens

    def _check_timeout(self) -> None:
        if self.fault is not None or not self.flights:
            return
        head = self.flights[0]
        if self.clock() - head.since > self.settings.step_timeout_s:
            missing = [r for r in range(len(self.results)) if r not in head.answers]
            self._fail(head.epoch, missing[0] if missing else None,
                       f"no answer to step {head.epoch} from rank(s) {missing} after {self.settings.step_timeout_s} s")

    def _check_ledger(self) -> None:
        """A ledger without capacity while requests hold rows was rebuilt under them (a change
        reloaded ``core.state.alloc``): what they hold is unknown, and admitting anything would
        give their items away again. Until changes refuse that (stage 2's step 5), say so."""
        if self._ready or self.fault is not None:
            return
        held = sum(1 for r in self.table.entries() if r.row is not None)
        if held:
            self._fail(self.table.last_submitted, None, f"the ledger has no capacity while {held} request(s) hold "
                                                        "rows: it was rebuilt under them")
        elif len(self.capacity) == len(self.results):
            for ranks in self.domains:  # nothing held: what the workers said still holds
                self._capacity(ranks[0], self.capacity[ranks[0]])

    def _fail(self, step: int, rank: int | None, reason: str) -> None:
        if self.fault is not None:
            return
        self.fault = Fault(step, rank, reason)
        self.fault_identity = self.identity()
        for outbox in self.plans:
            outbox.queue.clear()  # nothing more goes to a worker in an unknown state
        # No step goes out any more, so a request in the engine could never end by itself: end it
        # now. What it holds is given back once its steps are given up (``_recover``).
        ended = self.end_all(f"the engine stopped serving at step {step}: {reason}")
        self._note(f"stopped at step {step}: {reason}. {ended} request(s) ended with it; new requests get "
                   "an error until a change commits")

    def _recover(self) -> None:
        """A change committed since the failure: start over as a new instance of this unit would.
        Every request ended with the failure, so the steps still in flight (the failed one, and
        those sent with it) are given up by ``_adopt``, which gives their items back; the next
        plan is a full delta, which drops whatever rows the workers still have. Answers to the
        given-up steps that arrive later are dropped (``_take_result``)."""
        fault = self.fault
        assert fault is not None
        self.fault, self.fault_identity = None, ""
        self.flights.clear()
        for outbox in self.plans:
            outbox.queue.clear()
        self.full = True
        self.removed.clear()
        self.home.clear()
        self.result_gens = [r.generation for r in self.results]
        self.plan_gens = [o.writer.generation for o in self.plans]
        self.expect_results = [False] * len(self.plans)
        self.expect_plans = [False] * len(self.plans)
        self.adopted = False
        self._note(f"serving again: a change committed after the failure at step {fault.step} ({fault.reason})")

    # ------------------------------------------------------------------ frontend

    def _ingest(self) -> bool:
        if self.held:  # what comes now waits in the stream: it runs on what the change makes
            return False
        got = False
        for _ in range(256):
            raw = self.requests.read(timeout=0)
            if raw is None:
                break
            got = True
            self._take_requests(raw)
        return got

    def _take_requests(self, raw: bytes) -> None:
        try:
            record = decode_record(raw)
            if record.schema != REQUESTS:
                raise ValueError(f"{record.schema} on the requests stream")
            parts = read_sections(record, {Requests.SECTION: Requests, ReturnRoute.SECTION: ReturnRoute}).unwrap()
        except Exception as e:  # noqa: BLE001 - a bad record is dropped, never fatal
            self._note(f"requests record dropped: {type(e).__name__}: {e}")
            return
        requests = cast(Requests, parts[Requests.SECTION])
        route = cast(ReturnRoute, parts[ReturnRoute.SECTION]).tokens
        for new in requests.new:
            if self.drain_mode == "force":
                self._refuse(route, new.request, FORCED)
                continue
            if self.settings.asleep:
                self._refuse(route, new.request, ASLEEP)
                continue
            if self.fault is not None:
                self._refuse(route, new.request, f"the engine stopped serving at step {self.fault.step}: "
                                                 f"{self.fault.reason}")
                continue
            added = self.table.add(new, route=route)
            if not added.ok:
                self._refuse(route, new.request, "; ".join(d.reason for d in added.diagnostics))
                continue
            if self._ready and (why := self._too_long(self._extent(len(new.prompt), new.max_tokens))):
                self._error(added.value, why)
        for abort in requests.abort:
            request = self.table.get(abort.request)
            if request is None or request.route != route:
                continue
            self._emit(self.table.cancel(abort.request, self.identity()), self.table.last_submitted)
            if request.reclaimable:
                self._reap(abort.request)

    def _extent(self, prompt: int, max_tokens: int) -> int:
        """The positions a request may write: its prompt and ``max_tokens``, and with speculation the
        window a decode step writes past them, twice over for the step in flight before it (SGLang's
        ``get_alloc_reserve_per_decode``)."""
        return prompt + max_tokens + 2 * self.settings.draft_tokens

    def _too_long(self, extent: int) -> str:
        """Why a request of ``extent`` tokens (``_extent``) can never run, or ``""``."""
        what = "prompt plus max_tokens" + (" plus the speculation window" if self.settings.draft_tokens else "")
        if not any(ledger.fits(extent) for ledger in self.ledgers):
            return f"{extent} tokens ({what}) need more state than the engine has"
        if self.settings.max_context and extent > self.settings.max_context:
            return f"{extent} tokens ({what}); a row of the workers holds {self.settings.max_context}"
        return ""

    def _refuse(self, route: StreamAddress, rid: int, message: str) -> None:
        events = TokenEvents(self.identity(), Column.of("i64", [rid]), Column.of("i32", [0, 0]),
                             Column.of("i64", []), {rid: "abort"}, {rid: message})
        self._send(route, events, max(0, self.table.last_submitted))

    def _error(self, request: Request, message: str) -> None:
        """End a live request with an error the client sees."""
        rid = request.input.request
        self.table.cancel(rid, self.identity())
        if request.route is not None:
            self._refuse(request.route, rid, message)
        if request.reclaimable:
            self._reap(rid)

    def _emit(self, events: TokenEvents, epoch: int) -> None:
        req_arr = cast(Sequence[int], events.requests.values())
        if not req_arr:
            return
        first_req = self.table.get(req_arr[0])
        single_route = first_req.route if first_req is not None else None
        if single_route is not None:
            all_same = True
            for rid in req_arr:
                r_obj = self.table.get(rid)
                if r_obj is None or r_obj.route != single_route:
                    all_same = False
                    break
            if all_same:
                self._send(single_route, events, epoch)
                return
        ids = [int(x) for x in req_arr]
        offsets = [int(x) for x in events.offsets.values()]
        tokens = [int(x) for x in events.tokens.values()]
        grouped: dict[StreamAddress, dict[int, list[int]]] = {}
        for i, rid in enumerate(ids):
            request = self.table.get(rid)
            if request is not None and request.route is not None:
                grouped.setdefault(request.route, {})[rid] = tokens[offsets[i]:offsets[i + 1]]
        for route, outputs in grouped.items():
            starts, flat = ragged("i64", outputs.values())
            self._send(route, TokenEvents(events.identity, Column.of("i64", outputs), starts, flat,
                                          {r: f for r, f in events.finished.items() if r in outputs},
                                          {r: e for r, e in events.errors.items() if r in outputs}), epoch)

    def _send(self, route: StreamAddress, events: TokenEvents, epoch: int) -> None:
        outbox = self.routes.get(route)
        if outbox is None:
            outbox = self.routes[route] = Outbox(self.tokens_to(route))
        record = RecordBuilder(TOKENS, epoch).add(TokenEvents.SECTION, events).build()
        outbox.put(encode_record(record), self.clock())

    def _flush_routes(self) -> None:
        for route, outbox in list(self.routes.items()):
            try:
                outbox.flush()
            except Exception as e:  # noqa: BLE001 - only this route is lost
                self._drop(route, f"{type(e).__name__}: {e}")
                continue
            since = outbox.since
            if since is None:
                del self.routes[route]
            elif self.clock() - since > self.settings.frontend_timeout_s:
                self._drop(route, f"took nothing for {self.settings.frontend_timeout_s} s")

    def _drop(self, route: StreamAddress, why: str) -> None:
        """A frontend that does not take its tokens: its requests are cancelled. A later request
        from the same address is served again, over a new connection."""
        self.routes.pop(route, None)
        self.discard(route)
        n = 0
        for request in list(self.table.entries()):
            if request.route == route and not request.finished():
                n += 1
                self.table.cancel(request.input.request, self.identity())
                if request.reclaimable:
                    self._reap(request.input.request)
        self._note(f"frontend {route} dropped ({why}); {n} request(s) cancelled")

    def _reap(self, rid: int) -> None:
        request = self.table.get(rid)
        if request is None or not request.reclaimable:
            return
        # A request without a row holds nothing; every ledger forgets what it matched of it.
        targets = self.ledgers if request.row is None else [self._ledger(request)]
        if any(getattr(l, "caches", True) for l in targets):
            prompt_len = len(request.prompt_tokens)
            if request.computed <= prompt_len or not request.output_ids:
                computed: Sequence[int] = request.prompt_tokens[:request.computed]
            else:
                extra = request.output_ids[:request.computed - prompt_len]
                packed = request.input.prompt.data + Column.of("i64", extra).data
                computed = memoryview(packed).cast("q")
        else:
            computed = ()
        for ledger in targets:
            ledger.release(rid, self.table.last_retired, computed)
        if request.row is not None:
            self.removed.append(request.row)
        self.home.pop(rid, None)
        self.turned_away.pop(rid, None)
        self.table.reap(rid)

    # ------------------------------------------------------------------ planning

    def _flush(self) -> None:
        for rank, outbox in enumerate(self.plans):
            written: list[int] = []
            try:
                written = outbox.flush()
            except Exception as e:  # noqa: BLE001
                self._fail(self.flights[0].epoch if self.flights else self.table.last_submitted, rank,
                           f"writing a plan to rank {rank} failed: {type(e).__name__}: {e}")
            gen = outbox.writer.generation
            if gen != self.plan_gens[rank]:
                moved = self.plan_gens[rank] != 0
                self.plan_gens[rank] = gen
                if moved:  # before the plans just written are counted: they went to the new reader
                    self._plans_moved(rank, gen)
            if written:
                flights = {f.epoch: f for f in self.flights}
                dom = self._of(rank)
                for epoch in written:
                    flight = flights.get(epoch)
                    if flight is not None:
                        flight.written[rank] = gen
                        if dom < len(flight.unhomed):
                            for rid in flight.unhomed[dom]:
                                self.home.setdefault(rid, {}).setdefault(rank, gen)
                        else:
                            for row, rid in flight.rows.items():
                                if self._domain(row) == dom:
                                    self.home.setdefault(rid, {}).setdefault(rank, gen)
        self._flush_routes()

    def _plan(self) -> bool:
        if self.fault is not None or self.settings.asleep or not self._ready or \
                len(self.flights) >= max(1, self.depth()):
            return False
        if self.capacity_wanted:
            return False  # after a change: plan with the workers' capacity of now, not an older one
        if self.full and self.flights:
            return False  # a full delta goes out once every step sent before it is answered
        try:
            planned = self._plan_step()
        except PolicyError as e:
            if self.policy_error != str(e):
                self.policy_error = str(e)
                self._note(f"policy decision not carried out: {e}")
            return False
        except Exception as e:  # noqa: BLE001 - this step is not sent; the next turn tries again
            self._note(f"planning failed: {type(e).__name__}: {e}")
            return False
        self.policy_error = None
        return planned

    def _in_flight(self) -> tuple[dict[int, int], dict[int, int]]:
        tokens: dict[int, int] = {}
        samples: dict[int, int] = {}
        for f in self.flights:
            for row, rid in f.rows.items():
                tokens[rid] = tokens.get(rid, 0) + f.counts[row]
                if row in f.samples:
                    samples[rid] = samples.get(rid, 0) + 1
        return tokens, samples

    def _view(self, tokens: Mapping[int, int], samples: Mapping[int, int]
              ) -> tuple[dict[int, tuple[Kind, Request, int]], list[_View]]:
        """What can run, and each domain's view of it."""
        kinds: dict[int, tuple[Kind, Request, int]] = {}
        views = [_View() for _ in self.domains]
        queued = [0] * len(self.domains)
        for request in self.table.entries():
            if request.row is not None:
                dom = self._domain(request.row)
                views[dom].free_rows -= 1
                if not request.finished():
                    rem = len(request.prompt_tokens) - (request.computed + tokens.get(request.input.request, 0))
                    if rem > 0:
                        queued[dom] += rem
        for view in views:
            view.free_rows += self.settings.max_batch
        room = [view.free_rows for view in views]  # where a waiting request can be admitted: where it starts
        for arrival, request in enumerate(self.table.entries()):
            if request.finished():
                continue
            rid = request.input.request
            prompt = len(request.prompt_tokens)
            if request.row is not None:
                view = views[self._domain(request.row)]
                planned = request.computed + tokens.get(rid, 0)
            else:
                domain, planned = self._route(request, views, room, queued)
                view = views[domain]
            output = len(request.output_ids) + samples.get(rid, 0)
            c = Candidate(rid, arrival, prompt, min(planned, prompt), output, request.input.max_tokens)
            if request.row is None:
                kind: Kind = "waiting"
                view.waiting.append(c)
            elif planned < prompt:
                kind = "prefilling"
                view.prefilling.append(c)
            elif output < request.input.max_tokens:
                kind = "decoding"
                view.decoding.append(c)
            else:
                continue  # its last token is being sampled
            kinds[rid] = kind, request, planned
        for view in views:
            view.free_rows = max(0, view.free_rows)
        return kinds, views

    def _route(self, request: Request, views: Sequence[_View], room: list[int],
               queued: list[int]) -> tuple[int, int]:
        """The domain a waiting request would be admitted to, and how much of its prompt that
        domain's prefix cache holds. Of the domains with a free row whose ledger can hold it at all:
        not the one that just had no room for it, if there is another; then the one whose cache
        holds most of its prompt; of equals, the one with most room left after the requests routed
        before it. Without such a domain it waits where fewest wait, and starts at 0."""
        rid, prompt = request.input.request, request.prompt_tokens
        extent = self._extent(len(prompt), request.input.max_tokens)
        balance = self.balance_prefill()
        open_ = [d for d, view in enumerate(views) if view.free_rows > 0 and
                 (len(views) == 1 or self.ledgers[d].fits(extent))]
        if not open_:
            domain = (max(range(len(room)), key=lambda d: (-queued[d], room[d], -d))
                      if balance else max(range(len(room)), key=lambda d: (room[d], -d)))
            room[domain] -= 1
            queued[domain] += len(prompt)
            return domain, 0
        matches = {d: self.ledgers[d].match(rid, prompt) for d in open_}
        domain = (max(open_, key=lambda d: (d != self.turned_away.get(rid), matches[d], -queued[d], room[d], -d))
                  if balance else max(open_, key=lambda d: (d != self.turned_away.get(rid), matches[d], room[d], -d)))
        room[domain] -= 1
        queued[domain] += max(0, len(prompt) - matches[domain])
        return domain, matches[domain]

    def _decide(self, waiting: list[Candidate], prefilling: list[Candidate], decoding: list[Candidate],
                free_rows: int) -> list[Take]:
        if waiting:
            ordered = list(self.order.order(waiting))
            known = {c.request: c for c in waiting}
            if len(set(ordered)) != len(ordered) or not set(ordered) <= set(known):
                raise PolicyError(f"order {ordered} is not a list of distinct waiting requests")
            waiting = [known[r] for r in ordered]
        view = StepView(waiting, prefilling, decoding, self.settings.token_budget, free_rows)
        takes = list(self.compose.compose(view))
        allowed = {c.request: (kind, c) for kind, cs in (("waiting", waiting), ("prefilling", prefilling),
                                                         ("decoding", decoding)) for c in cs}
        seen: set[int] = set()
        prompt_tokens = new_rows = 0
        for t in takes:
            if t.request in seen or t.request not in allowed:
                raise PolicyError(f"{t} names a request twice, or one that cannot run now")
            seen.add(t.request)
            kind, c = allowed[t.request]
            if kind == "decoding":
                if t.tokens != 1:
                    raise PolicyError(f"{t}: a decoding request takes exactly 1 token")
                continue
            if not 1 <= t.tokens <= c.prompt - c.planned:
                raise PolicyError(f"{t}: {c.prompt - c.planned} prompt tokens are left")
            prompt_tokens += t.tokens
            new_rows += kind == "waiting"
        if prompt_tokens > self.settings.token_budget or new_rows > free_rows:
            raise PolicyError(f"{prompt_tokens} prompt tokens and {new_rows} new rows; the budget is "
                              f"{self.settings.token_budget} tokens and {free_rows} rows")
        return takes

    def _flight_eta(self, flight: Flight) -> float:
        started = max(flight.since, self._last_retire_at) if self._last_retire_at > flight.since else flight.since
        if not flight.prefills:
            return started + self._decode_step_s
        per_dom = [0] * len(self.domains)
        for row in flight.prefills:
            per_dom[self._domain(row)] += flight.counts.get(row, 0)
        peak_tok = max(per_dom, default=0)
        return started + max(self._decode_step_s, peak_tok / max(self._prefill_tok_s, 1.0))

    def _plan_step(self) -> bool:
        self._defer_until = None
        tokens, samples = self._in_flight()
        kinds, views = self._view(tokens, samples)
        if not kinds:
            self._coalesce_since = None
            self._decode_streak = 0
            return False
        now = self.clock()
        any_decoding = any(v.decoding for v in views)
        any_waiting = any(v.waiting for v in views)
        any_prefilling = any(v.prefilling for v in views)
        budget = self.settings.token_budget

        # 1. Idle multi-domain prefill coalescing: wait briefly when idle so concurrent HTTP arrivals
        #    spread across domains before sealing step 1 on a single GPU.
        coalesce_s = self.prefill_coalesce_s()
        if (not self.flights and coalesce_s > 0 and len(views) > 1 and not any_decoding
                and any_waiting and not all(v.waiting or v.prefilling for v in views)):
            if self._coalesce_since is None:
                self._coalesce_since = now
            if now < self._coalesce_since + coalesce_s:
                self._defer_until = self._coalesce_since + coalesce_s
                return False
        else:
            self._coalesce_since = None

        # 2. Coordinated prefill cadence while decoding: batch prefill chunks (both new admissions and
        #    remaining multi-step chunks) across DP+EP domains so all ranks run eager prefill together
        #    instead of stalling decoding ranks on isolated single-domain chunks.
        interval = self.prefill_interval()
        target = self.prefill_occupancy_target() * len(views)
        if interval > 0 and target > 0 and any_decoding:
            active = min(target, sum(len(view.decoding) for view in views))
            interval = max(1, (interval * active + target - 1) // target)
        hold_prefill = False
        if not any_decoding:
            self._decode_streak = 0
        elif interval > 0 and (any_waiting or any_prefilling):
            prefill_toks = [
                sum(c.prompt - c.planned for c in (*v.prefilling, *v.waiting))
                for v in views
            ]
            min_prefill_toks = min(prefill_toks, default=0)
            all_have_prefill = min_prefill_toks > 0
            all_saturated = min_prefill_toks >= max(1, (budget * 3) // 4)
            all_point = max(12, interval // 4)
            if self._decode_streak == 0:
                if not all_have_prefill:
                    hold_prefill = True
            elif self._decode_streak < interval:
                if not (
                    (all_have_prefill and self._decode_streak >= all_point)
                    or (all_saturated and self._decode_streak >= max(8, all_point // 2))
                ):
                    hold_prefill = True

        # 3. Overlap lead-time pacing: while a prefill step is running on GPU (len(flights) == 1),
        #    avoid sealing step N + 1 hundreds of ms early with idle domains unless all domains are
        #    already at least half-full of prefill tokens.
        lead_s = self.overlap_lead_s()
        if (len(self.flights) == 1 and lead_s > 0 and self.flights[0].prefills
                and not hold_prefill and (any_waiting or any_prefilling)):
            all_saturated = all(
                sum(c.prompt - c.planned for c in (*v.prefilling, *v.waiting)) >= max(1, budget // 2)
                for v in views
            )
            if not all_saturated:
                eta = self._flight_eta(self.flights[0])
                if now < eta - lead_s:
                    self._defer_until = eta - lead_s
                    return False

        epoch = max(self.table.last_submitted, 0) + 1
        used: set[int] | None = None
        batch = self.settings.max_batch

        def try_build(defer_prefill: bool) -> list[_Step]:
            nonlocal used
            built = [_Step(epoch, domain * batch) for domain in range(len(self.domains))]
            try:
                for step, view in zip(built, views, strict=True):
                    waiting_cs = [] if defer_prefill else view.waiting
                    prefilling_cs = [] if defer_prefill else view.prefilling
                    if not (waiting_cs or prefilling_cs or view.decoding):
                        continue
                    if waiting_cs:
                        if used is None:
                            used = {r.row for r in self.table.entries() if r.row is not None}
                        free = [row for row in range(step.base, step.base + batch) if row not in used]
                    else:
                        free = []
                    takes = self._decide(waiting_cs, prefilling_cs, view.decoding, view.free_rows)
                    if self.settings.draft_tokens and any(kinds[t.request][0] != "decoding" for t in takes):
                        # No mixed step with speculation (SGLang disables mixed chunks): the prompts go first.
                        takes = [t for t in takes if kinds[t.request][0] != "decoding"]
                    refused = self._carry_out(step, takes, kinds, tokens, free)
                    if not step.rows and refused:
                        takes = self._decide([], prefilling_cs, view.decoding, view.free_rows)
                        self._carry_out(step, takes, kinds, tokens, free)
                if not any(step.rows for step in built):
                    for step in built:
                        ledger = self.ledgers[step.base // batch]
                        for rid in {*step.rows.values(), *(r.input.request for r in step.admitted)}:
                            ledger.cancel(rid, epoch)
                        for request in step.admitted:
                            request.row = None
                            request.computed = 0
                return built
            except BaseException:
                for step in built:
                    ledger = self.ledgers[step.base // batch]
                    for rid in {*step.rows.values(), *(r.input.request for r in step.admitted)}:
                        ledger.cancel(rid, epoch)
                    for request in step.admitted:
                        request.row = None
                        request.computed = 0
                raise

        steps = try_build(hold_prefill)
        if not any(step.rows for step in steps) and hold_prefill:
            steps = try_build(False)
        if not any(step.rows for step in steps):
            return False
        try:
            self._send_plan(steps)
        except BaseException:  # nothing of this step went out: give back what it took
            for step in steps:
                ledger = self.ledgers[step.base // batch]
                for rid in {*step.rows.values(), *(r.input.request for r in step.admitted)}:
                    ledger.cancel(rid, epoch)
                for request in step.admitted:
                    request.row = None
                    request.computed = 0
            raise
        self._coalesce_since = None
        self._idle_gc_armed = True
        if any(step.prefills for step in steps):
            self._decode_streak = 0
        else:
            self._decode_streak += 1
        return True

    def _carry_out(self, step: _Step, takes: list[Take], kinds: Mapping[int, tuple[Kind, Request, int]],
                   in_flight: Mapping[int, int], free: list[int]) -> bool:
        """Admit and reserve for each take of one domain, in order; after the first waiting request
        that does not fit, no later waiting request is admitted. Returns whether one did not fit."""
        ledger = self.ledgers[step.base // self.settings.max_batch]
        refused = False
        for take in takes:
            kind, request, planned = kinds[take.request]
            rid = take.request
            if kind == "waiting":
                if refused:
                    continue
                extent = self._extent(len(request.prompt_tokens), request.input.max_tokens)
                if why := self._too_long(extent):
                    self._error(request, why)
                    continue
                # Where it would start now (an admission before it in this step may have evicted
                # part of its prefix): a chunk that reaches no page does not run, and admitting it
                # would only evict from the cache (Antigravity, #59; PrefillAdder checks first too).
                if self._chunk(request, ledger.match(rid, request.prompt_tokens), take.tokens) <= 0:
                    continue
                got = ledger.admit(rid, request.prompt_tokens, extent, step.epoch)
                if got is None:
                    refused = True
                    self.turned_away[rid] = step.base // self.settings.max_batch
                    continue
                self.turned_away.pop(rid, None)
                request.row = free.pop(0)
                request.computed = planned = got.prefix  # the cached prefix is computed
                step.admitted.append(request)
                step.added.append(AddedRow(request.row, rid, request.input.sampling))
                step.assign_all(request.row, got.items)
            row = request.row
            assert row is not None
            speculative = kind == "decoding" and self.settings.draft_tokens > 0
            n = self.settings.draft_tokens if speculative else take.tokens if kind == "decoding" else \
                self._chunk(request, planned, take.tokens)
            if n <= 0:
                if kind == "waiting":
                    ledger.cancel(rid, step.epoch)
                    step.unadmit(request, free)
                continue
            # Speculative: planned counts the steps in flight at D tokens each, the most they go, so
            # this reserves the window of the furthest this one can start at (2 x D with overlap).
            extra = ledger.reserve(rid, planned + n, step.epoch, speculative=speculative)
            if extra is None:  # what step.epoch took for it is back in the ledger
                if kind == "waiting":
                    step.unadmit(request, free)
                continue
            step.assign_all(row, extra.items)
            for state_kind, (src, dst) in extra.restore.items():
                step.copies.setdefault((state_kind, "before"), []).append((src, dst))
            for state_kind, src, dst in extra.snapshot:
                step.copies.setdefault((state_kind, "after"), []).append((src, dst))
            for state_kind, (dst, upto) in extra.track.items():
                step.track.setdefault(state_kind, []).append((row, dst, upto))
            step.rows[row], step.counts[row] = rid, n
            step.prompt_lengths.append(len(request.prompt_tokens))
            # A speculative decode row starts where the retired steps left it at least; the worker
            # knows where the steps in flight took it (BatchPlan.positions).
            step.positions.append(request.computed if speculative else planned)
            if speculative:
                step.speculative.add(row)
            if kind == "decoding":
                previous = rid in in_flight  # its last run is still in flight: the worker has the token
                step.from_previous.append(int(previous))
                if not previous:
                    step.tokens.append(request.output_ids[-1])
                step.decodes += 1
                step.next_tokens.append(-1)
            else:
                step.from_previous.append(0)
                step.tokens.extend(request.prompt_tokens[planned:planned + n])
                step.prefills += 1
                step.prefill_rows.add(row)
                # the draft model's extend takes the token after the chunk (BatchPlan.next_tokens)
                step.next_tokens.append(request.prompt_tokens[planned + n] if planned + n < len(request.prompt_tokens)
                                        else -1)
            if planned + n >= len(request.prompt_tokens):
                step.samples.add(row)
        return refused

    def _chunk(self, request: Request, planned: int, tokens: int) -> int:
        """Prompt tokens a take of ``tokens`` computes from ``planned``: no more than are left (an
        admission may start further on than the view did). A chunk that does not end its prompt
        ends on a page: the cache keeps whole pages (PrefillAdder: end = (prefix_len + extend_len)
        // page_size * page_size). 0 or less: nothing runs."""
        left = len(request.prompt_tokens) - planned
        n = min(tokens, left)
        if n < left:
            align = self.ledgers[0].align
            n = (planned + n) // align * align - planned
        return self.ledgers[0].prefill_end(planned, planned + n, len(request.prompt_tokens)) - planned

    def _send_plan(self, steps: Sequence[_Step]) -> None:
        """One step: each domain's part to its ranks, an empty one to a domain with nothing to run."""
        epoch = steps[0].epoch
        if self.full:
            for step in steps:
                step.added, step.assign = [], {}
            for request in self.table.entries():
                if request.row is not None and not request.finished():
                    step = steps[self._domain(request.row)]
                    step.added.append(AddedRow(request.row, request.input.request, request.input.sampling))
                    step.assign_all(request.row, self._ledger(request).items(request.input.request))
        first = steps[0]
        uniform_prefill = bool(first.prefills and not first.decodes and len(steps) > 1) and all(
            step.prefills and not step.decodes and step.positions == first.positions
            and step.prompt_lengths == first.prompt_lengths
            and tuple(step.counts.values()) == tuple(first.counts.values()) for step in steps[1:])
        records = [self._record(step, uniform_prefill) for step in steps]
        rows = {row: rid for step in steps for row, rid in step.rows.items()}
        counts = {row: n for step in steps for row, n in step.counts.items()}
        self.table.start_step(epoch, rows, counts)
        now = self.clock()
        for ranks, record in zip(self.domains, records, strict=True):
            for rank in ranks:
                self.plans[rank].put(encode_record(OpenRecord(record.schema, record.epoch, record.sections, rank)),
                                     now, epoch)
        per_dom = len(self.domains[0])
        unhomed = [[rid for rid in step.rows.values()
                    if rid not in self.home or len(self.home[rid]) < per_dom]
                   for step in steps]
        self.flights.append(Flight(epoch, rows, counts, frozenset(row for step in steps for row in step.samples), now,
                                   frozenset(row for step in steps for row in step.prefill_rows),
                                   frozenset(row for step in steps for row in step.speculative),
                                   unhomed=unhomed))
        self.full = False
        self.removed.clear()

    def _record(self, step: _Step, uniform_prefill: bool = False) -> OpenRecord:
        """A domain's part of a step, its rows as its ranks number them."""
        base = step.base
        phase: Literal["prefill", "decode", "mixed"] = (
            "decode" if not step.prefills else "prefill" if not step.decodes else "mixed")
        counts = [1 if row in step.speculative else n for row, n in step.counts.items()]
        batch = BatchPlan(Column.of("i32", [row - base for row in step.rows]), Column.of("i32", counts),
                          Column.of("i32", step.positions), Column("i64", step.tokens.tobytes()),
                          Column.of("u8", step.from_previous), phase, 0, self.settings.draft_tokens,
                          Column.of("i64", step.next_tokens if self.settings.draft_tokens or
                                    (step.prefills and any(t >= 0 for t in step.next_tokens)) else []),
                          Column.of("i32", step.prompt_lengths if phase != "decode" else []), uniform_prefill)
        state = StateOps([StateAssign(kind, Column.of("i32", [row - base for row in per_row]),
                                      *ragged("i32", per_row.values())) for kind, per_row in step.assign.items()],
                         [StateCopy(kind, when, Column.of("i32", [src for src, _ in cs]),
                                    Column.of("i32", [dst for _, dst in cs])) for (kind, when), cs in step.copies.items()],
                         [StateTrack(kind, Column.of("i32", [t[0] - base for t in ts]), Column.of("i32", [t[1] for t in ts]),
                                     Column.of("i32", [t[2] for t in ts])) for kind, ts in step.track.items()])
        removed = [] if self.full else [row - base for row in self.removed
                                        if base <= row < base + self.settings.max_batch]
        delta = RowDelta([AddedRow(a.row - base, a.request, a.sampling) for a in step.added],
                         Column.of("i32", removed), self.full)
        return (RecordBuilder(STEP_PLAN, step.epoch).add(RowDelta.SECTION, delta)
                .add(BatchPlan.SECTION, batch).add(StateOps.SECTION, state).build())


@dataclass
class _View:
    """One domain's part of what can run (``StepView``)."""

    waiting: list[Candidate] = field(default_factory=list[Candidate])
    prefilling: list[Candidate] = field(default_factory=list[Candidate])
    decoding: list[Candidate] = field(default_factory=list[Candidate])
    free_rows: int = 0


@dataclass
class _Step:
    """One domain's part of a step; rows as the scheduler numbers them (``base`` onwards)."""

    epoch: int
    base: int = 0
    rows: dict[int, int] = field(default_factory=dict[int, int])
    counts: dict[int, int] = field(default_factory=dict[int, int])
    positions: list[int] = field(default_factory=list[int])
    tokens: array[int] = field(default_factory=lambda: array("q"))
    from_previous: list[int] = field(default_factory=list[int])
    samples: set[int] = field(default_factory=set[int])
    added: list[AddedRow] = field(default_factory=list[AddedRow])
    assign: dict[str, dict[int, list[int]]] = field(default_factory=dict[str, dict[int, list[int]]])
    admitted: list[Request] = field(default_factory=list[Request])
    copies: dict[tuple[str, Literal["before", "after"]], list[tuple[int, int]]] = field(
        default_factory=dict[tuple[str, Literal["before", "after"]], list[tuple[int, int]]])
    track: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict[str, list[tuple[int, int, int]]])
    """kind -> (row, dst, upto): states the step keeps for the prefix cache."""
    prompt_lengths: list[int] = field(default_factory=list[int])
    prefill_rows: set[int] = field(default_factory=set[int])
    speculative: set[int] = field(default_factory=set[int])
    next_tokens: list[int] = field(default_factory=list[int])
    prefills: int = 0
    decodes: int = 0

    def assign_all(self, row: int, items: Mapping[str, list[int]]) -> None:
        for kind, ids in items.items():
            if ids:
                self.assign.setdefault(kind, {}).setdefault(row, []).extend(ids)

    def unadmit(self, request: Request, free: list[int]) -> None:
        row = request.row
        assert row is not None
        request.row = None
        request.computed = 0
        self.admitted.remove(request)
        self.added = [a for a in self.added if a.row != row]
        for per_row in self.assign.values():
            per_row.pop(row, None)
        free.insert(0, row)
