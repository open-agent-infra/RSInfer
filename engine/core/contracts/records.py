"""Per-step records between the scheduler host and the workers (stage2-contract.md §4).
Each class is one section of an ``OpenRecord`` (doc/specs/ipc.md §3), named by ``SECTION``;
a receiver decodes the sections it knows. The records between the frontend and the
scheduler host are in ``core.contracts.requests``.

The epoch of a record is its step number, assigned by the scheduler (change.md 4.2 step 4).
The scheduler may issue step ``N + 1`` before it has the result of step ``N`` (one step of
overlap, §4.4), so nothing in a plan may need a value only the result of the previous step
carries: a decode input token is named, not given (``BatchPlan.from_previous``).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import msgspec

from core.contracts.cache import CacheRef, CacheSave
from core.contracts.columns import Column
from core.contracts.sampling import SamplingParams

STEP_PLAN = "sched.step_plan@1"
"""Schema of the scheduler -> worker record, one per rank (P3)."""
STEP_RESULT = "worker.step_result@1"
"""Schema of the worker -> scheduler record."""


# ------------------------------------------------------------------ scheduler -> worker


class AddedRow(msgspec.Struct, frozen=True):
    row: int
    """Index into the worker's persistent batch (``worker.batch``)."""
    request: int
    sampling: SamplingParams = SamplingParams()
    cache: CacheRef | None = None


class RowDelta(msgspec.Struct, frozen=True):
    """How the persistent batch changes before this step. Deltas arrive in order on a
    lossless stream; ``full`` drops every row first (the answer to ``Resync``). A full delta
    re-adds every row the scheduler keeps, and the plan's ``StateOps`` assigns each of them
    every item it holds, not only new ones."""

    SECTION: ClassVar[str] = "sched.rows@1"
    added: list[AddedRow] = []
    removed: Column = Column("i32")
    full: bool = False


class BatchPlan(msgspec.Struct, frozen=True):
    """What every row computes in this step. ``num_tokens``, ``positions`` and ``from_previous``
    hold one value per row of ``rows``. Checking a plan against these rules is the worker's
    code, not part of the contract (stage2-contract.md §3, §4.2)."""

    SECTION: ClassVar[str] = "sched.batch@1"
    rows: Column
    """``i32``: rows of the persistent batch that run in this step."""
    num_tokens: Column
    """``i32`` per row: prompt tokens of this chunk, or 1 for a decode step (also with
    speculation: the one input token is the last one accepted, the drafts are the worker's)."""
    positions: Column
    """``i32`` per row: position of the row's first token in this step. With speculation, for a
    ``from_previous`` decode row only where it is at least: the steps before it may still be on
    their way, and how far they went is the worker's (doc/specs/mtp.md §2.2)."""
    tokens: Column
    """``i64``, flat: the input tokens of the rows whose ``from_previous`` is 0, in row order:
    such a row takes the next ``num_tokens`` of them, a ``from_previous`` row none. So its
    length is the sum of those rows' ``num_tokens``."""
    from_previous: Column
    """``u8`` per row: 1 = the row's single input token is the token the worker sampled for it
    the last time the row ran, resolved on the device without a host round trip. Only for a
    row that has run since it was last added (a full delta adds every row again) and whose
    last run ended its prompt or decoded; otherwise its token is given in ``tokens``. A worker
    with no token to resolve answers ``Resync`` and runs nothing."""
    phase: Literal["prefill", "decode", "mixed"]
    bucket: int
    """The scheduler's bucket hint (padded batch size); 0 = eager. The worker picks the plan it runs
    itself, from those ready (stage2-contract.md §9.1)."""
    draft_tokens: int = 0
    """Speculation (doc/specs/mtp.md): 0 without; else SGLang's ``num_draft_tokens`` D. Each decode
    row then verifies D tokens, its input token and D - 1 drafts, and advances by as many as are
    accepted (``Sampled.counts``), 1 to D; a prefill row runs the draft model's extend after the
    target's. Such a step is not mixed."""
    next_tokens: Column = Column("i64")
    """Optional ``i64`` per row: the known prompt token after the row's chunk, or -1 when the row
    needs a sampled token. A non-speculative worker can skip logits and sampling for a known
    continuation. Empty preserves the older behavior of sampling every row. With ``draft_tokens``,
    the draft model's extend takes it as the chunk's last input (SGLang's
    ``chunked_req_next_prompt_token``)."""

    prompt_lengths: Column = Column("i32")
    """Optional original prompt lengths, one per row. Bounded decoder replay uses the entire
    prompt suffix across chunk boundaries. Empty: older producers use full decoder replay."""


class StateAssign(msgspec.Struct, frozen=True):
    """Items of one state kind newly given to rows: pages for paged kinds, a slot at admission
    for slot kinds. Row ``rows[i]`` appends ``ids[offsets[i]:offsets[i + 1]]``. Ids are the
    ledger's, ``0 .. n - 1`` for the kind's ``n`` in ``Capacity``."""

    kind: str
    rows: Column
    offsets: Column
    ids: Column


class StateCopy(msgspec.Struct, frozen=True):
    """Item copies of one state kind: restore a snapshot into a row's slot on a prefix hit
    (``before`` the step, from ``admit``), or snapshot a slot at a block boundary (``after``
    it, from ``snapshot``). ``after`` sees only the state at the end of the row's chunk in
    this step: a snapshot boundary the chunk crosses inside it cannot be copied (§5.4)."""

    kind: str
    when: Literal["before", "after"]
    src: Column
    dst: Column


class StateTrack(msgspec.Struct, frozen=True):
    """For the prefix cache (prefix-cache.md, SGLang's ``mamba_track_*``): row ``rows[i]`` writes its
    state at position ``upto[i]`` (exclusive; inside this step's input or at its end) into item
    ``dst[i]``. A position inside the input is a multiple of the kind's chunk (64 tokens for GDN)
    from where the row starts in this step. ``upto[i] = -g``, for a decode row of a speculative step:
    at the multiple of ``g`` its accepted tokens cross, if they cross one (SGLang's
    ``mamba_steps_to_track`` over its ``mamba_track_grid``); the worker finds it on the device, the
    scheduler from ``Sampled``."""

    kind: str
    rows: Column
    dst: Column
    upto: Column


class StateOps(msgspec.Struct, frozen=True):
    """Every state kind's part of a step plan, in one section."""

    SECTION: ClassVar[str] = "state.ops@1"
    assign: list[StateAssign] = []
    copy: list[StateCopy] = []
    track: list[StateTrack] = []
    save: list[CacheSave] = []


# ------------------------------------------------------------------ worker -> scheduler


class Sampled(msgspec.Struct, frozen=True):
    """Tokens sampled in step ``epoch`` (the record's epoch). ``counts`` is 1 per row without
    speculation, the accepted length plus one with it.

    Every row of the plan samples, also a row whose chunk does not end its prompt; the scheduler
    drops that token, as sglang and mini-sglang drop a chunked request's (2026-11-08). The
    worker keeps every row's last token for ``from_previous`` regardless."""

    SECTION: ClassVar[str] = "worker.sampled@1"
    rows: Column
    counts: Column
    tokens: Column
    errors: dict[int, str] = {}
    """Rows whose restore failed before forward; zero count, no output, other rows still run."""


class StepStats(msgspec.Struct, frozen=True):
    """Where the time of one step went; how an agent sees that the CPU is the bottleneck.
    Device times come from events, so they may arrive one record late (``step``)."""

    SECTION: ClassVar[str] = "worker.stats@1"
    step: int
    host_us: float
    """Worker CPU time for the step: apply the delta, prepare, launch."""
    device_us: float
    idle_before_us: float
    """How long the device sat idle between the previous step's work and this one's."""


class Capacity(msgspec.Struct, frozen=True):
    """Usable items per state kind: the ledger hands out ids ``0 .. n - 1``, and a store keeps its
    padding and scratch items outside that range (``StateStore``). Sent at start, and again once a
    supported capacity change has taken effect."""

    SECTION: ClassVar[str] = "state.capacity@1"
    items: dict[str, int]
    identity: str = ""
    """The identity the worker served when it sent this (``""``: unknown, an older sender). After
    a change the scheduler plans again only once every rank sent one under the new identity: an
    older one may be on its way still, with the capacity of stores the change rebuilt
    (doc/specs/empty-apply.md)."""


class Resync(msgspec.Struct, frozen=True):
    """The worker lost its persistent batch (``worker.batch`` was rebuilt), or got a plan it
    cannot apply to it: send ``full``.

    Until a full delta arrives the worker runs no plan: it answers each with ``Resync`` and no
    ``Sampled``. The scheduler counts every step answered so as not run (with overlap, the next
    one may already be on its way): its tokens were not computed and its copies not made, the
    requests' lengths go back to the last step that returned ``Sampled``. Items assigned in
    those steps stay with their requests; the full delta assigns them again. Further
    ``Resync`` answers, to plans sent before the full delta, are expected."""

    SECTION: ClassVar[str] = "worker.resync@1"
    reason: str = ""


class StepFailure(msgspec.Struct, frozen=True):
    """The worker failed while running step ``epoch`` (the record's epoch), after it began, so it
    cannot say what the device did. Unlike ``Resync`` (a plan refused before anything ran), the
    step is not taken as not run: the scheduler stops sending this rank's domain steps, answers
    new requests with an error and reports it. Nothing is frozen; what happens to the requests
    it holds is for whoever changes the engine next to choose."""

    SECTION: ClassVar[str] = "worker.failure@1"
    reason: str
