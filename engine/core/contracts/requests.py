"""Per-step records between the frontend and the scheduler host (stage2-contract.md §4.2).
Each class is one section of an ``OpenRecord`` (doc/specs/ipc.md §3), named by ``SECTION``;
a receiver decodes the sections it knows. The records between the scheduler host and the
workers are in ``core.contracts.records``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import msgspec

from core.contracts.cache import CacheRef
from core.contracts.columns import Column
from core.contracts.sampling import SamplingParams

REQUESTS = "frontend.requests@1"
"""Schema of the frontend -> scheduler record."""
TOKENS = "sched.tokens@1"
"""Schema of the scheduler -> frontend record."""

ENDED_BY_CHANGE = "ended: the engine is being changed (apply --force)"
"""The error (``TokenEvents.errors``) of a request a change ended at once: the server's doing, which
the frontend answers with 503 and ``Retry-After`` (doc/specs/empty-apply.md). Recognised by its text,
whichever of the scheduler and the frontend saw the change first."""


class NewRequest(msgspec.Struct, frozen=True):
    request: int
    """Assigned by the frontend; unique while the request lives."""
    prompt: Column
    """``i64`` token ids."""
    sampling: SamplingParams = SamplingParams()
    max_tokens: int = 256
    stop_token_ids: list[int] = []
    cache: CacheRef | None = None
    """An optional published prefix, restored before the first forward."""


class AbortRequest(msgspec.Struct, frozen=True):
    request: int
    reason: str = ""
    """E.g. a stop string the detokenizer found, or the client went away."""


class Requests(msgspec.Struct, frozen=True):
    SECTION: ClassVar[str] = "frontend.batch@1"
    new: list[NewRequest] = []
    abort: list[AbortRequest] = []


type FinishReason = Literal["stop", "length", "abort", "version_switch"]


class TokenEvents(msgspec.Struct, frozen=True):
    """New tokens of one step for the frontend; ``requests[i]`` owns
    ``tokens[offsets[i]:offsets[i + 1]]``. With overlap a request may get one token past
    its stop condition; the scheduler drops it before it gets here."""

    SECTION: ClassVar[str] = "sched.events@1"
    identity: str
    """The identity the step ran under (the ``x-engine-identity`` header, v0-plan 5.8)."""
    requests: Column
    offsets: Column
    tokens: Column
    finished: dict[int, FinishReason] = {}
    errors: dict[int, str] = {}
    """Requests the scheduler refused, with why; they also appear in ``finished`` (``abort``)."""
