# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``worker.batch``: the persistent batch (``worker.batch@1``, stage2-contract.md §4.3).

Its tables are SGLang's ``HybridReqToTokenPool`` (srt/mem_cache/memory_pool.py, e838360;
LICENSE.sglang): ``req_to_token`` ``[rows + 1, max_context_len]`` int32 token slots, with a padding
row at index 0, and per slot kind the row -> item mapping (``req_index_to_mamba_index_mapping``).
Row ``r`` of the scheduler's records is index ``r + 1`` (SGLang's ``req_pool_idx``), and item ``i``
page or slot ``i + 1`` (the ones SGLang's allocators hand out; 0 is padding). The rows
themselves and their sampling parameters are what the scheduler sent (``RowDelta``); the tokens
sampled last, which ``from_previous`` names, are kept on the device.
"""

from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import torch

from core.contracts.records import BatchPlan, RowDelta, StateAssign, StateOps
from core.contracts.sampling import SamplingParams

PAGED_KIND = "kv.paged"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)
"""The kind whose items are pages of ``page_size`` token slots; every other kind is one item per row."""


@dataclass
class Row:
    request: int
    sampling: SamplingParams
    ids: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    has_previous: bool = False
    """A token was sampled for the row since it was added: ``from_previous`` may name it."""
    committed: int | None = None
    """Speculation: where its last step left the row; the scheduler knows less while steps are in flight."""
    has_drafts: bool = False
    """Speculation: ``spec.drafts`` and ``spec.hidden`` hold the row's own, from its last step."""


@dataclass(frozen=True)
class Tables:
    """``Binding``."""

    generation: int
    tensors: Mapping[str, torch.Tensor]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self.tensors


@dataclass
class PersistentBatch:
    """``worker.batch@1``."""

    platform: Any
    max_rows: int
    max_context_len: int
    page_size: int
    rows: dict[int, Row] = field(default_factory=dict[int, Row])
    tensors: dict[str, torch.Tensor] = field(default_factory=dict[str, torch.Tensor])
    allocated: list[Any] = field(default_factory=list[Any])
    spec: tuple[int, int, str] | None = None
    """What ``prepare_spec`` last made the speculation tables for."""
    spec_buffers: list[Any] = field(default_factory=list[Any])
    generation: int = 0
    rebuilt: bool = True
    """Built, or left half changed by a step that failed (``invalidate``), and not yet given a full
    delta: it refuses every other one (``Resync``)."""
    tracks: dict[str, tuple[list[int], list[int], list[int]]] = field(
        default_factory=dict[str, tuple[list[int], list[int], list[int]]])
    """The step's ``StateOps.track`` per kind: rows, destination items, positions."""
    _page_cols: tuple[torch.device, torch.Tensor] | None = None
    _expand_pages_on_device: bool = False

    def __post_init__(self) -> None:
        # +1 padding row at index 0: padded batches read and write there harmlessly.
        self._alloc("req_to_token", (self.max_rows + 1, self.max_pages * self.page_size), "i32")
        self._alloc("last_token", (self.max_rows + 1,), "i64")
        self._expand_pages_on_device = self.tensors["req_to_token"].is_cuda and find_spec("triton") is not None

    @property
    def max_pages(self) -> int:
        """Pages of one row: whole pages for ``max_context_len`` tokens, so any request of at most
        that many tokens fits."""
        return -(-self.max_context_len // self.page_size)

    def _alloc(self, name: str, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
        buf = self.platform.alloc(0, 0, shape=shape, dtype=dtype)
        self.allocated.append(buf)
        self.tensors[name] = self.platform.tensor(buf)
        return self.tensors[name]

    @property
    def binding(self) -> Tables:
        return Tables(self.generation, dict(self.tensors))

    def check(self, delta: RowDelta, plan: BatchPlan) -> str:
        if any(not 0 <= int(r) < self.max_rows for r in plan.rows.values()):
            raise ValueError("plan row is outside batch.max_rows")
        if any(int(p) + int(n) > self.max_context_len
               for p, n in zip(plan.positions.values(), plan.num_tokens.values(), strict=True)):
            raise ValueError("plan exceeds batch.max_context_len")
        if plan.draft_tokens and plan.phase == "decode":  # the verify window, from where each row is
            for r, p, prev in zip((int(x) for x in plan.rows.values()), plan.positions.values(),
                                  plan.from_previous.values(), strict=True):
                row = self.rows.get(r)
                at = row.committed if prev and row is not None and row.committed is not None else int(p)
                if at + plan.draft_tokens > self.max_context_len:
                    raise ValueError(f"row {r}'s verify window from {at} exceeds batch.max_context_len")
        if self.rebuilt and not delta.full:
            return "the persistent batch was rebuilt (or a step failed after changing it) and waits for a full delta"
        known = set() if delta.full else set(self.rows)
        for r in (int(x) for x in delta.removed.values()):
            if r not in known:
                return f"row {r} is removed but not in the persistent batch"
            known.discard(r)
        added = {a.row for a in delta.added}
        known |= added
        for r, prev in zip((int(x) for x in plan.rows.values()), plan.from_previous.values(), strict=True):
            if r not in known:
                return f"row {r} is not in the persistent batch"
            if prev and (r in added or not self.rows[r].has_previous):
                return f"row {r} names its previous token, which this batch does not have"
        return ""

    def apply(self, delta: RowDelta, state: StateOps) -> None:
        if not delta.full and not len(delta.removed) and not delta.added and not state.assign:
            self.tracks = ({t.kind: ([int(x) for x in t.rows.values()], [int(x) for x in t.dst.values()],
                                     [int(x) for x in t.upto.values()]) for t in state.track}
                           if state.track else {})
            return
        self._within_limits(delta, state)
        if delta.full:
            self.rows.clear()
            self.rebuilt = False
        for r in (int(x) for x in delta.removed.values()):
            del self.rows[r]
        for a in delta.added:
            self.rows[a.row] = Row(a.request, a.sampling)
        for assign in state.assign:
            self._assign_batch(assign)
        self.tracks = {t.kind: ([int(x) for x in t.rows.values()], [int(x) for x in t.dst.values()],
                                [int(x) for x in t.upto.values()]) for t in state.track}

    def track(self, kind: str) -> tuple[list[int], list[int], list[int]]:
        """This step's ``StateTrack`` of ``kind``: rows, destination items, positions."""
        return self.tracks.get(kind, ([], [], []))

    def _within_limits(self, delta: RowDelta, state: StateOps) -> None:
        """What ``apply`` cannot do, found before it changes anything: a row outside the batch,
        items for a row it does not have, more pages than one row holds."""
        pages = {} if delta.full else {r: len(row.ids.get(PAGED_KIND, [])) for r, row in self.rows.items()}
        for r in (int(x) for x in delta.removed.values()):
            pages.pop(r, None)
        for a in delta.added:
            if not 0 <= a.row < self.max_rows:
                raise ValueError(f"row {a.row} is outside the batch of {self.max_rows} rows")
            pages[a.row] = 0
        for assign in state.assign:
            offsets = [int(x) for x in assign.offsets.values()]
            for i, r in enumerate(int(x) for x in assign.rows.values()):
                if r not in pages:
                    raise ValueError(f"row {r} is given {assign.kind} items but is not in the batch")
                if assign.kind == PAGED_KIND:
                    pages[r] += offsets[i + 1] - offsets[i]
                    if pages[r] > self.max_pages:
                        raise ValueError(f"row {r} would hold {pages[r]} pages of {self.page_size} tokens, more than "
                                         f"{self.max_context_len} tokens")

    def prepare(self, kinds: Sequence[str]) -> None:
        """Allocate slot tables before capture; never replace an existing table."""
        for kind in kinds:
            name = f"slots.{kind}"
            if kind != PAGED_KIND and name not in self.tensors:
                self._alloc(name, (self.max_rows + 1,), "i32")
                self.generation += 1

    def _page_arange(self, device: torch.device) -> torch.Tensor:
        cur = self._page_cols
        if cur is None or cur[0] != device:
            cur = self._page_cols = (device, torch.arange(self.page_size, dtype=torch.int32, device=device))
        return cur[1]

    def _assign_batch(self, assign: StateAssign) -> None:
        if (self._expand_pages_on_device and assign.kind == PAGED_KIND
                and assign.ids.dtype == "i32" and len(assign.ids) >= 64):
            from kernels.page_assignment import assign_pages

            ids_mv = memoryview(assign.ids.data).cast("i")
            offsets_v = assign.offsets.values()
            packet = array("i")
            longest = 0
            for i, row in enumerate(assign.rows.values()):
                begin, end = int(offsets_v[i]), int(offsets_v[i + 1])
                if begin != end:
                    held = self.rows[int(row)].ids.setdefault(PAGED_KIND, [])
                    packet.extend((int(row) + 1, len(held), begin, end - begin))
                    held.extend(ids_mv[begin:end])
                    longest = max(longest, end - begin)
            if longest:
                nrows = len(packet) // 4
                packet.frombytes(assign.ids.data)
                table = self.tensors["req_to_token"]
                packed = on_device(torch.frombuffer(packet, dtype=torch.int32), table.device)
                assign_pages(table, packed, nrows, longest, self.page_size)
            return
        r_list = [int(x) for x in assign.rows.values()]
        if not r_list:
            return
        offsets = [int(x) for x in assign.offsets.values()]
        ids = [int(x) for x in assign.ids.values()]
        if assign.kind == PAGED_KIND:
            spans: list[tuple[int, int, int]] = []
            cu = 0
            for i, r in enumerate(r_list):
                a, b = offsets[i], offsets[i + 1]
                cnt = b - a
                held = self.rows[r].ids.setdefault(PAGED_KIND, [])
                if cnt > 0:
                    spans.append(((r + 1) * self.max_pages + len(held), cnt, cu))
                    cu += cnt
                    held.extend(ids[a:b])
            if not cu:
                return
            r2t = self.tensors["req_to_token"]
            device = r2t.device
            pages_host = torch.frombuffer(bytearray(assign.ids.data), dtype=torch.int32)[:cu]
            pages_dev = on_device(pages_host, device) + 1
            meta = on_device(torch.tensor(spans, dtype=torch.int64), device)
            seq = torch.repeat_interleave(torch.arange(len(spans), device=device), meta[:, 1], output_size=cu)
            flat_idx = meta[:, 0][seq] + torch.arange(cu, device=device) - meta[:, 2][seq]
            r2t.view(-1, self.page_size)[flat_idx] = pages_dev[:, None] * self.page_size + self._page_arange(device)
            return
        name = f"slots.{assign.kind}"
        self.prepare([assign.kind])
        slot_rows: list[int] = []
        slot_vals: list[int] = []
        for i, r in enumerate(r_list):
            a, b = offsets[i], offsets[i + 1]
            if b > a:
                held = self.rows[r].ids.setdefault(assign.kind, [])
                held.extend(ids[a:b])
                slot_rows.append(r + 1)
                slot_vals.append(ids[b - 1] + 1)
        if slot_rows:
            buf = self.tensors[name]
            pair = on_device(torch.tensor([slot_rows, slot_vals], dtype=torch.int64), buf.device)
            buf[pair[0]] = pair[1].to(buf.dtype)

    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        return {r: row.ids.get(kind, []) for r, row in self.rows.items()}

    def prepare_spec(self, draft_tokens: int, hidden_size: int, dtype: str) -> None:
        """The speculation tables; kept when asked for the same again (a reloaded executor, or
        ``core.plan`` before it), made again when ``spec.num_draft_tokens`` or the model changed."""
        if self.spec == (draft_tokens, hidden_size, dtype):
            return
        for buf in self.spec_buffers:  # what graphs recorded over them went with the plan that had them
            self.allocated.remove(buf)
            self.platform.free(buf)
        self._alloc("spec.drafts", (self.max_rows + 1, draft_tokens - 1), "i64")
        self._alloc("spec.hidden", (self.max_rows + 1, hidden_size), dtype)
        self.spec, self.spec_buffers = (draft_tokens, hidden_size, dtype), self.allocated[-2:]
        self.generation += 1

    def committed(self, row: int) -> int | None:
        return self.rows[row].committed

    def has_drafts(self, row: int) -> bool:
        return self.rows[row].has_drafts

    def speculated(self, rows: Sequence[int], committed: Sequence[int]) -> None:
        for r, at in zip(rows, committed, strict=True):
            self.rows[r].committed, self.rows[r].has_drafts = at, True

    def sampling(self, rows: Sequence[int]) -> list[SamplingParams]:
        return [self.rows[r].sampling for r in rows]

    def record(self, rows: Sequence[int], tokens: torch.Tensor) -> None:
        index = on_device(torch.tensor(list(rows), dtype=torch.int64) + 1, tokens.device)
        self.tensors["last_token"][index] = tokens.to(torch.int64)
        for r in rows:
            self.rows[r].has_previous = True

    def record_dev(self, rows: Sequence[int], index_plus1: torch.Tensor, tokens: torch.Tensor) -> None:
        self.tensors["last_token"][index_plus1] = tokens.to(torch.int64)
        for r in rows:
            self.rows[r].has_previous = True

    def mark_recorded(self, rows: Sequence[int]) -> None:
        for r in rows:
            self.rows[r].has_previous = True

    def invalidate(self) -> None:
        """A step failed after ``apply`` changed the batch: what it holds no longer matches what the
        scheduler thinks it does (rows added, their last tokens not recorded). Until a full delta
        rebuilds it, every other step is refused (``Resync``), not run on the half-changed tables."""
        self.rebuilt = True

    def free(self) -> None:
        self.generation += 1
        for buf in self.allocated:
            self.platform.free(buf)
        self.allocated.clear()
        self.tensors.clear()


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.batch = PersistentBatch(ctx.platform, config["batch.max_rows"], config["batch.max_context_len"],
                                     config["state.kv.page_size"])
        ctx.effect(lambda: self.batch.free, label="batch tables")
        ctx.provide("worker.batch@1", self.batch)
