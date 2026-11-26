# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``executors.pytorch``: the worker's one executor (``executor@1``), on every torch platform.
Adapted from SGLang's ``ModelRunner`` (srt/model_executor/model_runner.py, e838360;
LICENSE.sglang): the step's plan is picked first, a ready graph bucket for a decode step that
fits it, the eager plan otherwise (``_forward_raw``; stage2-contract.md §9.1), before any state
changes; then the state slots of rows starting from position 0 are cleared and snapshots
restored (``mamba_clear_indices``, copy-on-write sources), the ``ForwardBatch`` inputs are built
(``input_ids``, ``positions``), the plan runs the model with its logits processor, and the
sampler draws at the position of each row's last token (``positions`` for decode,
``seq_lens - 1`` for extend). A graph found stale runs eagerly instead. The rows, their lengths
and their state items come from the scheduler's records instead of ``ScheduleBatch``.

Known prompt continuations need no logits or sample. Those rows still retire with a zero token
count; an older plan without continuation hints retains the behavior of sampling every row.

With CUDA, ``submit`` returns once the step's work is queued (``Step``): the worker takes the
next plan while the device runs this one (one step of overlap, stage2-contract.md §4.4).

A plan without rows is a step in which this rank's scheduling domain has nothing to run while
another has (``sched.domains``): the delta is applied and nothing is sampled. A model whose ranks
compute each step together (``model.idle``: expert parallelism, doc/specs/v1-ds1017.md stage 3) takes
part in it without rows, also when the plan is refused (``Resync``): the other ranks wait for this
one in every layer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import msgspec
import torch

from core.contracts.columns import Column
from core.contracts.plan import Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan, Capacity, Resync, RowDelta, Sampled, StateOps, StepStats
from executors.pytorch.spec import Speculative, spec_draft_tokens
from platforms.base import StaleGraph


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


_EVENT_RING_SIZE = 32


class Mark:
    """A point in the device work of the steps: a timing CUDA event recorded on the stream, so it
    is passed when the device gets there; on the CPU, where that work runs inside ``submit``, the
    host clock when it is made."""

    def __init__(self, device: torch.device, event: torch.cuda.Event | None = None) -> None:
        self.event = (event if event is not None else torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
        self.at = 0.0
        if self.event is not None:
            self.event.record(torch.cuda.current_stream(device))
        else:
            self.at = time.perf_counter()

    def passed(self) -> bool:
        return self.event is None or self.event.query()

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def us_until(self, later: Mark) -> float:
        """Device time from this mark to ``later``; both passed."""
        if self.event is not None and later.event is not None:
            return self.event.elapsed_time(later.event) * 1e3
        return (later.at - self.at) * 1e6


@dataclass(frozen=True)
class Step:
    """``PendingStep``. With CUDA ``submit`` returns once the step's work is queued on the stream:
    the sampled tokens come back through pinned host memory (SGLang's ``copy_done`` event, srt/
    managers/utils.py), so the worker can take the next plan while the device runs this one. On
    the CPU the step has finished when ``submit`` returns."""

    rows: list[int]
    tokens: torch.Tensor
    """The sampled tokens on the host, excluding known continuations: filled once ``end`` is passed."""
    epoch: int
    host_us: float
    begin: Mark
    """Before the step's first device work (the persistent batch's tables)."""
    end: Mark
    """After its last (the copies after the forward)."""
    previous: Mark | None
    """The previous step's ``end``: the device sat idle from there to ``begin``."""
    logits: torch.Tensor
    """The sampled rows' last-token logits (a verify step's ``[rows * D, vocab]``); for evaluation and tests."""
    counts: torch.Tensor | None = None
    """A verify step's accepted lengths on the host (``tokens`` is then ``[rows, D]``); ``None``: one each."""
    sample_mask: list[bool] | None = None
    """Without speculation, rows that need a sampled token; ``None`` means every row."""

    errors: dict[int, str] = field(default_factory=dict[int, str])

    def ready(self) -> bool:
        return self.end.passed()

    def finish(self) -> Sampled:
        self.end.wait()
        if self.counts is None:
            counts = (Column("i32", b"\x01\x00\x00\x00" * len(self.rows)) if self.sample_mask is None
                      else Column.of("i32", [int(v) for v in self.sample_mask]))
            return Sampled(Column.of("i32", self.rows), counts,
                           Column.of("i64", self.tokens.tolist()), self.errors)
        counts = [int(n) for n in self.counts.tolist()]
        tokens = ([int(t) for t in self.tokens.tolist()] if self.tokens.ndim == 1 else
                  [int(t) for row, n in zip(self.tokens.tolist(), counts, strict=True) for t in row[:n]])
        return Sampled(Column.of("i32", self.rows), Column.of("i32", counts), Column.of("i64", tokens), self.errors)

    def stats(self) -> StepStats:
        """``device_us`` spans the step's device work, gaps between its kernels included;
        ``idle_before_us`` is the gap since the previous step's work (0 for the first)."""
        self.end.wait()
        idle = self.previous.us_until(self.begin) if self.previous is not None else 0.0
        return StepStats(self.epoch, self.host_us, self.begin.us_until(self.end), max(idle, 0.0))


def check_plan(plan: BatchPlan) -> None:
    """The rules of ``BatchPlan`` (stage2-contract.md §4.2); a plan that breaks them is a bug of
    the scheduler, not a reason to resync."""
    rows = list(plan.rows.values())
    num_tokens, positions, prev = (list(c.values()) for c in (plan.num_tokens, plan.positions, plan.from_previous))
    if not len(rows) == len(num_tokens) == len(positions) == len(prev):
        raise ValueError("a plan needs one num_tokens, position and from_previous per row")
    if len(set(rows)) != len(rows):
        raise ValueError("a row runs twice in one plan")
    if any(n < 1 for n in num_tokens) or any(p < 0 for p in positions):
        raise ValueError("every row runs at least one token from a position >= 0")
    if not rows or plan.phase not in ("prefill", "mixed", "decode"):
        raise ValueError("a plan needs rows and a valid phase")
    if plan.phase == "decode" and any(n != 1 for n in num_tokens):
        raise ValueError("decode requires exactly one token per row")
    if any(f and n != 1 for f, n in zip(prev, num_tokens, strict=True)):
        raise ValueError("a from_previous row runs exactly one token")
    if sum(n for f, n in zip(prev, num_tokens, strict=True) if not f) != len(plan.tokens):
        raise ValueError("tokens holds the input of every row without from_previous")
    if len(plan.prompt_lengths) and (plan.prompt_lengths.dtype != "i32" or len(plan.prompt_lengths) != len(rows)):
        raise ValueError("prompt_lengths has one i32 per row when present")
    if len(plan.next_tokens) and (plan.next_tokens.dtype != "i64" or len(plan.next_tokens) != len(rows)):
        raise ValueError("next_tokens has one i64 per row when present")
    if plan.tokens.dtype != "i64":
        raise ValueError("tokens is an i64 column")


def without_rows(plan: BatchPlan, excluded: set[int]) -> BatchPlan:
    """Keep the other rows and their flat token slices without changing their order."""
    keep: list[int] = []
    tokens: list[int] = []
    offset = 0
    flat = plan.tokens.values()
    for i, (row, count, previous) in enumerate(zip(plan.rows.values(), plan.num_tokens.values(),
                                                  plan.from_previous.values(), strict=True)):
        n = 0 if previous else int(count)
        if int(row) not in excluded:
            keep.append(i)
            tokens.extend(int(x) for x in flat[offset:offset + n])
        offset += n

    def select(column: Column) -> Column:
        return Column.of(column.dtype, [column.values()[i] for i in keep]) if len(column) else column

    return msgspec.structs.replace(plan, rows=select(plan.rows), num_tokens=select(plan.num_tokens),
                                   positions=select(plan.positions), tokens=Column.of("i64", tokens),
                                   from_previous=select(plan.from_previous), next_tokens=select(plan.next_tokens),
                                   prompt_lengths=select(plan.prompt_lengths))


class PyTorchExecutor:
    """``executor@1``."""

    def __init__(self, platform: Any, batch: Any, plans: PlanService, sampler: Any, stores: dict[str, Any],
                 spec: Speculative | None = None, idle: Callable[[], None] | None = None,
                 stepped: Callable[[], None] | None = None, cache: Any = None, model: Any = None) -> None:
        self.platform, self.batch, self.plans, self.sampler, self.stores = platform, batch, plans, sampler, stores
        self.cache = cache if cache is not None and cache.enabled else None
        self.model = model
        self.spec = spec
        """Speculation (doc/specs/mtp.md); ``None`` off."""
        self.idle = idle
        """The model's part in a step this rank has no rows in (``model.idle``); ``None``: none."""
        self.stepped = stepped
        """What the model does after each forward, outside any graph (``model.stepped``); ``None``: nothing."""
        self.last_end: Mark | None = None
        """The end of the device work of the step submitted last."""
        self.row_index: torch.Tensor | None = None
        """``0 .. n - 1`` on the device, as long as the largest decode step so far."""
        self._dec_h2d_pin: torch.Tensor | None = None
        self._dec_h2d_dev: torch.Tensor | None = None
        self._pref_tok_pin: torch.Tensor | None = None
        self._d2h_ring: torch.Tensor | None = None
        self._event_ring: list[torch.cuda.Event] = []
        self._event_cursor = 0
        if any("cuda" in getattr(d, "capabilities", ()) for d in platform.devices()) and torch.get_num_threads() > 8:
            torch.set_num_threads(8)

    def _mark(self, device: torch.device) -> Mark:
        if device.type != "cuda":
            return Mark(device)
        if not self._event_ring:
            self._event_ring = [torch.cuda.Event(enable_timing=True) for _ in range(_EVENT_RING_SIZE)]
            self._event_cursor = 0
        ev = self._event_ring[self._event_cursor]
        self._event_cursor = (self._event_cursor + 1) % _EVENT_RING_SIZE
        return Mark(device, ev)

    def capacity(self) -> Capacity:
        return Capacity({kind: store.capacity() for kind, store in self.stores.items()})

    def synchronize(self) -> None:
        self.platform.synchronize(0)

    def submit(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps) -> Step | Resync:
        start = time.perf_counter()
        if self.spec is not None:
            self.spec.settle()  # where the last verify left its rows (waits for it, not for its draft extend)
        elif plan.draft_tokens:
            raise ValueError("a speculative plan reached a worker with speculation off (spec.algorithm)")
        if not len(plan.rows):
            return self._empty(epoch, delta, plan, state, start)
        check_plan(plan)
        reason = self.batch.check(delta, plan)
        if reason:
            if self.idle is not None:
                self.idle()  # the step runs on the other ranks, which compute it with this one
            return Resync(reason)
        # A speculative step picks its plans itself (PlanService.verify), after the rows' positions.
        chosen, eager = self._plans(plan) if self.spec is None else (self.plans.get(PlanKey("prefill", 0)),) * 2
        last_token = self.batch.binding.buffers()["last_token"]
        device = last_token.device
        begin = self._mark(device)
        self.batch.apply(delta, state)
        try:
            if self.cache is not None:
                return self._cached(epoch, delta, plan, state, chosen, eager, last_token, begin, start)
            if any(a.cache is not None for a in delta.added) or state.save:
                raise ValueError("cache instructions reached a worker without a cache service")
            return self._run(epoch, plan, state, chosen, eager, last_token, begin, start)
        except BaseException:
            # The batch is changed and the step did not finish: a step sent after this one (with
            # sched.overlap) would run on half-changed tables. Refuse those until a full delta.
            self.batch.invalidate()
            raise

    def _cached(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps, chosen: Plan, eager: Plan,
                last_token: torch.Tensor, begin: Mark, start: float) -> Step:
        assert self.cache is not None
        errors: dict[int, str] = {}
        for added in delta.added:
            if added.cache is not None:
                try:
                    self.cache.load(added.cache, self.model, self.batch, added.row)
                except Exception as exc:
                    errors[added.row] = f"cache restore failed: {exc}"
        original = [int(r) for r in plan.rows.values()]
        working = without_rows(plan, set(errors)) if errors else plan
        if len(working.rows):
            step = self._run(epoch, working, state, chosen, eager, last_token, begin, start)
        else:
            end = self._mark(last_token.device)
            step = Step([], torch.empty(0, dtype=torch.int64), epoch, 0, begin, end, self.last_end,
                        torch.empty(0), torch.empty(0, dtype=torch.int32))
            self.last_end = end
        for save in state.save:
            if save.row not in errors:
                try:
                    self.cache.save(save.ref, self.model, self.batch, save.row)
                except Exception as exc:
                    # A failed backup never invalidates the live request's own state.
                    self.cache.note("publish_failed", str(exc))
        if errors:
            result = step.finish()
            counts = dict(zip((int(r) for r in result.rows.values()), (int(n) for n in result.counts.values()), strict=True))
            step = replace(step, rows=original, counts=torch.tensor([counts.get(r, 0) for r in original]),
                           sample_mask=None, errors=errors)
        return replace(step, host_us=(time.perf_counter() - start) * 1e6)

    def _empty(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps, start: float) -> Step | Resync:
        """A step without rows on this rank."""
        reason = self.batch.check(delta, plan)
        device = self.batch.binding.buffers()["last_token"].device
        begin = self._mark(device)
        if not reason:
            self.batch.apply(delta, state)
        try:
            if self.idle is not None:
                self.idle()
        except BaseException:
            self.batch.invalidate()
            raise
        if reason:
            return Resync(reason)
        end, previous = self._mark(device), self.last_end
        self.last_end = end
        return Step([], torch.empty(0, dtype=torch.int64), epoch, (time.perf_counter() - start) * 1e6, begin, end,
                    previous, torch.empty(0))

    def _run(self, epoch: int, plan: BatchPlan, state: StateOps, chosen: Plan, eager: Plan, last_token: torch.Tensor,
             begin: Mark, start: float) -> Step:
        device = last_token.device
        rows = [int(r) for r in plan.rows.values()]
        num_tokens = [int(n) for n in plan.num_tokens.values()]
        positions = [int(p) for p in plan.positions.values()]
        sample_at: list[int] | None = None
        sample_mask: list[bool] | None = None
        if self.spec is None and plan.phase != "decode" and len(plan.next_tokens):
            sample_mask = [int(token) < 0 for token in plan.next_tokens.values()]
            if not all(sample_mask):
                sample_at = [i for i, wanted in enumerate(sample_mask) if wanted]
            else:
                sample_mask = None

        # mamba_clear_indices: a row that starts from position 0 has no state yet (its conv runs
        # without an initial state); a row restored from a snapshot or continuing keeps its items.
        restored = {(c.kind, int(d)) for c in state.copy if c.when == "before" for d in c.dst.values()}
        # A from_previous row goes on from its last step: with speculation its plan position is only
        # where it is at least, 0 while its prefill is still in flight (doc/specs/mtp.md).
        from_prev = plan.from_previous.values()
        starting = [r for r, p, f in zip(rows, positions, from_prev, strict=True) if p == 0 and not f]
        if starting:
            for kind, store in self.stores.items():
                ids = self.batch.ids(kind)
                fresh = [i for r in starting for i in ids.get(r, []) if (kind, i) not in restored]
                if fresh:
                    store.clear(fresh)
        self._copies(state, "before")

        rows_dev_cached: torch.Tensor | None = None
        end_dev: torch.Tensor | None = None
        if plan.phase == "decode" and not len(plan.tokens) and all(n == 1 for n in num_tokens) and all(from_prev):
            if device.type == "cuda":
                n_rows = len(rows)
                cap = max(n_rows, int(getattr(self.batch, "max_rows", n_rows)), 128)
                if (self._dec_h2d_pin is None or self._dec_h2d_dev is None
                        or self._dec_h2d_dev.device != device or self._dec_h2d_pin.shape[2] < n_rows):
                    self._dec_h2d_pin = torch.empty((8, 2, cap), dtype=torch.int64, pin_memory=True)
                    self._dec_h2d_dev = torch.empty((2, cap), dtype=torch.int64, device=device)
                pin_slot = self._dec_h2d_pin[epoch & 7][:, :n_rows]
                pin_slot.copy_(torch.tensor([[r + 1 for r in rows], positions], dtype=torch.int64))
                dev_slot = self._dec_h2d_dev[:, :n_rows]
                dev_slot.copy_(pin_slot, non_blocking=True)
                rows_dev_cached, pos = dev_slot[0], dev_slot[1]
            else:
                rows_dev_cached = on_device(torch.tensor([r + 1 for r in rows], dtype=torch.int64), device)
                pos = on_device(torch.tensor(positions, dtype=torch.int64), device)
            input_ids = last_token[rows_dev_cached]
        else:
            n_tok = len(plan.tokens)
            if n_tok == 0:
                given = torch.empty(0, dtype=torch.int64, device=device)
            elif device.type == "cuda" and n_tok <= 65536:
                if self._pref_tok_pin is None or self._pref_tok_pin.shape[1] < n_tok:
                    self._pref_tok_pin = torch.empty((4, max(n_tok, 65536)), dtype=torch.int64, pin_memory=True)
                pin_tok = self._pref_tok_pin[epoch & 3][:n_tok]
                mv = memoryview(plan.tokens.data)
                pin_tok.copy_(torch.frombuffer(mv if not mv.readonly else bytearray(mv), dtype=torch.int64))
                given = pin_tok.to(device, non_blocking=True)
            else:
                given = on_device(torch.frombuffer(bytearray(plan.tokens.data), dtype=torch.int64), device)
            if not any(from_prev):
                input_ids = given
            else:
                parts: list[torch.Tensor] = []
                offset = 0
                for r, n, prev in zip(rows, num_tokens, from_prev, strict=True):
                    if prev:
                        parts.append(last_token[r + 1:r + 2])
                    else:
                        parts.append(given[offset:offset + n])
                        offset += n
                input_ids = torch.cat(parts)
            if device.type == "cuda" and len(rows) == 1:
                pos = torch.arange(positions[0], positions[0] + num_tokens[0], dtype=torch.int64, device=device)
            elif device.type == "cuda":
                nt_p0 = on_device(torch.tensor([num_tokens, positions], dtype=torch.int64), device)
                nt_dev, p0_dev = nt_p0[0], nt_p0[1]
                total_tok = sum(num_tokens)
                end_dev = torch.cumsum(nt_dev, 0)
                base_dev = end_dev - nt_dev
                seq_dev = torch.repeat_interleave(torch.arange(len(rows), device=device), nt_dev, output_size=total_tok)
                pos = p0_dev[seq_dev] + torch.arange(total_tok, dtype=torch.int64, device=device) - base_dev[seq_dev]
            else:
                pos = on_device(torch.cat([torch.arange(p, p + n) for p, n in zip(positions, num_tokens, strict=True)]),
                                device)
        if plan.phase == "decode":  # one token per row: each row's last token is its only one
            if self.row_index is None or len(self.row_index) < len(rows):
                self.row_index = torch.arange(max(len(rows), int(getattr(self.batch, "max_rows", len(rows)))),
                                              device=device)
            last_index = self.row_index[:len(rows)]
        elif device.type == "cuda" and len(rows) == 1 and sample_at is None:
            last_index = torch.full((1,), num_tokens[0] - 1, dtype=torch.int64, device=device)
        elif end_dev is not None and sample_at is None:
            last_index = end_dev - 1
        else:
            last = torch.cumsum(torch.tensor(num_tokens), dim=0) - 1
            if sample_at is not None:
                last = last[torch.tensor(sample_at, dtype=torch.int64)]
            last_index = on_device(last, device)

        counts: torch.Tensor | None = None
        if self.spec is not None and plan.phase == "decode":
            next_token_ids, _, done = self.spec.decode(plan, rows, positions, input_ids,
                                                            self.batch.track("gdn.slots"))
            counts, logits = done.lengths, next_token_ids
        elif self.spec is not None:
            next_token_ids, logits = self.spec.prefill(plan, rows, num_tokens, positions, input_ids, pos, last_index)
        else:
            try:
                logits = chosen.run(self.batch, plan, input_ids, pos, last_index)
            except StaleGraph:
                if chosen is eager or self.plans.required:
                    raise
                logits = eager.run(self.batch, plan, input_ids, pos, last_index)
            if self.stepped is not None:
                self.stepped()
            sample_rows = rows if sample_at is None else [rows[i] for i in sample_at]
            sample_pos = pos if plan.phase == "decode" else pos[last_index]
            if sample_rows:
                next_token_ids = self.sampler.sample(logits, self.batch.sampling(sample_rows), sample_pos)
                if sample_at is None and rows_dev_cached is not None and hasattr(self.batch, "record_dev"):
                    self.batch.record_dev(sample_rows, rows_dev_cached, next_token_ids)
                else:
                    self.batch.record(sample_rows, next_token_ids)
            else:
                next_token_ids = torch.empty(0, dtype=torch.int32, device=device)
        tokens = next_token_ids
        if device.type == "cuda":  # async_d2h: into pinned memory, without waiting for the device
            if counts is None and next_token_ids.ndim == 1:
                n_out = next_token_ids.shape[0]
                cap_out = max(n_out, int(getattr(self.batch, "max_rows", n_out)), 128)
                if (self._d2h_ring is None or self._d2h_ring.dtype != next_token_ids.dtype
                        or self._d2h_ring.shape[1] < n_out):
                    self._d2h_ring = torch.empty((8, cap_out), dtype=next_token_ids.dtype, pin_memory=True)
                tokens = self._d2h_ring[epoch & 7][:n_out]
                tokens.copy_(next_token_ids, non_blocking=True)
            else:
                tokens = torch.empty(next_token_ids.shape, dtype=next_token_ids.dtype, pin_memory=True)
                tokens.copy_(next_token_ids, non_blocking=True)
        self._copies(state, "after")
        end, previous = self._mark(device), self.last_end
        self.last_end = end
        host_us = (time.perf_counter() - start) * 1e6
        return Step(rows, tokens, epoch, host_us, begin, end, previous, logits, counts, sample_mask)

    def _plans(self, plan: BatchPlan) -> tuple[Plan, Plan]:
        """The step's plan and the eager one. A decode step takes the smallest ready graph bucket
        that holds its rows; any other step, or one no graph holds, the eager plan (bucket 0),
        unless the plans require graphs. The scheduler's ``bucket`` is only a hint."""
        phase = "decode" if plan.phase == "decode" else "prefill"
        eager = self.plans.get(PlanKey(phase, 0))
        fits = [b for b in self.plans.buckets(phase) if b >= len(plan.rows)] if phase == "decode" else []
        if phase == "decode" and not fits and self.plans.required:
            raise StaleGraph(f"no ready decode graph holds {len(plan.rows)} rows, and graphs are required; "
                             "no state was changed")
        return (self.plans.get(PlanKey(phase, min(fits))) if fits else eager), eager

    def _copies(self, state: StateOps, when: str) -> None:
        for c in state.copy:
            if c.when == when:
                self.stores[c.kind].copy([int(x) for x in c.src.values()], [int(x) for x in c.dst.values()])


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        model = ctx.use("model@1")
        stores = dict(model.stores)  # every kind the model's layers store
        batch, plans, sampler = ctx.use("worker.batch@1"), ctx.use("plan@1"), ctx.use("layers.sampler@1")
        draft_tokens = spec_draft_tokens(config)
        spec = None
        if draft_tokens:
            if not getattr(model, "draft_tokens", 0):
                raise ValueError(f"spec.algorithm = {config['spec.algorithm']!r}: the model drafts no tokens")
            spec = Speculative(plans, batch, sampler, draft_tokens, model.hidden_size, model.dtype)
        eager_idle = getattr(model, "idle", None)
        replay_idle = getattr(plans, "replay_idle", None)

        def idle() -> None:
            if eager_idle is not None and (replay_idle is None or not replay_idle()):
                eager_idle()

        ctx.provide("executor@1", PyTorchExecutor(ctx.platform, batch, plans, sampler, stores, spec,
                                                  idle if eager_idle is not None else None, getattr(model, "stepped", None),
                                                  ctx.get("kv.cache@1"), model))
