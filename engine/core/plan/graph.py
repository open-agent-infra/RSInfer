"""Decode graphs: one per bucket of rows, recorded while the worker is paused, and what they were
recorded over. A graph is SGLang's ``DecodeCudaGraphRunner`` for one batch size: fixed inputs, the
rows padded to the bucket with the padding row, the whole forward replayed."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from core.contracts.columns import Column
from core.contracts.plan import Backend, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.tensor import Tensor
from unit.runtime.proxy import provision_of

DEFAULT_MAX_BUCKET = 64
"""Without ``plan.graph_buckets``, the largest bucket; larger steps run eagerly. On a SM100 with
the 64-layer model, steps of 16 to 64 rows that ran eagerly were 2.7 to 3.3 times slower than
replayed ones, and the buckets up to 64 kept 140 MiB more than those up to 8 (review of #51,
doc/evaluations/real-kernels-fused.md)."""

FIXED_EXTENT_MAX_BUCKET = 8
"""The same while a captured step runs a solution made for captured steps only (``CAPTURE``): a
stand-in over fixed extents for a kernel that reads the host, as the decode attention of
``kernels.torch_graph``. Its cost follows ``batch.max_context_len`` per lane, which more rows
than this did not win back (doc/evaluations/cuda-graph-review-fixes.md)."""


def default_buckets(rows: int, fixed_extent: bool) -> list[int]:
    """The buckets when none are configured. As SGLang's default decode sizes (1, 2, 4, 8, 12, then
    every 8th; ``generate_cuda_graph_batch_sizes``, e838360), up to ``batch.max_rows`` or
    ``DEFAULT_MAX_BUCKET`` and with that size itself, so that up to there a step of any size is
    replayed; over fixed extents, the powers of two."""
    if fixed_extent:
        return [1 << i for i in range(min(rows, FIXED_EXTENT_MAX_BUCKET).bit_length())]
    limit = min(rows, DEFAULT_MAX_BUCKET)
    return sorted({b for b in (1, 2, 4, 8, 12, *range(16, limit + 1, 8)) if b <= limit} | {limit})


def buckets(mode: str, values: Sequence[int], rows: int, budget: int, fixed_extent: bool) -> list[int]:
    if mode not in ("auto", "off", "required"):
        raise ValueError("plan.cuda_graph must be auto, off or required")
    if rows < 1 or budget <= 0:
        raise ValueError("batch.max_rows and plan.graph_memory_bytes must be positive")
    if any(type(b) is not int or not 0 < b <= rows for b in values) or len(set(values)) != len(values):
        raise ValueError("plan.graph_buckets must be distinct positive integers <= batch.max_rows")
    chosen = sorted(values) if values else default_buckets(rows, fixed_extent)
    if mode == "required" and chosen[-1] != rows:
        raise ValueError("required graphs must cover batch.max_rows; list its exact capacity in plan.graph_buckets")
    return chosen


def padding_row() -> BatchPlan:
    """A decode step of the padding row alone (row -1, on the stores' scratch items): a lane with
    a defined result while there are no requests. A bucket's other lanes are padding: what a
    kernel computes for a padded lane is undefined, and they all write the padding items."""
    return BatchPlan(Column.of("i32", [-1]), Column.of("i32", [1]), Column.of("i32", [0]),
                     Column.of("i64", [0]), Column.of("u8", [0]), "decode", 0)


def fetch(backends: Sequence[Backend], input_ids: Tensor, positions: Tensor) -> None:
    """``Backend.fetch`` of the backends that have one: after ``prepare``, before the forward or the
    replay, outside the graph, with the step's inputs as the forward gets them."""
    for backend in backends:
        hook = getattr(backend, "fetch", None)
        if hook is not None:
            hook(input_ids, positions)


def eager_padding_step(batch: PersistentBatch, model: Any, backends: Sequence[Backend], stores: Sequence[Any],
                       device: torch.device) -> None:
    """One eager decode step of the padding row, on the scratch items. What a kernel library
    allocates for all its steps (a workspace, a compiled module) is allocated here."""
    for store in stores:
        store.clear(store.scratch())
    for backend in backends:
        backend.prepare(batch, padding_row(), PlanKey("decode", 0))
    token = torch.zeros(1, dtype=torch.int64, device=device)
    fetch(backends, token, torch.zeros_like(token))
    with torch.inference_mode():
        model.forward(token, torch.zeros_like(token), torch.arange(1, device=device))
    for store in stores:
        store.clear(store.scratch())


class Dependencies:
    """What the graphs were recorded over, by address: the instances behind the services, the
    registrations of the kernel catalog, and one binding generation per provider (``Binding``)."""

    def __init__(self, services: Sequence[Any], bindings: Sequence[Callable[[], Any]], catalog: Any) -> None:
        self.provisions = [provision_of(s) for s in services]
        self.bindings = list(bindings)
        self.generations = [b().generation for b in bindings]
        self.catalog, self.version = catalog, catalog.version

    def valid(self) -> bool:
        # A provision carries the provider's instance and its revocation: a reloaded provider is
        # another instance, whatever its generation says.
        if any(p.revoked for p in self.provisions) or self.catalog.version != self.version:
            return False
        try:
            return all(b().generation == g for b, g in zip(self.bindings, self.generations, strict=True))
        except (KeyError, RuntimeError):
            return False  # a provider freed its buffers before its service was revoked


class GraphPlan:
    """``Plan`` for one decode bucket."""

    def __init__(self, size: int, device: torch.device, backends: Sequence[Backend]) -> None:
        self.size, self.backends = size, list(backends)
        self.inputs = torch.zeros(size, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.inputs)
        self.last_index = torch.arange(size, device=device)
        self.output: torch.Tensor | None = None
        self.graph: Any = None
        self._dirty_from = size
        self.splits_head = False

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
            last_index: Tensor) -> Tensor:
        """The step's dependency check was ``PlanService.buckets``; this only replays."""
        n = len(plan.rows)
        if plan.phase != "decode" or not 0 < n <= self.size or any(t != 1 for t in plan.num_tokens.values()):
            raise ValueError("a decode graph runs one token per row, at most its bucket of rows")
        for backend in self.backends:
            backend.prepare(batch, plan, PlanKey("decode", self.size))
        if input_ids.data_ptr() != self.inputs.data_ptr():
            self.inputs[:n].copy_(input_ids)
        if positions.data_ptr() != self.positions.data_ptr():
            self.positions[:n].copy_(positions)
        if n < self._dirty_from:
            self.inputs[n:self._dirty_from].zero_()
            self.positions[n:self._dirty_from].zero_()
        self._dirty_from = n
        fetch(self.backends, self.inputs, self.positions)
        self.graph.replay()
        assert self.output is not None
        return self.output[:n]

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output = None, None
        self._dirty_from = self.size
        self.splits_head = False

    def padded(self, batch: PersistentBatch, model: Any, stores: Sequence[Any]) -> torch.Tensor:
        """An eager forward of the bucket with the padding row in its first lane, from a cleared
        padding item. The backends allocate the bucket's fixed buffers here the first time,
        outside inference mode: every later step updates them in place."""
        for store in stores:
            store.clear(store.scratch())
        for backend in self.backends:
            backend.prepare(batch, padding_row(), PlanKey("decode", self.size))
        fetch(self.backends, self.inputs, self.positions)
        forward = model.forward_backbone if getattr(model, "vp_head_ready", False) else model.forward
        with torch.inference_mode():
            return forward(self.inputs, self.positions, self.last_index)

    def capture(self, batch: PersistentBatch, model: Any, stores: Sequence[Any], recorder: Any) -> None:
        """Record the bucket's forward, after an eager one as the warm-up. A graph of one lane has
        no padded lane beside the padding row, so what it replays is defined: it must be the
        logits the same kernels gave eagerly."""
        split = bool(getattr(model, "vp_head_ready", False))
        forward = model.forward_backbone if split else model.forward
        with recorder.warmup():
            expected = self.padded(batch, model, stores).clone()
            for store in stores:
                store.clear(store.scratch())
        with torch.inference_mode(), recorder.capture() as graph:
            self.output = forward(self.inputs, self.positions, self.last_index)
        self.graph = graph
        self.splits_head = split
        self._dirty_from = self.size
        with recorder.warmup():
            graph.replay()
        assert self.output is not None
        if self.size == 1:
            torch.testing.assert_close(self.output, expected, rtol=1e-3, atol=1e-3, msg=lambda m: (
                f"a decode graph replays other logits than the eager forward it recorded: {m}"))


def verify_padding_row(steps: int) -> BatchPlan:
    """A verify step of the padding row alone (doc/specs/mtp.md): ``steps`` tokens on the scratch items."""
    return BatchPlan(Column.of("i32", [-1]), Column.of("i32", [steps]), Column.of("i32", [0]),
                     Column.of("i64", []), Column.of("u8", [0]), "decode", 0, steps)


class VerifyGraphPlan:
    """A verify step for one bucket of rows (doc/specs/mtp.md): SGLang's decode graph runner in
    ``TARGET_VERIFY``, ``num_tokens_per_bs`` the step's D tokens a row; the target's logits and hidden
    states of every token. The rows are padded with the padding row, D tokens each."""

    def __init__(self, size: int, steps: int, device: torch.device, backends: Sequence[Backend]) -> None:
        self.size, self.steps, self.backends = size, steps, list(backends)
        self.inputs = torch.zeros(size * steps, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.inputs)
        self.every = torch.arange(size * steps, device=device)
        self.output: tuple[torch.Tensor, torch.Tensor] | None = None
        self.graph: Any = None
        self.extend: DraftExtendGraphPlan | None = None
        """The draft model's extend after this bucket's verify step, recorded after it."""

    def run(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor, positions: Tensor
            ) -> tuple[Tensor, Tensor]:
        n = len(view.rows)
        if view.draft_tokens != self.steps or not 0 < n <= self.size:
            raise ValueError(f"a verify graph runs {self.steps} tokens a row, at most its bucket of {self.size} rows")
        for backend in self.backends:
            backend.prepare(batch, view, PlanKey("decode", self.size))
        tokens = n * self.steps
        self.inputs[:tokens].copy_(input_ids)
        self.positions[:tokens].copy_(positions)
        if tokens < len(self.inputs):
            self.inputs[tokens:].zero_()
            self.positions[tokens:].zero_()
        self.graph.replay()
        assert self.output is not None
        logits, hidden = self.output
        return logits[:tokens], hidden[:tokens]

    def close(self) -> None:
        if self.extend is not None:
            self.extend.close()
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output, self.extend = None, None, None

    def padded(self, batch: PersistentBatch, model: Any, stores: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        for store in stores:
            store.clear(store.scratch())
        for backend in self.backends:
            backend.prepare(batch, verify_padding_row(self.steps), PlanKey("decode", self.size))
        # Not inference mode: FlashInfer's prefill wrapper allocates buffers at its first forward that
        # each later plan, outside inference mode, writes into.
        with torch.no_grad():
            return model.forward_hidden(self.inputs, self.positions, self.every)

    def capture(self, batch: PersistentBatch, model: Any, stores: Sequence[Any], recorder: Any) -> None:
        """As ``GraphPlan.capture``: a warm-up eager forward, the capture, a replay. Not held to the eager
        logits: the padding row's D tokens all store their K/V in one slot (its row of ``req_to_token`` is
        the padding slot everywhere), which of them lands there is up to the device, eagerly as in the
        replay. A real row's tokens have slots of their own (tests/gpu: a replayed verify step is the
        eager one)."""
        with recorder.warmup():
            self.padded(batch, model, stores)
            for store in stores:
                store.clear(store.scratch())
        with torch.inference_mode(), recorder.capture() as graph:
            self.output = model.forward_hidden(self.inputs, self.positions, self.every)
        self.graph = graph
        with recorder.warmup():
            graph.replay()
        assert self.output is not None


def eager_verify_padding_step(batch: PersistentBatch, model: Any, backends: Sequence[Backend], stores: Sequence[Any],
                              device: torch.device, steps: int) -> None:
    """``eager_padding_step`` for speculation: one eager verify step of the padding row."""
    for store in stores:
        store.clear(store.scratch())
    for backend in backends:
        backend.prepare(batch, verify_padding_row(steps), PlanKey("decode", 0))
    tokens = torch.zeros(steps, dtype=torch.int64, device=device)
    with torch.no_grad():  # as VerifyGraphPlan.padded: the operators made here are planned later
        model.forward_hidden(tokens, torch.arange(steps, device=device), torch.arange(steps, device=device))
    for store in stores:
        store.clear(store.scratch())


class DraftGraphPlan:
    """The draft model's decode forward for one bucket of rows (doc/specs/mtp.md): one of the D - 2
    forwards of SGLang's ``EAGLEDraftCudaGraphRunner``, replayed for each of them, the attention
    backends prepared before every replay for that forward's positions (SGLang plans each forward's
    metadata before its replay too). The rows padded with the padding row, as a decode graph's."""

    def __init__(self, size: int, hidden_size: int, dtype: torch.dtype, device: torch.device,
                 backends: Sequence[Backend]) -> None:
        self.size, self.backends = size, list(backends)
        self.inputs = torch.zeros(size, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.inputs)
        self.hidden = torch.zeros(size, hidden_size, dtype=dtype, device=device)
        self.last_index = torch.arange(size, device=device)
        self.output: tuple[torch.Tensor, torch.Tensor] | None = None
        self.graph: Any = None

    def run(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor, positions: Tensor,
            hidden: Tensor) -> tuple[Tensor, Tensor]:
        n = len(view.rows)
        if view.phase != "decode" or view.draft_tokens or not 0 < n <= self.size \
                or any(t != 1 for t in view.num_tokens.values()):
            raise ValueError(f"a draft graph runs one token per row, at most its bucket of {self.size} rows")
        for backend in self.backends:
            backend.prepare(batch, view, PlanKey("decode", self.size))
        self.inputs[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        self.hidden[:n].copy_(hidden)
        if n < self.size:
            self.inputs[n:].zero_()
            self.positions[n:].zero_()
            self.hidden[n:].zero_()
        self.graph.replay()
        assert self.output is not None
        logits, out = self.output
        return logits[:n], out[:n]

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output = None, None

    def padded(self, batch: PersistentBatch, model: Any, stores: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        for store in stores:
            store.clear(store.scratch())
        for backend in self.backends:
            backend.prepare(batch, padding_row(), PlanKey("decode", self.size))
        with torch.no_grad():  # as VerifyGraphPlan.padded
            return model.draft_forward(self.inputs, self.positions, self.hidden, self.last_index)

    def capture(self, batch: PersistentBatch, model: Any, stores: Sequence[Any], recorder: Any) -> None:
        """As ``GraphPlan.capture``, with the same check: one lane is the padding row alone, one token,
        so what it replays is defined and must be what the same kernels gave eagerly."""
        with recorder.warmup():
            expected = self.padded(batch, model, stores)[0].clone()
            for store in stores:
                store.clear(store.scratch())
        with torch.no_grad(), recorder.capture() as graph:
            self.output = model.draft_forward(self.inputs, self.positions, self.hidden, self.last_index)
        self.graph = graph
        with recorder.warmup():
            graph.replay()
        assert self.output is not None
        if self.size == 1:
            torch.testing.assert_close(self.output[0], expected, rtol=1e-3, atol=1e-3, msg=lambda m: (
                f"a draft graph replays other logits than the eager forward it recorded: {m}"))


class DraftExtendGraphPlan:
    """The draft model's extend after a verify step replayed from ``verify`` (doc/specs/mtp.md, SGLang's
    ``EAGLEDraftExtendCudaGraphRunner``): the same D positions a lane, on the target's hidden states
    that graph left in its output. It attends over the same extents as that verify step, so it runs on
    the attention metadata the verify step prepared and prepares nothing: it replays right after its
    verify graph, nothing in between preparing the attention backends (``Plans.draft``). Each lane's
    logits and hidden state at its accepted end; a padded lane's at its first token."""

    def __init__(self, verify: VerifyGraphPlan, backends: Sequence[Backend]) -> None:
        self.verify, self.backends = verify, list(backends)
        size, steps = verify.size, verify.steps
        device = verify.inputs.device
        self.inputs = torch.zeros(size * steps, dtype=torch.int64, device=device)
        self.first = torch.arange(size, device=device) * steps
        self.last_index = self.first.clone()
        self.output: tuple[torch.Tensor, torch.Tensor] | None = None
        self.graph: Any = None

    def run(self, view: BatchPlan, input_ids: Tensor, last_index: Tensor) -> tuple[Tensor, Tensor]:
        n, steps = len(view.rows), self.verify.steps
        if view.draft_tokens != steps or not 0 < n <= self.verify.size:
            raise ValueError(f"a draft extend graph runs {steps} tokens a row, at most {self.verify.size} rows")
        self.inputs[:n * steps].copy_(input_ids)
        self.inputs[n * steps:].zero_()
        self.last_index[:n].copy_(last_index)
        self.last_index[n:].copy_(self.first[n:])
        self.graph.replay()
        assert self.output is not None
        logits, hidden = self.output
        return logits[:n], hidden[:n]

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output = None, None

    def _forward(self, model: Any) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.verify.output is not None, "the verify graph is recorded first"
        return model.draft_forward(self.inputs, self.verify.positions, self.verify.output[1], self.last_index)

    def capture(self, batch: PersistentBatch, model: Any, stores: Sequence[Any], recorder: Any) -> None:
        """After ``verify``'s capture: a warm-up eager forward on the verify bucket's metadata, the capture,
        a replay. Not held to the eager logits, for the reason a verify graph is not."""
        with recorder.warmup():
            for store in stores:
                store.clear(store.scratch())
            for backend in self.backends:
                backend.prepare(batch, verify_padding_row(self.verify.steps), PlanKey("decode", self.verify.size))
            with torch.no_grad():
                self._forward(model)
            for store in stores:
                store.clear(store.scratch())
        with torch.no_grad(), recorder.capture() as graph:
            self.output = self._forward(model)
        self.graph = graph
        with recorder.warmup():
            graph.replay()


def eager_draft_padding_steps(batch: PersistentBatch, model: Any, backends: Sequence[Backend], stores: Sequence[Any],
                              device: torch.device, steps: int, hidden: torch.Tensor) -> None:
    """The draft model's forwards a speculative step runs, eagerly on the padding row: a decode
    forward and the extend of a verify step, so that the kernels they use are seen before any of them
    is recorded (``eager_padding_step``). ``hidden`` ``[1, hidden_size]`` of the draft's dtype."""
    for store in stores:
        store.clear(store.scratch())
    token = torch.zeros(1, dtype=torch.int64, device=device)
    with torch.no_grad():
        for backend in backends:
            backend.prepare(batch, padding_row(), PlanKey("decode", 0))
        model.draft_forward(token, torch.zeros_like(token), hidden, torch.arange(1, device=device))
        for backend in backends:
            backend.prepare(batch, verify_padding_row(steps), PlanKey("decode", 0))
        tokens = torch.zeros(steps, dtype=torch.int64, device=device)
        model.draft_forward(tokens, torch.arange(steps, device=device), hidden.expand(steps, -1).contiguous(),
                            torch.zeros(1, dtype=torch.int64, device=device))
    for store in stores:
        store.clear(store.scratch())

