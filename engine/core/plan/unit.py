"""``core.plan``: the worker's execution plans (``plan@1``, stage2-contract.md §9), as SGLang's
``ModelRunner`` keeps its runners (srt/model_executor/runner/, e838360): an ``EagerRunner`` that
runs any step, and graph runners that replay the buckets they captured.

Bucket 0 prepares metadata and runs eagerly. The decode graph buckets are recorded when the unit
loads, at start or while the worker is paused, largest first into one memory pool; ``buckets``
lists them while what they were recorded over still holds. Nothing rebuilds them in place: a
change to what they were built from reloads this unit (``ctx.derived`` without a rebuild action),
after the units it was built from, and a rollback reloads the old one the same way.

``plan.cuda_graph``: ``off`` records nothing; ``auto`` records where it can and says why where it
cannot (no graph platform, a kernel that is not capturable, graphs that do not fit their memory);
``required`` fails the start in those cases, and has the executor refuse a decode step no graph
holds, before it changes any state.

At start the unit has every backend fix its kernel solutions per phase (``Backend.resolve``; the
selections are the plans' by-value dependencies) and logs one record of what runs on what, with
which state formats (§5.5.3, §8.1). A backend named in the config that cannot run here fails the
start.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from core.contracts.kernels import CAPTURE
from core.contracts.plan import GRAPH, Backend, PersistentBatch, Phase, Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from core.contracts.tensor import Tensor
from platforms.base import StaleGraph
from core.plan.graph import (Dependencies, DraftExtendGraphPlan, DraftGraphPlan, GraphPlan, VerifyGraphPlan,
                             buckets, eager_draft_padding_steps, eager_padding_step, eager_verify_padding_step, fetch)
from unit.runtime.proxy import provision_of


def phase_of(plan: BatchPlan) -> Phase:
    """A mixed step runs as an extend, as SGLang's ``EagerRunner`` runs MIXED as EXTEND."""
    return "decode" if plan.phase == "decode" else "prefill"


class EagerPlan:
    """``Plan`` for bucket 0: SGLang's ``EagerRunner``."""

    def __init__(self, model: Any, backends: Sequence[Backend]) -> None:
        self.model, self.backends = model, list(backends)

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
            last_index: Tensor) -> Tensor:
        key = PlanKey(phase_of(plan), 0)
        for backend in self.backends:
            backend.prepare(batch, plan, key)
        fetch(self.backends, input_ids, positions)
        return self.model.forward(input_ids, positions, last_index)

    def run_backbone(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
                     last_index: Tensor) -> Tensor:
        key = PlanKey(phase_of(plan), 0)
        for backend in self.backends:
            backend.prepare(batch, plan, key)
        fetch(self.backends, input_ids, positions)
        return self.model.forward_backbone(input_ids, positions, last_index)


def resolution(platform: Any, states: Sequence[StateDecl],
               backends: Mapping[str, Mapping[str, Mapping[str, str]]]) -> dict[str, Any]:
    """The start-up record: the platform, each state kind's dtypes (and page size), and per phase
    each backend's choice, with the one of a decode step replayed from a graph; verify does not run
    in stage 2 yet."""
    kinds: dict[str, dict[str, Any]] = {}
    for d in states:
        kinds.setdefault(d.kind, {**{name: dtype for name, (_, dtype) in d.tensors.items()},
                                  **({"page_size": d.tokens_per_item} if d.tokens_per_item else {})})
    devices = platform.devices()
    return {"platform": platform.info().name, "device": list(devices[0].capabilities) if devices else [],
            "states": kinds, **{phase: {name: r[phase] for name, r in backends.items()}
                                for phase in ("prefill", "decode", GRAPH)}, "verify": "disabled"}


class Plans:
    """``plan@1``."""

    def __init__(self, model: Any, backends: Sequence[Backend], required: bool = False, *,
                 attention: Sequence[Backend] = (), gdn: Sequence[Backend] = ()) -> None:
        self.eager = EagerPlan(model, backends)
        self.required = required
        self.attention, self.gdn = list(attention), list(gdn)
        """The backends a draft forward prepares (the draft model has attention layers only), and those
        that commit a verify step's GDN states."""
        self.resolved: Mapping[str, Any] = {}
        """The start-up record (``resolution``, and what became of the graphs)."""
        self.idle_graph: Any = None
        self.graphs: dict[int, GraphPlan] = {}
        self.verify_graphs: dict[int, VerifyGraphPlan] = {}
        """With speculation, the verify steps' graphs instead of the decode steps' (doc/specs/mtp.md), each
        with the draft model's extend after it."""
        self.draft_graphs: dict[int, DraftGraphPlan] = {}
        """With speculation of more than two tokens, the draft model's decode forward."""
        self.replayed: VerifyGraphPlan | None = None
        """The verify graph the step in progress replayed: its draft extend replays on its metadata."""
        self.deps: Dependencies | None = None
        """What every bucket was recorded over."""

    def replay_idle(self) -> bool:
        if self.idle_graph is None or self.deps is None or not self.deps.valid():
            return False
        self.eager.model.prepare_idle()
        self.idle_graph.replay()
        self.eager.model.stepped()
        if getattr(self.eager.model, "vp_head_ready", False):
            self.eager.model.vp_head_step(None)
        return True

    def buckets(self, phase: Phase) -> list[int]:
        """A step's one dependency check: the executor asks here before it changes any state."""
        if phase != "decode" or self.deps is None or not self.deps.valid():
            return []
        return list(self.graphs)

    def get(self, key: PlanKey) -> Plan:
        if key.bucket == 0:
            return self.eager
        if key.phase == "decode" and key.bucket in self.graphs:
            return self.graphs[key.bucket]
        raise LookupError(f"no {key.phase} graph of bucket {key.bucket} is ready")

    # ------------------------------------------------------------ speculation (doc/specs/mtp.md)

    def prefill_hidden(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
                       last_index: Tensor) -> tuple[Tensor, Tensor]:
        for backend in self.eager.backends:
            backend.prepare(batch, plan, PlanKey("prefill", 0))
        return self.eager.model.forward_hidden(input_ids, positions, last_index)

    def verify(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor,
               positions: Tensor) -> tuple[Tensor, Tensor]:
        self.replayed = None
        fits = [b for b in self.verify_graphs if b >= len(view.rows)]
        if fits and self.deps is not None and self.deps.valid():
            try:
                graph = self.verify_graphs[min(fits)]
                out = graph.run(batch, view, input_ids, positions)
                self.replayed = graph
                return out
            except StaleGraph:
                if self.required:
                    raise
        elif self.required and self.verify_graphs:
            raise StaleGraph(f"no ready verify graph holds {len(view.rows)} rows, and graphs are required")
        for backend in self.eager.backends:
            backend.prepare(batch, view, PlanKey("decode", 0))
        every = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.eager.model.forward_hidden(input_ids, positions, every)

    def draft(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor, positions: Tensor, hidden: Tensor,
              last_index: Tensor) -> tuple[Tensor, Tensor]:
        """The draft model's forward: replayed for a speculative step's decode forwards and its extend
        (a decode view with ``draft_tokens``, after ``verify``), eager for a prefill's extend."""
        if view.phase == "decode" and view.draft_tokens:
            graph, self.replayed = self.replayed, None
            if graph is not None and graph.extend is not None:
                return graph.extend.run(view, input_ids, last_index)
            if self.required and self.verify_graphs:
                raise StaleGraph("a draft extend after a verify step that was not replayed, and graphs are required")
        elif view.phase == "decode":
            fits = [b for b in self.draft_graphs if b >= len(view.rows)]
            if fits and self.deps is not None and self.deps.valid():
                return self.draft_graphs[min(fits)].run(batch, view, input_ids, positions, hidden)
            if self.required and self.draft_graphs:
                raise StaleGraph(f"no ready draft graph holds {len(view.rows)} rows, and graphs are required")
        for backend in self.attention:
            backend.prepare(batch, view, PlanKey(phase_of(view), 0))
        return self.eager.model.draft_forward(input_ids, positions, hidden, last_index)

    def commit(self, steps: Tensor, track_steps: Tensor) -> None:
        for backend in self.gdn:
            cast(Any, backend).commit(steps, track_steps)

    def close(self) -> None:
        """After the step stream and the consumers of the last logits are done. The graphs' pool
        goes back to the device: PyTorch would keep its blocks cached where the next graphs,
        recorded into a pool of their own, cannot use them."""
        if not self.graphs and not self.verify_graphs and self.idle_graph is None:
            return
        torch.cuda.synchronize()
        if self.idle_graph is not None:
            self.idle_graph.close()
            self.idle_graph = None
        for graph in [*self.graphs.values(), *self.verify_graphs.values(), *self.draft_graphs.values()]:
            graph.close()
        self.graphs, self.verify_graphs, self.draft_graphs, self.deps, self.replayed = {}, {}, {}, None, None
        for backend in self.eager.backends:
            backend.release_graphs()
        torch.cuda.empty_cache()


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        mode, budget = config["plan.cuda_graph"], config["plan.graph_memory_bytes"]
        model = ctx.use("model@1")
        names = list(dict.fromkeys(model.assembly.layers))
        backends: list[Backend] = [model.backends[name] for name in names]
        self.model, self.backends = model, backends
        self.catalog = ctx.use("kernels.catalog@1")
        self.batch, self.weights = ctx.use("worker.batch@1"), ctx.use("state.weights@1")
        if steps := int(getattr(model, "draft_tokens", 0)):
            # Record graph dependencies after all batch buffers exist; the executor's
            # initialization must not invalidate the verify graphs just captured here.
            self.batch.prepare_spec(steps, model.hidden_size, model.dtype)
        self.stores = {kind: store for kind, store in model.stores.items()
                       if any(d.kind == kind for d in model.assembly.states)}
        self.plans = Plans(model, backends, required=mode == "required",
                           attention=[b for n, b in zip(names, backends, strict=True) if n == "layers.attn@1"],
                           gdn=[b for n, b in zip(names, backends, strict=True) if n == "layers.gdn@1"])
        ctx.effect(lambda: self.plans.close, label="decode graphs")
        resolved = resolution(ctx.platform, model.assembly.states,
                              {name: b.resolve(model.assembly.states) for name, b in zip(names, backends, strict=True)})
        uncapturable = sorted({definition for chosen in resolved[GRAPH].values()
                               for definition, solution in chosen.items() if not solution})
        captured = [solution.removesuffix(" (fallback)") for chosen in resolved[GRAPH].values()
                    for definition, solution in chosen.items() if definition != "backend" and solution]
        sizes = buckets(mode, config["plan.graph_buckets"], config["batch.max_rows"], budget,
                        fixed_extent=any(CAPTURE in self.catalog.describe(s).requires for s in captured))
        reason = ("disabled" if mode == "off" else
                  "asleep (holder.sleep)" if config["holder.sleep"] else
                  "platform has no CUDA graph support" if "cuda_graph" not in resolved["device"] else
                  f"uncapturable decode kernels: {uncapturable}" if uncapturable else "")
        build: dict[str, Any] = {}
        if not reason:
            try:
                reason, build = self.record(sizes, budget, ctx.platform.graph())
            except (MemoryError, torch.OutOfMemoryError) as e:
                if self.plans.required:
                    raise
                reason = "the graphs do not fit: " + ". ".join(str(e).split(". ")[:2])
        if reason and self.plans.required and not config["holder.sleep"]:  # asleep, nothing runs
            raise ValueError(f"plan.cuda_graph=required: {reason}")
        # No rebuild action: core.deps reloads this unit when one of these changes, or when a unit
        # that registers kernels is added. Also without graphs: a kernel unit may bring them.
        ctx.derived("decode_graphs", rebuild="",
                    units=[*self.catalog.sources(), str(provision_of(self.catalog).owner.unit)])
        resolved["cuda_graph"] = {"mode": mode, "buckets": list(self.plans.graphs or self.plans.verify_graphs),
                                  **({"verify": True} if self.plans.verify_graphs else {}),
                                  **({"draft": True} if self.plans.draft_graphs else {}), "reason": reason, **build}
        self.plans.resolved = resolved
        print(f"core.plan: {json.dumps(resolved, sort_keys=True)}", file=sys.stderr, flush=True)
        ctx.provide("plan@1", self.plans)

    def record(self, sizes: Sequence[int], budget: int, recorder: Any) -> tuple[str, dict[str, Any]]:
        """Record a graph per bucket, or say why none is recorded. All or none: a failure frees
        what it recorded, and so does a set that keeps more device memory than ``budget``."""
        states = self.model.assembly.states
        self.batch.prepare(list(self.stores))
        device = self.batch.binding.buffers()["last_token"].device
        stores = list(self.stores.values())
        # An eager step of the padding row first. It shows what the forward runs besides the
        # backends' kernels (norms, linears, ...): eager's choices, which a captured step shares, so
        # they have to be capturable too. And the kernel libraries allocate in it what all their
        # steps share, eager ones included: that is not the graphs', stays when they are closed,
        # and is there before their memory is counted, on a cold start as after a reload.
        steps = int(getattr(self.model, "draft_tokens", 0))  # speculation: verify steps are replayed instead
        attention = self.plans.attention
        hidden = self.batch.binding.buffers()["spec.hidden"][:1] if steps else None
        with recorder.warmup(), self.catalog.recording() as used:
            if steps:
                assert hidden is not None
                eager_verify_padding_step(self.batch, self.model, self.backends, stores, device, steps)
                eager_draft_padding_steps(self.batch, self.model, attention, stores, device, steps, hidden.clone())
            else:
                eager_padding_step(self.batch, self.model, self.backends, stores, device)
        uncapturable = sorted(s for s in used if not self.catalog.describe(s).capturable)
        if uncapturable:
            return f"uncapturable decode kernels: {uncapturable}", {}
        if not steps:
            prepare_vp = getattr(self.model, "prepare_vp_head", None)
            if prepare_vp is not None:
                prepare_vp()
        graphs: dict[int, Any] = {size: VerifyGraphPlan(size, steps, device, self.backends) if steps
                                  else GraphPlan(size, device, self.backends) for size in sorted(sizes, reverse=True)}
        drafts: dict[int, DraftGraphPlan] = ({size: DraftGraphPlan(size, self.model.hidden_size, hidden.dtype, device, attention)
                                              for size in graphs} if steps > 2 and hidden is not None else {})
        seconds: dict[str, float] = {}
        before = 0
        try:
            if 1 not in graphs:
                # Only a graph of one lane can be held to the eager logits (GraphPlan.capture):
                # with no such bucket, record one for the check alone, and have the backends
                # drop what they keep for it (they hold no other bucket yet).
                probe: Any = VerifyGraphPlan(1, steps, device, self.backends) if steps else GraphPlan(1, device, self.backends)
                try:
                    probe.capture(self.batch, self.model, stores, recorder)
                finally:
                    probe.close()
                    for backend in self.backends:
                        backend.release_graphs()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()  # count what the graphs keep, not what a cached block happened to hold
            before = torch.cuda.memory_reserved(device)
            # The largest first, into one pool: the smaller ones fit in what it took.
            for size, graph in graphs.items():
                started = time.perf_counter()
                graph.capture(self.batch, self.model, stores, recorder)
                if steps:  # its draft extend runs on its metadata and its output: right after it
                    graph.extend = DraftExtendGraphPlan(graph, attention)
                    graph.extend.capture(self.batch, self.model, stores, recorder)
                if size in drafts:
                    drafts[size].capture(self.batch, self.model, stores, recorder)
                seconds[str(size)] = round(time.perf_counter() - started, 3)
            capture_idle = getattr(self.model, "capture_idle", None)
            if not steps and capture_idle is not None:
                self.plans.idle_graph = capture_idle(recorder)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            kept = torch.cuda.memory_reserved(device) - before
            if kept > budget:
                raise MemoryError(f"the decode graphs keep {kept} bytes of device memory, more than "
                                  f"plan.graph_memory_bytes = {budget}")
        except BaseException:
            torch.cuda.synchronize(device)
            if self.plans.idle_graph is not None:
                self.plans.idle_graph.close()
                self.plans.idle_graph = None
            for graph in [*graphs.values(), *drafts.values()]:
                graph.close()
            for backend in self.backends:
                backend.release_graphs()
            torch.cuda.empty_cache()
            raise
        # One generation per provider: a store counts its own over all its layers.
        layer = {kind: next(d.layer for d in states if d.kind == kind) for kind in self.stores}
        self.plans.deps = Dependencies(
            [self.model, self.batch, self.weights, self.catalog, *self.backends, *stores],
            [lambda: self.batch.binding, lambda: self.weights.bind(""),
             *(lambda b=b: b.binding for b in self.backends),
             *(lambda kind=kind: self.stores[kind].bind(layer[kind]) for kind in self.stores)], self.catalog)
        if steps:
            self.plans.verify_graphs = dict(sorted(graphs.items()))
            self.plans.draft_graphs = dict(sorted(drafts.items()))
        else:
            self.plans.graphs = dict(sorted(graphs.items()))
        return "", {"seconds": seconds, "kept_bytes": kept, "idle_experts": self.plans.idle_graph is not None}


if TYPE_CHECKING:
    _: type[PlanService] = Plans
    __: type[Plan] = EagerPlan
