# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""``layers.ds1017``: DeepSeek-V4.1's layers (``layers.ds1017@1``, doc/specs/v1-ds1017.md), after the
reference implementation shipped with the checkpoint (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``): ``Attention`` with its
``Compressor`` and ``Indexer``, ``Engram`` with ``NgramHashState``, ``MoE``, and the Hyper-Connections
of ``Block``. The kernels are the catalog's (``ds.*``), fixed per phase at start.

What moved. That implementation keeps one cache row per batch row, all rows at the same position,
prefilled in one piece from position 0. Here the rows are the persistent batch's, each at its own
position, a step runs any number of tokens of each, and the caches are the stores', in DeepInfer's
byte formats (``kernels.ds1017_formats``), which its kernels read:

- The window KV is a ring per request and layer (``state.swa.slots@1``, ``ring``, one page of
  ``window`` FP8 records): position ``p`` at entry ``p % window``.
- Compressed latents and the indexer's keys are paged (``state.kv.paged@1``, ``kv`` and ``ik`` of the
  source layer), ``page_size / ratio`` a page: the latent of tokens ``[j * ratio, (j + 1) * ratio)``
  at row ``slot(j * ratio) / ratio``, ``slot`` the token slot.
- A group still filling (``cstate``: score and KV of each member) and the tokens an n-gram reaches
  back to (``tail``) are in the request's slot.

The order of a token's keys is that implementation's (the window oldest first, then the chosen
latents by position), which its attention kernel's result depends on in the last bit.

Two ways to lay a step out (``prepare``), as DeepInfer's decode and prefill:

- A decode step (one token per row) derives everything on the device from the rows' table indices
  and positions, with fixed shapes: a graph records it. It attends in the caches
  (``ds.cache_attn@1``) and the indexer reads the keys through the rows' pages
  (``ds.index_topk_paged@1``). A graph bucket scores the indexer over ``batch.max_context_len``
  positions, an eager step over the longest row's.
- Any other step is laid out on the host (``Layout``): rows of any length from any position, a
  compression group completed by tokens of an earlier step. Its layers load what they attend to out
  of the caches into one source (the window's earlier keys, the latents of its rows: load segments)
  and the indexer scores ranges of the loaded keys (``ds.index_topk@1``).
"""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.kernels import KernelFn
from core.contracts.plan import GRAPH, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from kernels import ds1017_formats as formats
from kernels import ds1017_yoco as yoco
from layers.solutions import Kernels
from models.ds1017_config import SLOT_KIND, DSConfig, compressed_token_map, engram_layout

PREFILL_ONLY = frozenset({"ds.transform@1", "ds.load_values@1", "ds.load_keys@1", "ds.index_topk@1", "ds.compress@1",
                          "ds.sparse_attn_out@1"})
"""What only a step laid out on the host runs."""
DECODE_ONLY = frozenset({"ds.cache_attn_out@1", "ds.index_topk_paged@1", "ds.compress_decode@1"})
"""What only a decode step runs: a graph records these, not the others."""

DEFINITIONS = ("ds.linear@1", "ds.rms_norm@1", "ds.transform@1", "ds.store_cache@1", "ds.load_values@1",
               "ds.load_keys@1", "ds.quant_index_q@1", "ds.index_topk@1", "ds.index_topk_paged@1", "ds.compress@1",
               "ds.compress_decode@1", "ds.hc_step@1", "ds.hc_pre@1", "ds.hc_post@1", "ds.moe@1",
               "ds.engram_hash@1", "ds.attn_qkv@1", "ds.cache_attn_out@1", "ds.sparse_attn_out@1", "ds.engram@1")

MEGA = frozenset({"deepinfer.ds.moe@1"})
"""Solutions that read the expert stacks in the fused MoE kernel's row order (``formats.expert_layout``)."""

Weights = Mapping[str, torch.Tensor]


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there (pinned, non-blocking)."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class CompressionGroups:
    """State updates for a prefill writing groups of more than one token."""

    seq_token: torch.Tensor
    """``[rows]``: a token of each row (its first). The rest is the compressor's metadata, as DeepInfer's."""
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    cu_c: torch.Tensor
    c_seq: torch.Tensor
    """``[entries]``: the row of each compressed entry; a row reserves at least one."""
    first: torch.Tensor
    """``[entries]``: the position of an entry's first token (0 for a reserved one)."""
    real: torch.Tensor
    """``[entries]``: the entry is a group this step completes."""
    entry_token: torch.Tensor
    """``[entries]``: a token of the entry's row."""


@dataclass(frozen=True)
class Latents:
    """One compression ratio in a step laid out on the host: where its rows' latents are, and what the
    compressor (above one) needs."""

    segments: torch.Tensor
    """``[n, 4]`` int32 load segments of the rows' latents, in step order (``kernels.ds1017_compose``)."""
    total: int
    width: int
    """The most latents a row has after the step."""
    ks: torch.Tensor
    """``[tokens]``: where its row's latents begin among the loaded ones; ``ke`` how many it reaches past that."""
    ke: torch.Tensor
    groups: CompressionGroups | None = None


@dataclass(frozen=True)
class PrefixSnapshot:
    """Token indices preceding each mid-chunk checkpoint and their private destinations in ``swa.slots``."""

    slots: torch.Tensor
    ends: torch.Tensor
    ring_tok: torch.Tensor
    ring_dst32: torch.Tensor
    cstate: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    @classmethod
    def prepare(cls, batch: PersistentBatch, plan: BatchPlan, cfg: DSConfig, device: torch.device,
                slots: Any, encoder_ratios: Sequence[int]) -> PrefixSnapshot | None:
        tracked, destinations, cutoffs = batch.track(SLOT_KIND)
        if not tracked:
            return None
        chunks: dict[int, tuple[int, int, int]] = {}
        offset = 0
        for row, start, count in zip(plan.rows.values(), plan.positions.values(), plan.num_tokens.values(), strict=True):
            chunks[int(row)] = offset, int(start), int(count)
            offset += int(count)
        v_rows: list[int] = []
        v_dsts: list[int] = []
        ends: list[int] = []
        spans: list[int] = []
        v_cuts: list[int] = []
        for row, dst, cutoff in zip(tracked, destinations, cutoffs, strict=True):
            chunk = chunks.get(int(row))
            if chunk is None:
                continue
            off, start, count = chunk
            if start < int(cutoff) <= start + count:
                v_rows.append(int(row))
                v_dsts.append(int(dst))
                ends.append(off + int(cutoff) - start)
                spans.append(int(cutoff) - start)
                v_cuts.append(int(cutoff))
        if not v_rows:
            return None
        live = batch.ids(SLOT_KIND)
        slots.copy([live[r][-1] for r in v_rows], v_dsts)
        win = cfg.sliding_window
        dst1 = [d + 1 for d in v_dsts]
        meta = on_device(torch.tensor([ends, v_cuts, dst1, spans], dtype=torch.int64), device)
        ends_d, cuts_d, slots_d, spans_d = meta[0], meta[1], meta[2], meta[3]
        back_w = torch.arange(-win, 0, device=device)
        ring_tok = (ends_d[:, None] + back_w).clamp_min(0).flatten()
        ring_valid = back_w[None, :] >= -spans_d[:, None]
        ring_pos = cuts_d[:, None] + back_w
        ring_dst32 = torch.where(ring_valid, slots_d[:, None] * win + ring_pos % win, -1).flatten().to(torch.int32)
        cstate: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for ratio in encoder_ratios:
            if ratio <= 1:
                continue
            if min(spans) >= ratio:
                back_r = torch.arange(-ratio, 0, device=device)
                cstate[ratio] = (slots_d[:, None], (cuts_d[:, None] + back_r) % ratio, ends_d[:, None] + back_r)
            else:
                c_slots: list[int] = []
                c_cols: list[int] = []
                c_toks: list[int] = []
                for d1, end_i, cut_i, span_i in zip(dst1, ends, v_cuts, spans, strict=True):
                    m = min(span_i, ratio)
                    for k in range(-m, 0):
                        c_slots.append(d1)
                        c_cols.append((cut_i + k) % ratio)
                        c_toks.append(end_i + k)
                tri = on_device(torch.tensor([c_slots, c_cols, c_toks], dtype=torch.int64), device)
                cstate[ratio] = (tri[0], tri[1], tri[2])
        return cls(slots_d, ends_d, ring_tok, ring_dst32, cstate)


@dataclass(frozen=True)
class AttentionLayout:
    """Read and write indices shared by the attention layers of one query view."""

    rows: torch.Tensor
    """``[tokens]``: each token's row in the batch's tables."""
    ring_segments: torch.Tensor
    """``[rows with earlier tokens, 4]``: each such row's ring, loaded whole (the window's earlier keys)."""
    ring_total: int
    window_index: torch.Tensor
    """``[tokens, window]``: each token's window keys, oldest first, as indices into the layer's source
    (the loaded rings, then this step's tokens); ``-1`` for none."""
    keep: torch.Tensor
    """Token indices of the tokens still in their row's window after the step: into the ring."""
    latents: Mapping[int, Latents]
    window_cnt32: torch.Tensor | None = None


@dataclass(frozen=True)
class Layout(AttentionLayout):
    """Full layout, including the producer's Engram history."""

    pos: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    start: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    ngram: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    ngram_from: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    last: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    tail: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    tail_from: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))


class Layers:
    """``layers.ds1017@1``."""

    def __init__(self, catalog: Any, config: Any, cfg: DSConfig, kv: Any, slots: Any, token_map: Sequence[int] | None,
                 max_context_len: int, group: Any, placement: Any) -> None:
        self.cfg, self.kv, self.slots = cfg, kv, slots
        self.yoco_mode = str(config["dsv4.yoco_mode"])
        self.yoco_keep = int(config["dsv4.yoco_keep_tokens"])
        self.yoco_paged_consumer = bool(config["dsv4.yoco_paged_consumer"]) if "dsv4.yoco_paged_consumer" in config else True
        self.yoco_boundary = cfg.n_prefill_layer
        if self.yoco_mode not in ("off", "compact", "mask"):
            raise ValueError("dsv4.yoco_mode must be off, compact or mask")
        if self.yoco_mode != "off":
            if self.yoco_keep < cfg.sliding_window + 1:
                raise ValueError("dsv4.yoco_keep_tokens must cover sliding_window + 1")
            if cfg.n_prefill_layer >= cfg.num_hidden_layers or cfg.kv_source(cfg.n_prefill_layer) != cfg.n_prefill_layer:
                raise ValueError("bounded replay needs a ratio-one decoder KV source")
            if any(cfg.ratio(i) != 1 or cfg.kv_source(i) != cfg.n_prefill_layer
                   for i in range(cfg.n_prefill_layer, cfg.num_hidden_layers)):
                raise ValueError("bounded replay needs decoder layers sharing the first decoder's ratio-one KV")
            if any(i >= cfg.n_prefill_layer for i in cfg.engram_layer_ids):
                raise ValueError("bounded replay needs Engram layers before the decoder boundary")
            if config["alloc.radix_cache"] and config["alloc.prefix_replay_tokens"] < self.yoco_keep:
                raise ValueError("bounded replay with the prefix cache requires alloc.prefix_replay_tokens >= dsv4.yoco_keep_tokens")
        self._replay_rows: tuple[list[int], list[int], list[int], list[bool], list[int]] | None = None
        self._replay: yoco.Replay | None = None
        self._full_kv_stored = False
        self._snapshot: PrefixSnapshot | None = None
        self._suffix_shape: tuple[int, int, int, torch.device] | None = None
        self._suffix_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._last_swa_only = max((i for i in range(cfg.num_hidden_layers) if not cfg.ratio(i)), default=-1)
        self._last_engram = max(cfg.engram_layer_ids, default=-1)
        self._all_ratios = tuple(sorted({r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r}))
        self._encoder_ratios = tuple(sorted({cfg.ratio(i) for i in range(min(cfg.n_prefill_layer, cfg.num_hidden_layers))
                                             if cfg.ratio(i)}))
        kv_order = [s for s in cfg.kv_source_layer_ids if s < cfg.num_hidden_layers]
        self._prev_kv_src = {kv_order[k]: kv_order[k - 1] for k in range(1, len(kv_order))}
        idx_order = [s for s in cfg.index_source_layer_ids if s < cfg.num_hidden_layers]
        self._prev_idx_src = {idx_order[k]: idx_order[k - 1] for k in range(1, len(idx_order))}

        self.group = group
        """``comm.group@1``: the ranks that hold this model's experts and Engram tables between them."""
        self.placement = placement
        """``expert.placement@1``: the slot of each expert, when they are placed by the tokens they take."""
        self.chosen: Mapping[str, Mapping[str, str]] = {}
        """``resolve``'s answer: phase -> definition -> solution."""
        self.tables: Callable[[], tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = lambda: ((), ())
        """This rank's rows of each Engram layer's table, weights and scales, as loaded now (``set_tables``)."""
        self.bucket = 0
        """The graph bucket of the step being prepared; 0: eager."""
        self.fetched: tuple[torch.Tensor, torch.Tensor] | None = None
        """Among ranks that share the Engram tables: the rows this step's tokens hash to, bytes and
        scales ``[tokens, layers, cols, ...]``, brought in before the forward (``fetch``)."""
        self.fetched_fixed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        """Bucket -> the buffers a graph of that bucket reads them from."""
        if cfg.n_routed_experts % group.size:
            raise ValueError(f"comm.size = {group.size} does not divide the {cfg.n_routed_experts} routed experts")
        self.kernels = Kernels(catalog, config, "dsv4")
        self.table = "prefill"
        """Which of the fixed solutions the forward runs: a phase, or ``GRAPH`` for a bucket."""
        self.max_context_len = max_context_len
        self.engram_first = min(cfg.engram_layer_ids) if cfg.engram_layer_ids else -1
        if cfg.engram_layer_ids and token_map is None:
            raise ValueError("Engram needs the compressed token map: an engram_token_map.json or a tokenizer.json in "
                             "dsv4.token_map_path, model.path or weights.path")
        self.token_map_host = list(token_map or ())
        self.constants: dict[str, torch.Tensor] = {}
        self.fixed: dict[int, torch.Tensor] = {}
        """Bucket -> its rows' table indices, the one buffer a graph of that bucket reads."""
        self.valid_fixed: dict[int, torch.Tensor] = {}
        """Bucket -> bool mask marking real rows vs bucket padding rows."""
        self.fixed_pin: dict[int, torch.Tensor] = {}
        self.valid_count: dict[int, int] = {}
        self._pin_turn = 0
        self._merged_weights: dict[tuple[Any, ...], tuple[tuple[int, ...], torch.Tensor]] = {}
        self._decode_meta_bufs: dict[tuple[int, int, bool], dict[str, torch.Tensor]] = {}
        self._pre0_buf: tuple[torch.device, torch.Tensor] | None = None
        self.attn_staged: dict[str, bool] = {}
        self.engram_staged: dict[str, bool] = {}
        self.mega_decode_attn: dict[str, bool] = {}
        self.mega_prefill_attn: dict[str, bool] = {}
        self.deepinfer_index: dict[str, bool] = {}
        self._resolved_version = -1
        self._warmed_prefill_version = -1
        self.generation = 0
        # The step in progress.
        self.batch_tables: Mapping[str, torch.Tensor] = {}
        self.layout: AttentionLayout | None = None
        self.decode_rows: torch.Tensor | None = None
        self.decode_valid: torch.Tensor | None = None
        self.extent: dict[int, int] = {}
        """Ratio -> how many latents a decode step's indexer scores (the longest row's, or the bucket's)."""
        self.step: dict[Any, Any] = {}
        """What ``begin`` derives for the forward, and what a source layer hands to the layers after it."""

    # ------------------------------------------------------------------ Backend

    @property
    def binding(self) -> Layers:
        return self

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {**{f"rows.{size}": t for size, t in self.fixed.items()},
                **{f"valid.{size}": t for size, t in self.valid_fixed.items()},
                **{f"engram.{size}.{i}": t for size, pair in self.fetched_fixed.items() for i, t in enumerate(pair)}}

    def release_graphs(self) -> None:
        self.generation += bool(self.fixed or self.valid_fixed)
        self.fixed.clear()
        self.valid_fixed.clear()
        self.fixed_pin.clear()
        self.valid_count.clear()
        self.fetched_fixed.clear()
        self._decode_meta_bufs.clear()

    def hc_pre0(self, tokens: int, device: torch.device) -> torch.Tensor:
        cur = self._pre0_buf
        if cur is not None and cur[0] == device and cur[1].shape[0] >= tokens:
            return cur[1][:tokens]
        cap = max(tokens, 36864)
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            p = torch.zeros(tokens, self.cfg.hc_mult, dtype=torch.float32, device=device)
            p[:, 0] = 1.0
            return p
        with torch.inference_mode(False):
            buf = torch.zeros(cap, self.cfg.hc_mult, dtype=torch.float32, device=device)
            buf[:, 0] = 1.0
        self._pre0_buf = (device, buf)
        return buf[:tokens]

    def _merged_linear(self, tag: str, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        r1 = w1._base if w1._base is not None else w1
        r2 = w2._base if w2._base is not None else w2
        key = (tag, r1.data_ptr(), r2.data_ptr(), w1.storage_offset(), w2.storage_offset(), w1.shape, w2.shape)
        vers = (r1._version, r2._version)
        hit = self._merged_weights.get(key)
        if hit is not None and hit[0] == vers:
            return hit[1]
        with torch.inference_mode(False):
            merged = torch.cat([w1.detach().clone(), w2.detach().clone()], dim=0).contiguous()
        self._merged_weights[key] = (vers, merged)
        weakref.finalize(r1, self._merged_weights.pop, key, None)
        return merged

    def set_tables(self, tables: Callable[[], tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]]) -> None:
        """Where this rank's rows of the Engram layers' tables are, in the layers' order: asked at
        every step, the model's weights may have been loaded again."""
        self.tables = tables

    def _sync_flags(self) -> None:
        if self.kernels.calls and self.kernels.catalog.version != self.kernels.version:
            self.chosen = self.kernels.resolve(self.kernels.calls, self.kernels.shared, self.kernels.outside)
        if self.kernels.version != self._resolved_version:
            self.attn_staged = {
                t: (m.get("ds.hc_step@1", "").split(" ")[0] == "deepinfer.ds.hc_step@1"
                    and m.get("ds.attn_qkv@1", "").split(" ")[0] == "deepinfer.ds.attn_qkv@1")
                for t, m in self.chosen.items()
            }
            self.engram_staged = {
                t: (self.group.size > 1 and bool(self.cfg.engram_layer_ids)
                    and m.get("ds.engram_fetch@1", "").split(" ")[0] == "deepinfer.ds.engram_fetch@1"
                    and m.get("ds.engram@1", "").split(" ")[0] == "deepinfer.ds.engram@1")
                for t, m in self.chosen.items()
            }
            self.mega_decode_attn = {
                t: m.get("ds.cache_attn_out@1", "").split(" ")[0] == "deepinfer.ds.cache_attn_out@1"
                for t, m in self.chosen.items()
            }
            self.mega_prefill_attn = {
                t: m.get("ds.sparse_attn_out@1", "").split(" ")[0] == "deepinfer.ds.sparse_attn_out@1"
                for t, m in self.chosen.items()
            }
            self.deepinfer_index = {
                t: (m.get("ds.index_topk@1", "").split(" ")[0] == "deepinfer.ds.index_topk@1"
                    or m.get("ds.index_topk_paged@1", "").split(" ")[0] == "deepinfer.ds.index_topk_paged@1")
                for t, m in self.chosen.items()
            }
            self._resolved_version = self.kernels.version

    def _warm_prefill(self) -> None:
        """Warm single-GPU prefill operators on tiny scratch tensors at worker startup so cold TileLang
        and DeepGEMM caches compile in parallel across all ranks instead of stalling one rank mid-flight."""
        if self._warmed_prefill_version == self.kernels.version:
            return
        if getattr(getattr(self.slots, "holder", None), "asleep", False):
            return
        prefill = self.chosen.get("prefill", {})
        if not any(v.startswith("deepinfer.") for k, v in prefill.items() if k != "backend"):
            return
        slot_tensors = getattr(self.slots, "tensors", None)
        if not slot_tensors:
            return
        sample = next(iter(next(iter(slot_tensors.values())).values()), None)
        if sample is None or sample.device.type != "cuda" or sample.is_meta:
            return
        device = sample.device
        cfg, page, win = self.cfg, self.kv.page_size, self.cfg.sliding_window
        n, hd, rd, eps = 8, cfg.head_dim, cfg.qk_rope_head_dim, cfg.rms_norm_eps
        with torch.inference_mode(False):
            pos = torch.arange(n, dtype=torch.int64, device=device)
            self.constant("arange.win", device)
            self.constant("arange.win32", device)
            self.hc_pre0(64, device)
            if cfg.engram_layer_ids:
                self.constant("arange.ng", device)
                self.constant("engram.primes32", device)
                self.constant("engram.offsets32", device)
            try:
                from kernels import ds1017_triton as fast_triton

                xh = torch.zeros(n, cfg.hidden_size, dtype=torch.bfloat16, device=device)
                hh = torch.zeros(n, cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=device)
                pm = torch.zeros(n, cfg.hc_mult, dtype=torch.float32, device=device)
                cm = torch.zeros(n, cfg.hc_mult, cfg.hc_mult, dtype=torch.float32, device=device)
                nw_h = torch.ones(cfg.hidden_size, dtype=torch.bfloat16, device=device)
                li = torch.arange(n, dtype=torch.int64, device=device)
                if cfg.hc_mult == 4:
                    fast_triton.hc_epilogue_norm(xh, hh, pm, cm, pm, nw_h, eps, None)
                    fast_triton.hc_epilogue_norm(xh, hh, pm, cm, pm, nw_h, eps, li)
                if win == 128 and (not cfg.engram_layer_ids or cfg.engram_max_ngram_size == 4):
                    has_eng = bool(cfg.engram_layer_ids)
                    num_p = max(1, -(-self.max_context_len // page))
                    mb = fast_triton.alloc_decode_meta_buffers(n, win, num_p, has_eng, device)
                    st_d = torch.zeros(n + 2, dtype=torch.int32, device=device)
                    r2t_d = torch.zeros((n + 2, num_p * page), dtype=torch.int32, device=device)
                    rows_d = torch.ones(n, dtype=torch.int64, device=device)
                    tmap_d = self.constant("token_map", device) if has_eng else None
                    tail_d = torch.zeros((n + 2, 3), dtype=torch.int64, device=device) if has_eng else None
                    fast_triton.decode_begin_meta(
                        rows_d, pos, pos, st_d, r2t_d, page, win, mb, tmap_d, tail_d, 0
                    )
                    idx_map_d = torch.zeros((n, max(128, -(-cfg.index_topk // 128) * 128)), dtype=torch.int32, device=device)
                    fast_triton.map_logical_to_req_tokens_inplace(idx_map_d, rows_d, r2t_d)
            except Exception:  # noqa: BLE001
                pass
            cos_sin = self.cos_sin(0, device)
            norm_hd = torch.ones(hd, dtype=torch.bfloat16, device=device)
            x_hd = torch.zeros(n, hd, dtype=torch.bfloat16, device=device)
            rows_neg1 = torch.full((n,), -1, dtype=torch.int64, device=device)
            ring_scratch = torch.zeros(win, hd + hd // formats.BLOCK, dtype=torch.uint8, device=device)
            seg = torch.tensor([[0, 0, min(n, win), 0]], dtype=torch.int32, device=device)
            self.kernels("prefill", "ds.transform@1")(x_hd, norm_hd, eps, cos_sin, pos, rd, "fp8")
            self.kernels("prefill", "ds.store_cache@1")(x_hd, None, eps, None, None, 0, ring_scratch, rows_neg1, "fp8",
                                                        win, False)
            self.kernels("prefill", "ds.load_values@1")(ring_scratch, win, seg, min(n, win), "fp8")
            q = torch.zeros(n, cfg.num_attention_heads, hd, dtype=torch.bfloat16, device=device)
            src = torch.zeros(16, hd, dtype=torch.bfloat16, device=device)
            idx = torch.zeros(n, 8, dtype=torch.int64, device=device)
            sink = torch.zeros(cfg.num_attention_heads, dtype=torch.float32, device=device)
            wo_a = torch.zeros(cfg.o_groups * cfg.o_lora_rank, cfg.num_attention_heads * hd // cfg.o_groups,
                               dtype=torch.bfloat16, device=device)
            wo_b = torch.zeros(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank, dtype=torch.uint8, device=device)
            so_b = torch.full((-(-cfg.hidden_size // 32), -(-(cfg.o_groups * cfg.o_lora_rank) // 32)), 127,
                              dtype=torch.uint8, device=device)
            po_b = formats.pack_scales(formats.block_rows(so_b, cfg.hidden_size))
            self.kernels("prefill", "ds.sparse_attn_out@1")(
                q, cos_sin, pos, rd, src, idx, sink, hd ** -0.5, cfg.o_groups, wo_a, wo_b, so_b, po_b)
            ratios = sorted({r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r})
            if ratios:
                ih, idim = cfg.index_n_heads, cfg.index_head_dim
                iq = torch.full((n, ih, idim), 0.01, dtype=torch.bfloat16, device=device)
                q_data, q_scales = self.kernels("prefill", "ds.quant_index_q@1")(iq, cos_sin, pos, rd)
                iw = torch.full((n, ih), 0.01, dtype=torch.bfloat16, device=device)
                norm_ik = torch.ones(idim, dtype=torch.bfloat16, device=device)
                x_ik = torch.full((n, idim), 0.01, dtype=torch.bfloat16, device=device)
                for ratio in ratios:
                    entries = page // ratio
                    kv_s = torch.zeros(entries, hd // 2 + hd // formats.LATENT_BLOCK, dtype=torch.uint8, device=device)
                    ik_s = torch.zeros(entries, idim // 2 + idim // formats.BLOCK, dtype=torch.uint8, device=device)
                    seg_r = torch.tensor([[0, 0, min(n, entries), 0]], dtype=torch.int32, device=device)
                    self.kernels("prefill", "ds.store_cache@1")(x_hd, norm_hd, eps, cos_sin, pos, rd, kv_s, rows_neg1,
                                                                "latent", entries, True)
                    self.kernels("prefill", "ds.store_cache@1")(x_ik, norm_ik, eps, cos_sin, pos, rd, ik_s, rows_neg1,
                                                                "key", entries, False)
                    self.kernels("prefill", "ds.load_values@1")(kv_s, entries, seg_r, min(n, entries), "latent")
                    self.kernels("prefill", "ds.load_keys@1")(ik_s, entries, seg_r, min(n, entries), idim)
                    if ratio > 1:
                        c_score = torch.zeros(ratio, hd, dtype=torch.float32, device=device)
                        c_kv = torch.zeros(ratio, hd, dtype=torch.float32, device=device)
                        c_state = torch.zeros(1, ratio - 1, 2, hd, dtype=torch.float32, device=device)
                        z0 = torch.zeros(1, dtype=torch.int64, device=device)
                        cu = torch.tensor([0, ratio], dtype=torch.int64, device=device)
                        cu_c = torch.tensor([0, 1], dtype=torch.int64, device=device)
                        self.kernels("prefill", "ds.compress@1")(c_score, c_kv, c_state, z0, cu, cu, cu_c, z0, ratio)
                modes = sorted({
                    (cfg.ratio(i), "source" if i == cfg.candidate_source_layer_id
                     else "user" if 0 <= cfg.candidate_source_layer_id < i else "dense")
                    for i in cfg.index_source_layer_ids if i < cfg.num_hidden_layers
                })
                for ratio, mode in modes:
                    m_keys = max(n, cfg.candidate_block_size)
                    kd = torch.zeros(m_keys, idim // 2, dtype=torch.uint8, device=device)
                    ks_s = torch.full((m_keys, idim // formats.BLOCK), 127, dtype=torch.uint8, device=device)
                    ks = torch.zeros(n, dtype=torch.int64, device=device)
                    ke = torch.arange(1, n + 1, dtype=torch.int64, device=device).clamp_max(m_keys)
                    cand: torch.Tensor | None = None
                    if mode == "user":
                        cand = torch.full((n, cfg.candidate_topk_blocks), -1, dtype=torch.int64, device=device)
                        cand[:, 0] = 0
                    self.kernels("prefill", "ds.index_topk@1")(
                        q_data, q_scales, iw, kd, ks_s, ks, ke, self.max_context_len // ratio,
                        min(cfg.index_topk, m_keys), mode, cand, cfg.candidate_topk_blocks, cfg.candidate_block_size)
            z_lin = torch.zeros((1, 16), dtype=torch.float32, device=device)
            torch.nn.functional.linear(z_lin, z_lin)
            del z_lin
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._warmed_prefill_version = self.kernels.version

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        cfg, page = self.cfg, self.kv.page_size
        ratios = {r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r}
        # DeepGEMM's paged indexer takes blocks of 32, 64 or 128 keys: a page holds page / ratio.
        blocks = {page // r for r in ratios}
        # DeepGEMM's fused MoE and Hyper-Connections kernels: four copies of the stream, widths their
        # tiles divide (the stream in pieces of 1024 values, the scales of 128 rows together).
        mega = cfg.hc_mult == 4 and cfg.hidden_size % 1024 == 0 and (2 * cfg.moe_intermediate_size) % 128 == 0
        mega_attn = (mega and cfg.num_attention_heads in (64, 128) and cfg.head_dim == 512
                     and cfg.qk_rope_head_dim == 64 and cfg.q_lora_rank >= 128 and cfg.q_lora_rank % 128 == 0
                     and cfg.num_attention_heads % cfg.o_groups == 0
                     and (cfg.num_attention_heads * cfg.head_dim // cfg.o_groups) % 128 == 0
                     and cfg.o_lora_rank >= 32 and cfg.o_lora_rank % 32 == 0
                     and (cfg.o_groups * cfg.o_lora_rank) % 128 == 0)
        mega_engram = mega and cfg.engram_head_dim >= 128 and cfg.engram_head_dim % 128 == 0
        shape = {"heads": cfg.num_attention_heads, "head_dim": cfg.head_dim, "index_head_dim": cfg.index_head_dim,
                 "index_block": min(blocks) if blocks and blocks <= {32, 64, 128} else 0, "mega": int(mega),
                 "mega_attn": int(mega_attn), "mega_engram": int(mega_engram)}
        names = self.definitions()
        calls = {"prefill": {name: shape for name in names if name not in DECODE_ONLY},
                 "decode": {name: shape for name in names if name not in PREFILL_ONLY}}
        # Among ranks the Engram rows are brought in before the forward (``fetch``) unless the backend
        # stages its own capturable fetch inside ``begin``.
        self.chosen = self.kernels.resolve(calls, outside=frozenset({"ds.engram_fetch@1"}))
        self._sync_flags()
        self._warm_prefill()
        return self.chosen

    def mega_experts(self) -> bool:
        """Whether the routed experts' kernel reads the expert stacks in the fused MoE kernel's row
        order: the model lays them out so (``formats.expert_layout``). One answer for every step."""
        if not self.chosen:
            raise RuntimeError("the layers' kernels are not resolved yet (core.plan does it at start)")
        answers = {solution.split(" ")[0] in MEGA for table in self.chosen.values()
                   for definition, solution in table.items() if definition == "ds.moe@1" and solution}
        if len(answers) > 1:
            raise ValueError("the routed experts' kernels of prefill and decode steps read the experts' weights "
                             "in different row orders: give both the same backend (dsv4.prefill_backend, "
                             "dsv4.decode_backend)")
        return answers == {True}

    def definitions(self) -> tuple[str, ...]:
        """What a step calls: ``DEFINITIONS``, and among ranks that share the Engram tables the
        kernel that brings their rows in (the MoE's kernel reaches the other ranks' experts itself)."""
        if self.group.size == 1 or not self.cfg.engram_layer_ids:
            return DEFINITIONS
        return (*DEFINITIONS, "ds.engram_fetch@1")

    def k(self, definition: str) -> KernelFn:
        return self.kernels(self.table, definition)

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        if plan.draft_tokens:
            raise ValueError("layers.ds1017 runs no speculative step yet")
        self._sync_flags()
        self.group.uniform_prefill = plan.uniform_prefill and key.phase == "prefill"
        self.table = GRAPH if key.bucket else key.phase
        self.bucket = key.bucket
        self.batch_tables = batch.binding.buffers()
        device = self.batch_tables["req_to_token"].device
        rows = [int(r) + 1 for r in plan.rows.values()]
        counts = [int(n) for n in plan.num_tokens.values()]
        starts = [int(p) for p in plan.positions.values()]
        self.step = {}
        self._full_kv_stored = False
        self._replay_rows = None
        self._replay = None
        has_yoco = key.phase != "decode" and self.yoco_mode != "off" and bool(len(plan.prompt_lengths))
        enc_ratios = self._encoder_ratios if has_yoco else self._all_ratios
        self._snapshot = (PrefixSnapshot.prepare(batch, plan, self.cfg, device, self.slots, enc_ratios)
                          if key.phase != "decode" else None)
        if key.phase == "decode":
            self.layout = None
            if key.bucket:
                buf = self.fixed.get(key.bucket)
                vbuf = self.valid_fixed.get(key.bucket)
                if buf is None or vbuf is None:
                    buf = self.fixed[key.bucket] = torch.zeros(key.bucket, dtype=torch.int64, device=device)
                    vbuf = self.valid_fixed[key.bucket] = torch.zeros(key.bucket, dtype=torch.bool, device=device)
                    self.valid_count.pop(key.bucket, None)
                    self.generation += 1
                n_r = len(rows)
                if device.type == "cuda":
                    pin_ring = self.fixed_pin.get(key.bucket)
                    if pin_ring is None:
                        pin_ring = self.fixed_pin[key.bucket] = torch.empty(
                            (8, key.bucket), dtype=torch.int64, pin_memory=True)
                    pin_slot = pin_ring[self._pin_turn & 7]
                    self._pin_turn += 1
                    padded_rows = rows if n_r == key.bucket else [*rows, *([0] * (key.bucket - n_r))]
                    pin_slot.copy_(torch.tensor(padded_rows, dtype=torch.int64))
                    buf.copy_(pin_slot, non_blocking=True)
                else:
                    buf[:n_r].copy_(on_device(torch.tensor(rows, dtype=torch.int64), device))
                    if n_r < key.bucket:
                        buf[n_r:].zero_()
                if self.valid_count.get(key.bucket) != n_r:
                    vbuf[:n_r].fill_(True)
                    if n_r < key.bucket:
                        vbuf[n_r:].fill_(False)
                    self.valid_count[key.bucket] = n_r
                given = buf
                self.decode_valid = vbuf
            else:
                given = on_device(torch.tensor(rows, dtype=torch.int64), device)
                self.decode_valid = None
            self.decode_rows = given
            if device.type == "cuda":
                self.constant("arange.win32", device)
                self.constant("scalar.neg1_i32", device)
                self.constant("arange.page_cols", device)
                if self.cfg.engram_layer_ids:
                    self.constant("arange.ng", device)
            ratios = {r for r in self.cfg.compress_ratios[:self.cfg.num_hidden_layers] if r}
            self.extent = {r: self.max_context_len // r if key.bucket else max((p + 1) // r for p in starts)
                           for r in ratios}
        else:
            self.decode_rows = None
            self.decode_valid = None
            # In a mixed step, a row that goes on from the token it sampled is decoding.
            decoding = [bool(f) and n == 1 for f, n in zip(plan.from_previous.values(), counts, strict=True)]
            self.layout = self.lay_out(rows, counts, starts, decoding, device)
            if has_yoco:
                lengths = [int(n) for n in plan.prompt_lengths.values()]
                if len(lengths) != len(rows) or any(n <= 0 for n in lengths):
                    raise ValueError("bounded replay needs a positive prompt length per row")
                self._replay_rows = rows, counts, starts, decoding, lengths
                self._replay = yoco.retained(rows, counts, starts, lengths, self.yoco_keep, device,
                                             compact=self.yoco_mode == "compact")

    def set_yoco_paged_consumer(self, value: Any) -> None:
        self.yoco_paged_consumer = bool(value)

    def compacts_yoco_boundary(self, total_tokens: int) -> bool:
        if self.yoco_mode != "compact" or self._replay is None:
            return False
        return sum(self._replay.counts) < total_tokens

    def end(self) -> None:
        self.step.clear()
        self.fetched = None

    # ------------------------------------------------------------------ a step laid out on the host

    def lay_out(self, rows: Sequence[int], counts: Sequence[int], starts: Sequence[int], decoding: Sequence[bool],
                device: torch.device, lower: Sequence[int] | None = None,
                consumer_of: tuple[Layout, torch.Tensor | None] | None = None) -> Layout:
        """The step's index tensors. Per row the plan's few numbers go to the device, and every per-token
        tensor is computed there: on the host the ``[tokens, window]`` arithmetic of a 32K-token step
        took over a second (2026-11-16, SM100), longer than the step's forward. Nothing is read back from
        the device, sizes included (``torch.nonzero``): that would wait for the work queued there, with
        ``sched.overlap`` for the step still running."""
        cfg, w = self.cfg, self.cfg.sliding_window
        total = sum(counts)

        def put(values: Sequence[Any] | torch.Tensor, dtype: torch.dtype = torch.int64) -> torch.Tensor:
            return on_device(values if isinstance(values, torch.Tensor) else torch.tensor(list(values), dtype=dtype),
                             device)

        rows_host = torch.tensor(list(rows), dtype=torch.int64)
        n_host = torch.tensor(list(counts), dtype=torch.int64)
        p0_host = torch.tensor(list(starts), dtype=torch.int64)
        base_host = torch.cumsum(n_host, 0) - n_host
        kept_host = n_host.clamp_max(w)  # a row keeps the last tokens of its chunk, a window of them at most
        kept_total = int(kept_host.sum())
        kept_base_host = torch.cumsum(kept_host, 0) - kept_host
        # The rings of the rows with earlier visible tokens are loaded whole, one after the other.
        earlier = [i for i, p in enumerate(starts) if p > 0 and (lower is None or decoding[i] or lower[i] < p)]
        ring_base_host = torch.zeros(len(rows), dtype=torch.int64)
        ring_base_host[earlier] = torch.arange(len(earlier), dtype=torch.int64) * w
        ring_total = len(earlier) * w

        lower_host = torch.tensor(lower, dtype=torch.int64) if lower is not None else torch.zeros_like(p0_host)
        row_pack = put(torch.stack([rows_host, n_host, p0_host, base_host, kept_host, kept_base_host, ring_base_host,
                                    torch.tensor(decoding, dtype=torch.int64), lower_host]))
        rows_dev, n, p0, base, kept, kept_base, ring_base = row_pack[:7].unbind()
        seq_arange = torch.arange(len(rows), device=device)
        if device.type == "cuda" and total and w == 128 and self.deepinfer_index.get(self.table, False):
            from kernels.ds1017_triton import prefill_window_meta

            meta, window_index, keep = prefill_window_meta(row_pack, total, max(counts), kept_total, ring_total, w)
            tok_seq, tok_row, tok_p0, tok_base, pos = meta.unbind()
        else:
            tok_seq = torch.repeat_interleave(seq_arange, n, output_size=total)
            tok_row, tok_p0, tok_base = rows_dev[tok_seq], p0[tok_seq], base[tok_seq]
            pos = tok_p0 + torch.arange(total, device=device) - tok_base

            ahead = ~row_pack[7][tok_seq].bool()
            pos32 = pos.to(torch.int32)
            tok_p0_32 = tok_p0.to(torch.int32)
            tok_base_32 = tok_base.to(torch.int32)
            col32 = self.constant("arange.win32", device)
            visible_start = (pos32 - w + 1).clamp_min(0)
            if lower is not None:
                visible_start = torch.maximum(visible_start, row_pack[8][tok_seq].int())
            keys32 = torch.where(
                ahead[:, None], visible_start[:, None] + col32, pos32[:, None] - w + 1 + col32)
            valid = (keys32 >= 0) & (keys32 <= pos32[:, None])
            in_chunk = keys32 >= tok_p0_32[:, None]
            chunk32 = (tok_base_32[:, None] + keys32 - tok_p0_32[:, None]).clamp(0, max(total - 1, 0))
            kept_seq = torch.repeat_interleave(seq_arange, kept, output_size=kept_total)
            keep = ((base + n - kept - kept_base)[kept_seq] + torch.arange(kept_total, device=device))
            window_index = torch.where(
                valid,
                torch.where(in_chunk, ring_total + chunk32, ring_base.to(torch.int32)[tok_seq][:, None] + keys32 % w),
                -1,
            ).to(torch.int32)
        slot_table = self.batch_tables[f"slots.{SLOT_KIND}"]
        if earlier:
            ring_pair = put(torch.stack([rows_host[earlier], ring_base_host[earlier]]))
            ring_rows, ring_bases = ring_pair[0], ring_pair[1]
            ring_segments = torch.stack([slot_table[ring_rows].long(), torch.zeros_like(ring_rows),
                                         torch.full_like(ring_rows, w), ring_bases], dim=1).int()
        else:
            ring_segments = torch.empty((0, 4), dtype=torch.int32, device=device)

        latents: dict[int, Latents] = {}
        table = self.batch_tables["req_to_token"]
        page = self.kv.page_size
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        cu_q_host = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(n_host, 0)]) if consumer_of is None else None
        cu_k_host = (torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(p0_host + n_host, 0)])
                     if consumer_of is None else None)
        end_host = p0_host + n_host
        for ratio in self._all_ratios:
            if consumer_of is not None:
                if ratio == 1:
                    full, indices = consumer_of
                    source = full.latents[ratio]
                    ks = source.ks if indices is None else yoco.gather(source.ks, indices)
                    ke = source.ke if indices is None else yoco.gather(source.ke, indices)
                    latents[ratio] = Latents(source.segments, source.total, source.width, ks, ke)
                continue
            entries = page // ratio
            keys_host = torch.div(end_host, ratio, rounding_mode="floor")
            seg_counts = torch.div(keys_host + entries - 1, entries, rounding_mode="floor")
            row_k_host = keys_host.cumsum(0) - keys_host
            key_count, num_segs = int(keys_host.sum()), int(seg_counts.sum())
            width = int(keys_host.max()) if len(rows) else 0
            if num_segs:
                key_counts, row_k_dev, seg_n, seg_base = put(torch.stack(
                    [keys_host, row_k_host, seg_counts, seg_counts.cumsum(0) - seg_counts])).unbind()
                seg_seq = torch.repeat_interleave(seq_arange, seg_n, output_size=num_segs)
                page_no = torch.arange(num_segs, device=device) - seg_base[seg_seq]
                first_key = page_no * entries
                page_ids = torch.div(table[rows_dev[seg_seq], page_no * page].long(), page, rounding_mode="floor")
                segments = torch.stack((page_ids, torch.zeros_like(page_ids),
                                        (key_counts[seg_seq] - first_key).clamp_max(entries),
                                        row_k_dev[seg_seq] + first_key), dim=1).int()
            else:
                row_k_dev = put(row_k_host)
                segments = torch.empty((0, 4), dtype=torch.int32, device=device)
            ks = row_k_dev[tok_seq]
            ke = ks + (pos + 1) // ratio
            if ratio == 1:
                latents[ratio] = Latents(segments, key_count, width, ks, ke)
                continue
            assert cu_q_host is not None and cu_k_host is not None
            done_host = keys_host - torch.div(p0_host, ratio, rounding_mode="floor")
            reserved_host = done_host.clamp_min(1)
            cu_c_host = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(reserved_host, 0)])
            total_c = int(cu_c_host[-1])
            done_t, reserved = put(torch.stack([done_host, reserved_host])).unbind()
            cu_q, cu_k, cu_c = put(torch.stack([cu_q_host, cu_k_host, cu_c_host])).unbind()
            c_seq = torch.repeat_interleave(seq_arange, reserved, output_size=total_c)
            j = torch.arange(total_c, device=device) - cu_c[:-1][c_seq]
            real = j < done_t[c_seq]
            first = torch.where(real, (p0[c_seq] // ratio + j) * ratio, 0)
            groups = CompressionGroups(base, cu_q, cu_k, cu_c, c_seq, first, real, base[c_seq])
            latents[ratio] = Latents(segments, key_count, width, ks, ke, groups)

        if consumer_of is None and cfg.engram_layer_ids:
            ng = max(cfg.engram_max_ngram_size, 1)
            back = pos[:, None] - torch.arange(ng, device=device)
            ngram_from = torch.where(back < 0, 2, torch.where(back >= tok_p0[:, None], 0, 1))
            ngram = torch.where(ngram_from == 0, tok_base[:, None] + back - tok_p0[:, None], tok_p0[:, None] - 1 - back).clamp_min(0)
            row_end = p0 + n - 1
            after = row_end[:, None] - torch.arange(max(ng - 1, 1), device=device)
            tail_from = torch.where(after < 0, 2, torch.where(after >= p0[:, None], 0, 1))
            tail = torch.where(tail_from == 0, base[:, None] + after - p0[:, None], p0[:, None] - 1 - after).clamp_min(0)
        else:
            ngram = ngram_from = tail = tail_from = empty_i64

        return Layout(rows=tok_row, ring_segments=ring_segments, ring_total=ring_total,
                      window_index=window_index, keep=keep, latents=latents,
                      pos=pos, start=tok_p0, ngram=ngram, ngram_from=ngram_from,
                      last=base + n - 1, tail=tail, tail_from=tail_from)

    def _suffix_window(self, rows: int, length: int, win: int, device: torch.device
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (rows, length, win, device)
        if self._suffix_shape != shape or self._suffix_indices is None:
            q = torch.arange(length, dtype=torch.int32, device=device)[:, None]
            col = torch.arange(win, dtype=torch.int32, device=device)[None, :]
            base = torch.arange(rows, dtype=torch.int32, device=device) * length
            keys = base[:, None, None] + (q + 1 - win).clamp_min(0) + col
            window = torch.where(col <= q, keys, -1).reshape(rows * length, win).contiguous()
            keep = (base.long()[:, None] + torch.arange(max(0, length - win), length, device=device)).flatten()
            cnt = (q.squeeze(1) + 1).clamp_max(win).repeat(rows).contiguous()
            self._suffix_indices, self._suffix_shape = (window, keep, cnt), shape
        return self._suffix_indices

    def consumer_inputs(self, y: torch.Tensor, h: torch.Tensor, pre: torch.Tensor,
                        post: torch.Tensor, comb: torch.Tensor, last_index: torch.Tensor,
                        weights: Weights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor, torch.Tensor]:
        """DeepInfer compact_yoco_boundary / _store_yoco_full_kv at the first decoder layer."""
        if self._replay_rows is None or self._replay is None:
            return y, h, pre, post, comb, last_index
        rows, counts, starts, decoding, _ = self._replay_rows
        replay = self._replay
        pool = self.kv.bind(self.yoco_boundary).buffers()
        self._compress(self.yoco_boundary, y, weights, pool, self.kv.page_size, fuse_index=False)
        self._snapshot = None
        old = self.step
        full = self.layout
        assert isinstance(full, Layout)
        indices = replay.indices if replay.counts != counts else None
        single_slice = (
            indices is not None
            and len(rows) == 1
            and not decoding[0]
            and replay.counts[0] > 0
            and replay.starts[0] - starts[0] == counts[0] - replay.counts[0]
        )
        cut = counts[0] - replay.counts[0] if single_slice else 0
        fields = ("rows", "pos", "pos32", "slot", "slot32")
        if single_slice:
            self.step = {k: old[k][cut:] for k in fields}
        elif indices is not None:
            self.step = {k: yoco.gather(old[k], indices) for k in fields}
        else:
            self.step = {k: old[k] for k in fields}
        step = self.step
        win = self.cfg.sliding_window
        whole_suffix = (self.yoco_mode == "compact" and not any(decoding) and replay.starts == replay.lower
                        and all(n == self.yoco_keep for n in replay.counts))
        if whole_suffix:
            window, keep, win_cnt = self._suffix_window(len(rows), self.yoco_keep, win, y.device)
            src = full.latents[1]
            if single_slice:
                ks, ke = src.ks[cut:], src.ke[cut:]
            elif indices is not None:
                ks, ke = yoco.gather(src.ks, indices), yoco.gather(src.ke, indices)
            else:
                ks, ke = src.ks, src.ke
            self.layout = AttentionLayout(
                step["rows"], full.ring_segments[:0], 0, window, keep,
                {1: Latents(src.segments, src.total, src.width, ks, ke)}, win_cnt)
            if len(rows) == 1 and self.yoco_keep >= win:
                keep_slice: torch.Tensor | slice = slice(self.yoco_keep - win, self.yoco_keep)
                step["keep_idx"] = keep_slice
                step["keep_ring_slot32"] = (
                    step["slot32"][keep_slice] * win + step["pos32"][keep_slice] % win).contiguous()
            else:
                step["keep_idx"] = keep
                step["keep_ring_slot32"] = (step["slot32"][keep] * win + step["pos32"][keep] % win).contiguous()
            c_total = len(rows) * self.yoco_keep
            if (
                self.yoco_paged_consumer
                and c_total > 0
                and src.total >= 1_500_000
                and y.device.type == "cuda"
                and self.mega_decode_attn.get("decode", False)
                and self.deepinfer_index.get(self.table, False)
                and win % 128 == 0
                and self.cfg.candidate_topk_blocks * self.cfg.candidate_block_size >= self.cfg.index_topk
            ):
                dim = self.cfg.head_dim
                swa_blocks = max(1, -(-c_total // win))
                step["paged_consumer"] = True
                step["swa_fp8"] = torch.empty(
                    (swa_blocks, win, dim + dim // formats.BLOCK), dtype=torch.uint8, device=y.device)
                step["swa_self_slot32"] = torch.arange(c_total, dtype=torch.int32, device=y.device)
                step["swa_ring_slots"] = torch.empty(0, dtype=torch.int64, device=y.device)
                step["keep_swa_src"] = keep
                step["keep_dst_safe"] = torch.where(
                    step["keep_ring_slot32"] >= 0, step["keep_ring_slot32"].long(), 0).contiguous()
                step["swa_win_idx32"] = window
                step["swa_win_cnt32"] = win_cnt
                target_k = min(self.cfg.index_topk, src.width)
                step["chosen_cnt_1"] = (step["pos32"] + 1).clamp_max(target_k).contiguous()
        else:
            lay = self.lay_out(rows, replay.counts, replay.starts, decoding, y.device, replay.lower, (full, indices))
            self.layout, keep = lay, lay.keep
            lower = on_device(torch.tensor(replay.lower, dtype=torch.int64), y.device)
            lower = torch.repeat_interleave(lower, on_device(torch.tensor(replay.counts), y.device),
                                           output_size=sum(replay.counts))
            positions = step["pos"][keep]
            destinations = step["slot"][keep] * win + positions % win
            step["keep_idx"] = keep
            step["keep_ring_slot32"] = torch.where(positions >= lower[keep], destinations, -1).int().contiguous()
        self._full_kv_stored = True
        self._replay_rows = None
        self._replay = None
        if single_slice:
            y, h, pre, post, comb = y[cut:], h[cut:], pre[cut:], post[cut:], comb[cut:]
            last_index = (last_index - cut).clamp(0, replay.counts[0])
        elif indices is not None:
            y, h, pre, post, comb = (yoco.gather(t, indices) for t in (y, h, pre, post, comb))
            last_index = torch.searchsorted(indices, last_index)
        return y, h, pre, post, comb, last_index

    # ------------------------------------------------------------------ constants

    def constant(self, name: str, device: torch.device) -> torch.Tensor:
        key = f"{name}@{device}"
        if key not in self.constants:
            cfg = self.cfg
            if name in ("cos_sin.window", "cos_sin.compress"):
                compress = name == "cos_sin.compress"
                freqs = formats.rope_table(
                    cfg.qk_rope_head_dim, self.max_context_len, cfg.rope_original_seq_len if compress else 0,
                    cfg.compress_rope_theta if compress else cfg.rope_theta, cfg.rope_factor, cfg.rope_beta_fast,
                    cfg.rope_beta_slow, device)
                value = torch.cat([freqs.real, freqs.imag], dim=-1).contiguous()  # DeepInfer's cos_sin_cache
            elif name == "token_map":
                value = torch.tensor(self.token_map_host, dtype=torch.int64, device=device)
            elif name == "arange.win":
                if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    return torch.arange(cfg.sliding_window, device=device)
                value = torch.arange(cfg.sliding_window, device=device)
            elif name == "arange.win32":
                if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    return torch.arange(cfg.sliding_window, dtype=torch.int32, device=device)
                value = torch.arange(cfg.sliding_window, dtype=torch.int32, device=device)
            elif name == "arange.ng":
                if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    return torch.arange(cfg.engram_max_ngram_size, device=device)
                value = torch.arange(cfg.engram_max_ngram_size, device=device)
            elif name == "scalar.neg1_i32":
                if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    return torch.full((), -1, dtype=torch.int32, device=device)
                value = torch.full((), -1, dtype=torch.int32, device=device)
            elif name == "arange.page_cols":
                page = self.kv.page_size
                cols = -(-self.max_context_len // page) * page
                if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    return torch.arange(0, cols, page, dtype=torch.int64, device=device)
                value = torch.arange(0, cols, page, dtype=torch.int64, device=device)
            elif name in ("engram.primes32", "engram.offsets32"):
                primes, offsets, _ = engram_layout(cfg)
                raw = primes if name == "engram.primes32" else offsets
                value = torch.tensor(raw, dtype=torch.int32, device=device).contiguous()
            else:
                primes, offsets, multipliers = engram_layout(cfg)
                value = torch.tensor({"engram.primes": primes, "engram.offsets": offsets,
                                      "engram.multipliers": multipliers}[name], dtype=torch.int64, device=device)
            self.constants[key] = value
        return self.constants[key]

    # ------------------------------------------------------------------ the forward

    def begin(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Once per forward, before the layers: what every layer shares, and the Engram hash ids."""
        tables = self.batch_tables
        slot_table = tables[f"slots.{SLOT_KIND}"]
        rows = self.layout.rows if self.layout is not None else self.decode_rows
        assert rows is not None
        win = self.cfg.sliding_window
        if (
            self.layout is None
            and positions.device.type == "cuda"
            and self.mega_decode_attn.get(self.table, False)
            and win == 128
            and (not self.cfg.engram_layer_ids or self.cfg.engram_max_ngram_size == 4)
            and set(self.cfg.compress_ratios[:self.cfg.num_hidden_layers]) <= {0, 1, 2}
        ):
            from kernels import ds1017_triton as fast_triton

            device = positions.device
            page = self.kv.page_size
            r2t = tables["req_to_token"]
            num_pages = -(-r2t.shape[1] // page)
            has_eng = bool(self.cfg.engram_layer_ids)
            n_b = rows.shape[0]
            if self.bucket:
                bkey = (self.bucket, num_pages, has_eng)
                bufs = self._decode_meta_bufs.get(bkey)
                if bufs is None:
                    bufs = self._decode_meta_bufs[bkey] = fast_triton.alloc_decode_meta_buffers(
                        self.bucket, win, num_pages, has_eng, device
                    )
            else:
                bufs = fast_triton.alloc_decode_meta_buffers(n_b, win, num_pages, has_eng, device)
            tmap = self.constant("token_map", device) if has_eng else None
            tail = self.slots.bind(self.engram_first).buffers()["tail"] if has_eng else None
            pad_comp = int(self.token_map_host[self.cfg.engram_pad_token_id]) if has_eng else 0
            fast_triton.decode_begin_meta(
                rows, positions, input_ids, slot_table, r2t, page, win, bufs, tmap, tail, pad_comp
            )
            self.step = {
                "rows": rows,
                "pos": positions,
                "pos32": bufs["pos32"],
                "pos1_32": bufs["pos1_32"],
                "slot": bufs["slot"],
                "slot32": bufs["slot32"],
                "tok_slot32": bufs["tok_slot32"],
                "ring_slot32": bufs["ring_slot32"],
                "window": bufs["window"],
                "window_cnt": bufs["window_cnt"],
                "pages32": bufs["pages32"],
                ("lens32", 1): bufs["lens32_1"],
                ("lens32", 2): bufs["lens32_2"],
                ("comp_dest", 2): (bufs["comp_fc32"], bufs["comp_dc32"]),
            }
            if has_eng:
                hashes = self.k("ds.engram_hash@1")(
                    bufs["engram_tok32"],
                    self.constant("engram.multipliers", device),
                    self.constant("engram.primes32", device),
                    self.constant("engram.offsets32", device),
                )
                if self.group.size == 1:
                    self.step["hashes"] = hashes
                elif self.engram_staged.get(self.table, False):
                    self.k("ds.engram_fetch@1")(self.group, hashes, *self.tables(), True)
            return

        slot_raw = slot_table[rows]
        slot32 = (slot_raw if slot_raw.dtype == torch.int32 and slot_raw.is_contiguous()
                  else slot_raw.to(torch.int32).contiguous())
        slot = slot32.long()
        pos32 = positions.to(torch.int32).contiguous()
        tok_slot = tables["req_to_token"][rows, positions]
        tok_slot32 = (tok_slot if tok_slot.dtype == torch.int32 and tok_slot.is_contiguous()
                      else tok_slot.to(torch.int32).contiguous())
        self.step = {
            "rows": rows, "pos": positions, "pos32": pos32,
            "slot": slot, "slot32": slot32,
            "tok_slot32": tok_slot32,
        }
        if self.layout is None:
            self.step["ring_slot32"] = (slot32 * win + pos32 % win).contiguous()
            if self.mega_decode_attn.get(self.table, False) and win % 128 == 0:
                at32 = pos32[:, None] - self.constant("arange.win32", positions.device)
                neg1 = self.constant("scalar.neg1_i32", positions.device)
                self.step["window"] = torch.where(
                    at32 >= 0, slot32[:, None] * win + at32.clamp_min(0) % win, neg1).contiguous()
                self.step["window_cnt"] = (pos32 + 1).clamp(0, win).contiguous()
            else:
                at = positions[:, None] - win + 1 + self.constant("arange.win", positions.device)
                self.step["window"] = torch.where(at >= 0, slot[:, None] * win + at % win, -1)
                self.step["window_cnt"] = None
            page = self.kv.page_size
            max_cols = tables["req_to_token"].shape[1]
            page_cols = self.constant("arange.page_cols", positions.device)
            if page_cols.shape[0] * page != max_cols:
                page_cols = torch.arange(0, max_cols, page, dtype=torch.int64, device=positions.device)
            self.step["pages32"] = torch.div(
                tables["req_to_token"][rows[:, None], page_cols], page, rounding_mode="floor"
            ).to(torch.int32).contiguous()
        else:
            lay = self.layout
            run_keep_ring32 = (slot[lay.keep] * win + positions[lay.keep] % win).to(torch.int32)
            keep_idx: torch.Tensor | slice = lay.keep
            if self._snapshot is not None:
                snap = self._snapshot
                keep_idx = torch.cat([lay.keep, snap.ring_tok])
                run_keep_ring32 = torch.cat([run_keep_ring32, snap.ring_dst32])
            self.step["keep_idx"] = keep_idx
            self.step["keep_ring_slot32"] = run_keep_ring32.contiguous()
        if self.cfg.engram_layer_ids:
            if self.group.size == 1:
                self.step["hashes"] = self._hashes(input_ids, positions, self.step["slot"])
            elif self.engram_staged.get(self.table, False):
                hashes = self._hashes(input_ids, positions, self.step["slot"])
                self.k("ds.engram_fetch@1")(self.group, hashes.long(), *self.tables(), True)

    def fetch(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """After ``prepare`` and before the forward, outside any graph, with the step's inputs (a
        bucket's padded ones): among ranks that share the Engram tables, the rows this step's tokens
        hash to, from the ranks that hold them, when the backend's fetch is not staged in the forward."""
        if self.group.size == 1 or not self.cfg.engram_layer_ids or self.engram_staged.get(self.table, False):
            return
        rows = self.layout.rows if self.layout is not None else self.decode_rows
        assert rows is not None
        slot = self.batch_tables[f"slots.{SLOT_KIND}"][rows].long()
        hashes = self._hashes(input_ids, positions, slot)
        held, scales = self.kernels("decode", "ds.engram_fetch@1")(self.group, hashes.long(), *self.tables())
        if self.bucket:
            fixed = self.fetched_fixed.get(self.bucket)
            if fixed is None:
                fixed = self.fetched_fixed[self.bucket] = (held.clone(), scales.clone())
                self.generation += 1
            else:
                fixed[0].copy_(held)
                fixed[1].copy_(scales)
            held, scales = fixed
        self.fetched = (held, scales)

    def _hashes(self, input_ids: torch.Tensor, positions: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """``NgramHashState.forward``: ``[tokens, engram layers, hash cols]`` rows of the layers' tables."""
        cfg, device = self.cfg, input_ids.device
        compressed = self.constant("token_map", device)[input_ids]
        pad = self.constant("token_map", device)[cfg.engram_pad_token_id]
        tail = self.slots.bind(self.engram_first).buffers()["tail"]
        ng = cfg.engram_max_ngram_size
        if self.layout is None:  # one token per row: every look-back is in the tail
            tokens = torch.cat([compressed[:, None], tail[slot]], dim=1)
            tokens = torch.where(positions[:, None] < self.constant("arange.ng", device), pad, tokens)
            tail[slot] = tokens[:, :ng - 1]  # the padding of a short sequence is never read back
        else:
            lay = self.layout
            assert isinstance(lay, Layout)
            earlier = tail[slot[:, None], lay.ngram.clamp_max(ng - 2)]
            tokens = torch.where(lay.ngram_from == 0, compressed[lay.ngram.clamp_max(compressed.shape[0] - 1)],
                                 torch.where(lay.ngram_from == 1, earlier, pad))
            if self._snapshot is not None and ng > 1:
                tail[self._snapshot.slots] = tokens[self._snapshot.ends - 1, :ng - 1]
            row_slot = slot[lay.last]
            kept = tail[row_slot[:, None], lay.tail.clamp_max(ng - 2)]
            tail[row_slot] = torch.where(lay.tail_from == 0, compressed[lay.tail.clamp_max(compressed.shape[0] - 1)],
                                         torch.where(lay.tail_from == 1, kept, pad))
        return self.k("ds.engram_hash@1")(tokens, self.constant("engram.multipliers", device),
                                          self.constant("engram.primes", device), self.constant("engram.offsets", device))

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.linear@1")(x, weight)

    def norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.rms_norm@1")(x, weight, self.cfg.rms_norm_eps)

    def hc_epilogue_norm(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor,
        last_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.attn_staged.get(self.table, False)
            and x.device.type == "cuda"
            and h.shape[1] == 4
            and x.shape[-1] >= 1024
        ):
            from kernels import ds1017_triton as fast_triton

            return fast_triton.hc_epilogue_norm(
                x, h, post, comb, pre, norm_weight, self.cfg.rms_norm_eps, last_index
            )
        if last_index is not None:
            h, pre = h[last_index], pre[last_index]
            x, post, comb = x[last_index], post[last_index], comb[last_index]
        return self.norm(self.hc_pre(self.hc_post(x, h, post, comb), pre), norm_weight)

    def hc_step(self, x: torch.Tensor | None, stream: torch.Tensor, pre: torch.Tensor, post: torch.Tensor | None,
                comb: torch.Tensor | None, w: Weights, which: str, *, stage_attn: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Between two sublayers (``ds.hc_step@1``): the output ``x`` of the one before written back to
        the stream, the next coefficients, and the input of the sublayer ``which`` (``attn`` or
        ``ffn``). Returns that input, the stream, and the new ``pre``, ``post`` and ``comb``."""
        cfg = self.cfg
        # A backend whose MoE or Attention reads its FP8 input staged by this step writes it there now.
        if which == "ffn":
            stage: Any = (self.group, self.placement.slots(cfg.n_routed_experts), cfg.num_experts_per_tok,
                          cfg.moe_intermediate_size)
        elif which == "attn" and stage_attn and self.attn_staged.get(self.table, False):
            stage = "attn"
        else:
            stage = None
        return self.k("ds.hc_step@1")(x, stream, pre, post, comb, w[f"hc_{which}_fn"], w[f"hc_{which}_scale"],
                                       w[f"hc_{which}_base"], w[f"{which}_norm.weight"], cfg.rms_norm_eps,
                                       cfg.hc_sinkhorn_iters, cfg.hc_eps, stage)

    def hc_pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_pre@1")(x, pre)

    def hc_post(self, y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_post@1")(y, residual, post, comb)

    def engram(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``Engram.forward`` on the stream ``[tokens, hc, dim]``: the layer's table rows at the hash ids,
        projected to one key per copy and a shared value, added through a gate."""
        cfg = self.cfg
        index = cfg.engram_layer_ids.index(layer)
        held = scales = ids = None
        if self.group.size == 1:
            ids = self.step["hashes"][:, index]
        elif not self.engram_staged.get(self.table, False):
            assert self.fetched is not None
            held, scales = self.fetched[0][:, index], self.fetched[1][:, index]
        out = self.k("ds.engram@1")(index, x, held, scales, ids, w["engram.embed.weight"], w["engram.embed.scale"],
                                     w["engram.wkv.weight"], w["engram.wkv.scale"], w["engram.wkv.packed"],
                                     w["engram.q_weight"], w["engram.k_weight"], cfg.rms_norm_eps, True)
        if layer == self._last_engram:
            self.step.pop("hashes", None)
            self.fetched = None
        return out

    def moe(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``MoE.forward``: the routed experts' weighted sum and the shared expert."""
        cfg = self.cfg
        # Among ranks this rank's stacks hold its part of the layer's experts, where the placement says.
        return self.k("ds.moe@1")(
            self.group, x, w["ffn.gate.weight"], w["ffn.gate.bias"], w["ffn.gate.bf16"],
            *(w[f"ffn.experts.{m}.{part}"] for m in ("w13", "w2") for part in ("weight", "packed")),
            *(w[f"ffn.shared_experts.{m}.{part}"] for m in ("w13", "w2") for part in ("weight", "scale", "packed")),
            cfg.num_experts_per_tok, cfg.routed_scaling_factor, cfg.swiglu_limit, *self.placement.maps(layer),
            self.decode_valid if self.layout is None and x.shape[0] else None)

    def idle(self, weights: Sequence[Weights]) -> None:
        """A step in which this rank has no rows, among ranks that share the experts and the Engram
        tables: its part in what the other ranks compute, layer by layer in the forward's order (their
        tokens may have chosen its experts, or hash to its rows)."""
        if self.group.size == 1:
            return
        self.idle_fetch(weights)
        self.idle_experts(weights)

    def idle_fetch(self, weights: Sequence[Weights]) -> None:
        self._sync_flags()
        self.table = "decode"
        if self.cfg.engram_layer_ids:
            device = weights[0]["ffn.gate.weight"].device
            ids = torch.empty(0, len(self.cfg.engram_layer_ids), self.cfg.engram_hash_cols,
                              dtype=torch.int64, device=device)
            self.k("ds.engram_fetch@1")(self.group, ids, *self.tables(), self.engram_staged.get("decode", False))

    def idle_experts(self, weights: Sequence[Weights]) -> None:
        x = weights[0]["ffn.gate.weight"].new_empty((0, self.cfg.hidden_size), dtype=torch.bfloat16)
        for layer in range(self.cfg.num_hidden_layers):
            self.moe(layer, x, weights[layer])

    # ------------------------------------------------------------------ attention

    def cos_sin(self, layer: int, device: torch.device) -> torch.Tensor:
        """The layer's rotary table as cosines and sines (``ds.rope@1``)."""
        return self.constant("cos_sin.compress" if self.cfg.ratio(layer) else "cos_sin.window", device)

    def attention(self, layer: int, x: torch.Tensor, w: Weights,
                  selected: torch.Tensor | slice | None = None) -> torch.Tensor:
        """``Attention.forward`` on a layer's normed input ``[tokens, dim]``."""
        cfg, step = self.cfg, self.step
        heads, dim, rd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.rms_norm_eps
        pos32 = step["pos32"]
        table = self.cos_sin(layer, pos32.device)
        has_iq = layer in cfg.index_source_layer_ids
        q, kv, iq = self.k("ds.attn_qkv@1")(
            x, w["attn.wq_a.weight"], w["attn.wq_a.scale"], w["attn.wq_a.packed"],
            w["attn.wkv.weight"], w["attn.wkv.scale"], w["attn.wkv.packed"], w["attn.q_norm.weight"], eps,
            w["attn.wq_b.weight"], w["attn.wq_b.scale"], w["attn.wq_b.packed"],
            w["attn.indexer.wq_b.weight"] if has_iq else None,
            w["attn.indexer.wq_b.scale"] if has_iq else None,
            w["attn.indexer.wq_b.packed"] if has_iq else None,
            heads, dim)
        ring = self.slots.bind(layer).buffers()["ring"]  # [slots, window, record]: a page a request
        win = cfg.sliding_window
        sink = w["attn.attn_sink"]
        store = self.k("ds.store_cache@1")
        ratio = cfg.ratio(layer)
        if self.layout is None:
            # A decode step attends in the caches: the token's own record goes into the ring first,
            # then its window is the ring's records at the positions up to its own, oldest first.
            store(kv, w["attn.kv_norm.weight"], eps, table, pos32, rd, ring.flatten(0, 1), step["ring_slot32"], "fp8", win)
            latents = chosen = chosen_cnt = None
            if ratio:
                latents, chosen = self._compressed(layer, x, iq, w)
                chosen_cnt = step.get(("chosen_cnt", ratio)) if self.mega_decode_attn.get(self.table, False) else None
            out = self.k("ds.cache_attn_out@1")(
                q, table, pos32, rd, ring, step["window"], latents, chosen, sink, dim ** -0.5, cfg.o_groups,
                w["attn.wo_a.weight"], w["attn.wo_b.weight"], w["attn.wo_b.scale"], w["attn.wo_b.packed"],
                step.get("window_cnt"), chosen_cnt)
            return out if selected is None else out[selected]
        # A step laid out on the host attends over one source of keys, as DeepInfer's prefill does:
        # the window's earlier keys loaded out of the rings, this step's own, the latents loaded.
        # The ring is read before this step's keys go into it (they may take the same entries).
        lay = self.layout
        tokens = kv.shape[0]
        keep_idx = step["keep_idx"]
        if step.get("paged_consumer", False):
            from kernels import ds1017_triton as fast_triton

            kv_src, idx_src = cfg.kv_source(layer), cfg.index_source(layer)
            pool = self.kv.bind(kv_src).buffers()
            entries = self.kv.page_size // ratio
            swa_fp8 = step["swa_fp8"]
            swa_flat = swa_fp8.flatten(0, 1)
            if lay.ring_total:
                swa_flat[:lay.ring_total] = ring[step["swa_ring_slots"]].reshape(lay.ring_total, -1)
            store(kv, w["attn.kv_norm.weight"], eps, table, pos32, rd, swa_flat, step["swa_self_slot32"], "fp8", win)
            if step["keep_dst_safe"].numel():
                ring.flatten(0, 1)[step["keep_dst_safe"]] = swa_flat[step["keep_swa_src"]]
            if layer == idx_src:
                prev_idx = self._prev_idx_src.get(layer)
                if prev_idx is not None:
                    step.pop(("chosen_phys", prev_idx), None)
                assert iq is not None
                target_k = min(cfg.index_topk, lay.latents[ratio].width)
                pad_k = max(128, -(-target_k // 128) * 128)
                if target_k == 0:
                    phys_buf = torch.full((tokens, pad_k), -1, dtype=torch.int32, device=kv.device)
                else:
                    phys_buf = torch.empty((tokens, pad_k), dtype=torch.int32, device=kv.device)
                    if pad_k > target_k:
                        phys_buf[:, target_k:].fill_(-1)
                    step["_prefill_row_local"] = True
                    if target_k >= 512 and target_k % 128 == 0:
                        step["_prefill_out_buf"] = phys_buf[:, :target_k]
                    try:
                        chosen = self._index(layer, x, iq, w, pool["ik"], entries)
                    finally:
                        step.pop("_prefill_row_local", None)
                        step.pop("_prefill_out_buf", None)
                    ch_i32 = chosen if chosen.dtype == torch.int32 else chosen.to(torch.int32)
                    if ch_i32.data_ptr() != phys_buf.data_ptr():
                        phys_buf[:, :ch_i32.shape[1]].copy_(ch_i32)
                        if ch_i32.shape[1] < pad_k:
                            phys_buf[:, ch_i32.shape[1]:].fill_(-1)
                    fast_triton.map_logical_to_req_tokens_inplace(
                        phys_buf, step["rows"], self.batch_tables["req_to_token"]
                    )
                step["chosen_phys", idx_src] = phys_buf
            chosen_phys = step["chosen_phys", idx_src]
            chosen_cnt = step["chosen_cnt_1"]
            swa_win_idx = step["swa_win_idx32"]
            swa_win_cnt = step["swa_win_cnt32"]
            query_pos = pos32
            if selected is not None:
                q, query_pos = q[selected], query_pos[selected]
                swa_win_idx, swa_win_cnt = swa_win_idx[selected], swa_win_cnt[selected]
                chosen_phys, chosen_cnt = chosen_phys[selected], chosen_cnt[selected]
            return self.kernels("decode", "ds.cache_attn_out@1")(
                q, table, query_pos, rd, swa_fp8, swa_win_idx, pool["kv"].unflatten(0, (-1, entries)),
                chosen_phys, sink, dim ** -0.5, cfg.o_groups,
                w["attn.wo_a.weight"], w["attn.wo_b.weight"], w["attn.wo_b.scale"], w["attn.wo_b.packed"],
                swa_win_cnt, chosen_cnt)
        if not ratio:
            src = step.get("source_swa")
            if src is None:
                src = step["source_swa"] = torch.empty(lay.ring_total + tokens, dim, dtype=kv.dtype, device=kv.device)
            if lay.ring_total:
                self.k("ds.load_values@1")(ring.flatten(0, 1), win, lay.ring_segments, lay.ring_total, "fp8",
                                           src[:lay.ring_total])
            kv_out = self.k("ds.transform@1")(kv, w["attn.kv_norm.weight"], eps, table, pos32, rd, "fp8",
                                              src[lay.ring_total:])
            store(kv_out[keep_idx], None, eps, None, None, 0, ring.flatten(0, 1), step["keep_ring_slot32"], "fp8", win)
            idx = step.get("win_idx32")
            if idx is None:
                win_cols = lay.window_index.shape[1]
                pad_w = -(-win_cols // 128) * 128 if self.mega_prefill_attn.get(self.table, False) else win_cols
                w32 = lay.window_index if lay.window_index.dtype == torch.int32 else lay.window_index.to(torch.int32)
                if pad_w == win_cols:
                    idx = step["win_idx32"] = w32.contiguous()
                else:
                    idx = step["win_idx32"] = torch.full((tokens, pad_w), -1, dtype=torch.int32, device=kv.device)
                    idx[:, :win_cols].copy_(w32)
        else:
            kv_src, idx_src = cfg.kv_source(layer), cfg.index_source(layer)
            pool = self.kv.bind(kv_src).buffers()
            entries = self.kv.page_size // ratio
            lat = lay.latents[ratio]
            if layer == kv_src:
                if not self._full_kv_stored:
                    self._compress(layer, x, w, pool, entries)
                step.pop("source_swa", None)
                step.pop("ik_loaded", None)
                shape = (lay.ring_total + tokens + lat.total, dim)
                src = step.get("shared_source")
                if src is None or src.shape != shape:
                    step.pop("shared_source", None)
                    del src
                    src = step["shared_source"] = torch.empty(shape, dtype=kv.dtype, device=kv.device)
                if lat.total:
                    self.k("ds.load_values@1")(pool["kv"], entries, lat.segments, lat.total, "latent",
                                               src[lay.ring_total + tokens:])
            else:
                src = step["shared_source"]
            if lay.ring_total:
                self.k("ds.load_values@1")(ring.flatten(0, 1), win, lay.ring_segments, lay.ring_total, "fp8",
                                           src[:lay.ring_total])
            kv_out = self.k("ds.transform@1")(kv, w["attn.kv_norm.weight"], eps, table, pos32, rd, "fp8",
                                              src[lay.ring_total:lay.ring_total + tokens])
            store(kv_out[keep_idx], None, eps, None, None, 0, ring.flatten(0, 1), step["keep_ring_slot32"], "fp8", win)
            if layer == idx_src:
                prev_idx = self._prev_idx_src.get(layer)
                if prev_idx is not None:
                    step.pop(("sparse_idx", prev_idx), None)
                assert iq is not None
                offset = lay.ring_total + tokens
                use_di_off = self.deepinfer_index.get(self.table, False)
                win_raw32 = step.get("win_raw32")
                if win_raw32 is None:
                    w32 = lay.window_index if lay.window_index.dtype == torch.int32 else lay.window_index.to(torch.int32)
                    win_raw32 = step["win_raw32"] = w32.contiguous()
                win_cols = win_raw32.shape[1]
                target_k = min(cfg.index_topk, lay.latents[ratio].width)
                total_k = win_cols + target_k
                pad_k = -(-total_k // 128) * 128 if self.mega_prefill_attn.get(self.table, False) else total_k
                sp_idx = torch.empty((tokens, pad_k), dtype=torch.int32, device=kv.device)
                sp_idx[:, :win_cols].copy_(win_raw32)
                if pad_k > total_k:
                    sp_idx[:, total_k:].fill_(-1)
                target_slice = sp_idx[:, win_cols:total_k]
                if use_di_off:
                    step["_prefill_idx_offset"] = offset
                    if target_k >= 512 and target_k % 128 == 0 and pad_k % 8 == 0 and win_cols % 8 == 0:
                        step["_prefill_out_buf"] = target_slice
                try:
                    chosen = self._index(layer, x, iq, w, pool["ik"], entries)
                finally:
                    if use_di_off:
                        step.pop("_prefill_idx_offset", None)
                        step.pop("_prefill_out_buf", None)
                ch_i32 = chosen if chosen.dtype == torch.int32 else chosen.to(torch.int32)
                if ch_i32.shape != target_slice.shape:
                    total_k = win_cols + ch_i32.shape[1]
                    pad_k = -(-total_k // 128) * 128 if self.mega_prefill_attn.get(self.table, False) else total_k
                    sp_idx = torch.empty((tokens, pad_k), dtype=torch.int32, device=kv.device)
                    sp_idx[:, :win_cols].copy_(win_raw32)
                    if pad_k > total_k:
                        sp_idx[:, total_k:].fill_(-1)
                    target_slice = sp_idx[:, win_cols:total_k]
                if use_di_off:
                    if ch_i32.data_ptr() != target_slice.data_ptr():
                        target_slice.copy_(ch_i32)
                else:
                    target_slice.copy_(torch.where(ch_i32 >= 0, ch_i32 + offset, -1))
                step["sparse_idx", idx_src] = sp_idx
            idx = step["sparse_idx", idx_src]
        query_pos = step["pos32"]
        if selected is not None:
            q, query_pos, idx = q[selected], query_pos[selected], idx[selected]
        out = self.k("ds.sparse_attn_out@1")(
            q, table, query_pos, rd, src, idx, sink, dim ** -0.5, cfg.o_groups,
            w["attn.wo_a.weight"], w["attn.wo_b.weight"], w["attn.wo_b.scale"], w["attn.wo_b.packed"])
        if not ratio and layer == self._last_swa_only:
            step.pop("source_swa", None)
            step.pop("win_idx32", None)
        return out

    def _compressed(self, layer: int, x: torch.Tensor, iq: torch.Tensor | None, w: Weights
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``Attention._compress_kv``. A source layer compresses first; an index source chooses, and the
        layers after it take its choice. In a decode step: the source layer's latent cache by its pages
        ``[pages, entries, record]`` and per token its chosen latents as rows of it; in a step laid out
        on the host: the latents of the step's rows loaded ``[latents, dim]`` and the chosen ones as
        indices into them. ``-1``: none."""
        cfg, step = self.cfg, self.step
        ratio, source = cfg.ratio(layer), cfg.kv_source(layer)
        pool = self.kv.bind(source).buffers()
        entries = self.kv.page_size // ratio
        if layer == source:
            self._compress(layer, x, w, pool, entries)
        if layer == cfg.index_source(layer):
            assert iq is not None
            step["chosen", ratio] = self._index(layer, x, iq, w, pool["ik"], entries)
            step["chosen_cnt", ratio] = step.pop("last_latent_cnt", None)
        chosen = step["chosen", ratio]
        if self.layout is None:
            return pool["kv"].unflatten(0, (-1, entries)), chosen
        lat = self.layout.latents[ratio]
        if not lat.total:
            return pool["kv"].new_zeros(0, cfg.head_dim, dtype=x.dtype), chosen
        return self.k("ds.load_values@1")(pool["kv"], entries, lat.segments, lat.total, "latent"), chosen

    def _compress(self, layer: int, x: torch.Tensor, w: Weights, pool: Weights, entries: int,
                  *, fuse_index: bool = True) -> None:
        """``Compressor.forward_store_kv_cache`` and the index keys of ``MQA.store_index_kv``: the
        latents of the groups this step completes, normed, rotated at their first position and
        quantized into ``kv``; the indexer's key of each (from the latent before rotation) into ``ik``.
        A latent's row is its first token's slot over the ratio."""
        cfg, step = self.cfg, self.step
        ratio, rd, eps = cfg.ratio(layer), cfg.qk_rope_head_dim, cfg.rms_norm_eps
        pos, rows = step["pos"], step["rows"]
        table = self.batch_tables["req_to_token"]
        linear = self.k("ds.linear@1")
        fuse_di = self.deepinfer_index.get(self.table, False)
        if ratio == 1:
            wkv = w["attn.compressor.wkv.weight"]
            w_idx = w.get("attn.indexer.weights_proj.weight") if fuse_index and fuse_di and layer == cfg.index_source(layer) else None
            if w_idx is not None and wkv.dtype == w_idx.dtype and (wkv.shape[0] + w_idx.shape[0]) % 16 == 0:
                c1_out = linear(x, self._merged_linear("c1_idx", wkv, w_idx))
                pre = c1_out[:, :wkv.shape[0]]
                step[("_pre_idx_weights", layer)] = c1_out[:, wkv.shape[0]:]
            else:
                pre = linear(x, wkv)
            first_c, dest = step["pos32"], step["tok_slot32"]
        else:
            state = self.slots.bind(layer).buffers()["cstate"]
            xf = x if self.layout is not None else x.float()
            wkv, wgate = w["attn.compressor.wkv.weight"], w["attn.compressor.wgate.weight"]
            if fuse_di and wkv.dtype == wgate.dtype and (wkv.shape[0] + wgate.shape[0]) % 8 == 0:
                kvg = linear(xf, self._merged_linear("kv_gate", wkv, wgate))
                kv, score = kvg[:, :wkv.shape[0]], kvg[:, wkv.shape[0]:]
            else:
                kv, score = linear(xf, wkv), linear(xf, wgate)
            if self.layout is None:
                pos1_32 = step.get("pos1_32")
                if pos1_32 is None:
                    pos1_32 = step["pos1_32"] = (step["pos32"] + 1).contiguous()
                pre = self.k("ds.compress_decode@1")(score, kv, state, step["slot32"], pos1_32, ratio)
                comp_dest = step.get(("comp_dest", ratio))
                if comp_dest is None:
                    fc = (pos + 1 - ratio).clamp_min(0)
                    dc = torch.where((pos + 1) % ratio == 0, table[rows, fc].long() // ratio, -1)
                    comp_dest = step["comp_dest", ratio] = (
                        fc.to(torch.int32).contiguous(), dc.to(torch.int32).contiguous())
                first_c, dest = comp_dest
            else:
                if self._snapshot is not None:
                    cs = self._snapshot.cstate.get(state.shape[1])
                    if cs is not None:
                        c_slots, c_cols, c_toks = cs
                        state[c_slots, c_cols, 0] = score[c_toks].to(state.dtype)
                        state[c_slots, c_cols, 1] = kv[c_toks].to(state.dtype)
                groups = self.layout.latents[ratio].groups
                assert groups is not None
                if not groups.c_seq.numel():
                    return
                pre = self.k("ds.compress@1")(score, kv, state, step["slot"][groups.seq_token], groups.cu_q,
                                              groups.cu_k, groups.cu_c, groups.c_seq, ratio)
                comp_dest = step.get(("comp_dest", ratio))
                if comp_dest is None:
                    fc = groups.first.clamp_min(0)
                    dc = torch.where(groups.real, table[rows[groups.entry_token], fc].long() // ratio, -1)
                    comp_dest = step["comp_dest", ratio] = (
                        fc.to(torch.int32).contiguous(), dc.to(torch.int32).contiguous())
                first_c, dest = comp_dest
        freqs = self.cos_sin(layer, pos.device)
        store = self.k("ds.store_cache@1")
        latent = store(pre, w["attn.compressor.norm.weight"], eps, freqs, first_c, rd, pool["kv"], dest,
                       "latent", entries, True)
        store(linear(latent, w["attn.indexer.wk.weight"]), w["attn.indexer.k_norm.weight"], eps, freqs, first_c,
              rd, pool["ik"], dest, "key", entries)

    def _index(self, layer: int, x: torch.Tensor, iq: torch.Tensor, w: Weights, keys: torch.Tensor, entries: int
               ) -> torch.Tensor:
        """``MQA.forward``: per token the ``index_topk`` best latents of its row it reaches, ascending by
        position (in a decode step as cache rows, else as indices into the step's loaded latents),
        ``-1`` after them. A candidate source also chooses the candidate blocks the layers after it
        score in."""
        cfg, step = self.cfg, self.step
        ratio, rd, h, d = cfg.ratio(layer), cfg.qk_rope_head_dim, cfg.index_n_heads, cfg.index_head_dim
        pos32 = step["pos32"]
        q = iq.unflatten(-1, (h, d))
        q_data, q_scales = self.k("ds.quant_index_q@1")(q, self.cos_sin(layer, pos32.device), pos32, rd)
        raw_w = step.pop(("_pre_idx_weights", layer), None)
        if raw_w is None:
            raw_w = self.k("ds.linear@1")(x, w["attn.indexer.weights_proj.weight"])
        weights = raw_w * (d ** -0.5 * h ** -0.5)
        source, uses = layer == cfg.candidate_source_layer_id, 0 <= cfg.candidate_source_layer_id < layer
        mode = "source" if source else "user" if uses else "dense"
        candidates = step.get("candidates") if uses else None
        blocks, block = cfg.candidate_topk_blocks, cfg.candidate_block_size
        if self.layout is None:
            # Every row's pages up to the longest context the batch takes: a kernel compiled for that
            # width serves every step (DeepInfer passes its model's longest length); the scores are
            # over the first ``width`` positions.
            width = self.extent[ratio]
            lens32 = step.get(("lens32", ratio))
            if lens32 is None:
                lens32 = step["lens32", ratio] = ((pos32 + 1) // ratio).contiguous()
            dec_cache = step if (self.deepinfer_index.get(self.table, False)
                                 and self.mega_decode_attn.get(self.table, False)) else None
            chosen, picked = self.k("ds.index_topk_paged@1")(q_data, q_scales, weights, keys, entries, step["pages32"],
                                                             lens32, width, min(cfg.index_topk, width),
                                                             mode, candidates, blocks, block, dec_cache)
        else:
            lat = self.layout.latents[ratio]
            ks_ke32 = step.get(("ks_ke32", ratio))
            if ks_ke32 is None:
                ks_ke32 = step["ks_ke32", ratio] = (
                    lat.ks.to(torch.int32).contiguous(), lat.ke.to(torch.int32).contiguous())
            ik_loaded = step.get("ik_loaded")
            if ik_loaded is None:
                ik_loaded = step["ik_loaded"] = self.k("ds.load_keys@1")(
                    keys, entries, lat.segments, lat.total, d)
            k_data, k_scales = ik_loaded
            pre_cache = step if self.deepinfer_index.get(self.table, False) else None
            chosen, picked = self.k("ds.index_topk@1")(q_data, q_scales, weights, k_data, k_scales, ks_ke32[0],
                                                       ks_ke32[1], self.max_context_len // ratio,
                                                       min(cfg.index_topk, lat.width), mode, candidates, blocks, block,
                                                       pre_cache)
        if source:
            step["candidates"] = picked
        return chosen


def token_map(path: str, cfg: DSConfig) -> list[int] | None:
    """The compressed token map Engram hashes over: ``engram_token_map.json`` of the model directory
    (a generated checkpoint has one), else built from its ``tokenizer.json`` as the reference builds it."""
    if not cfg.engram_layer_ids:
        return None
    ready = os.path.join(path, "engram_token_map.json") if path else ""
    if ready and os.path.isfile(ready):
        with open(ready) as f:
            lookup = json.load(f)
        size = max(lookup) + 1
    else:
        tokenizer = os.path.join(path, "tokenizer.json") if path else ""
        if not tokenizer or not os.path.isfile(tokenizer):
            return None
        from tokenizers import Tokenizer

        lookup, size = compressed_token_map(Tokenizer.from_file(tokenizer))
    if size != cfg.engram_compressed_vocab_size:
        raise ValueError(f"the compressed token map has {size} ids, the config says "
                         f"{cfg.engram_compressed_vocab_size}: every hash would land elsewhere")
    return lookup


def hf_config(path: str, inline: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    if not inline:
        raise ValueError("layers.ds1017 needs model.path or model.hf_config")
    return inline


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        cfg = DSConfig.from_hf(hf_config(config["model.path"], dict(config["model.hf_config"])))
        cfg.check()
        directory = config["dsv4.token_map_path"] or config["model.path"] or config["weights.path"]
        layers = Layers(
            ctx.use("kernels.catalog@1"), config, cfg, ctx.use("state.kv.paged@1"), ctx.use("state.swa.slots@1"),
            token_map(directory, cfg), config["batch.max_context_len"], ctx.use("comm.group@1"),
            ctx.use("expert.placement@1"))
        ctx.on_config("dsv4.yoco_paged_consumer", layers.set_yoco_paged_consumer)
        ctx.provide("layers.ds1017@1", layers)

