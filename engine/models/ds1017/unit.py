# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""``models.ds1017``: DeepSeek-V4.1 text (DS 1017), ``model@1``. After ``Transformer`` and ``Block`` of
the reference implementation shipped with the checkpoint (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``), without vision and
without the draft layers (doc/specs/v1-ds1017.md).

What moved: the parameters are declared to ``state.weights@1`` under the checkpoint's names, in the
checkpoint's bytes where it is quantized (FP8 weights and their ue8m0 scales, FP4 experts, the Engram
tables: ``u8``); ``load_weights`` does what that implementation's ``convert.py`` does for one rank
(``wo_a`` dequantized to bf16, the head and the pooling compressors' projections in fp32). Beside
that, as DeepInfer lays weights out for its GEMM kernels: a layer's routed experts are two stacks,
each expert's ``w1`` over its ``w3`` (one product gives a SwiGLU's two inputs) and ``w2``, and the
shared expert's ``w1`` over its ``w3`` likewise; the
experts' scales are kept as a GEMM kernel reads them (``formats.pack_scales``), and so are each FP8
weight's, beside the checkpoint's block scales. They are declared tensors filled by ``load_weights``,
so a ``refit`` writes them again. The layers are ``layers.ds1017``, which the
executor prepares for the step; the residual stream is ``hc_mult`` copies ``[tokens, hc, dim]``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from core.contracts.model import Assembly, WeightDecl
from kernels import ds1017_formats as formats
from models.ds1017_config import KV_KIND, SLOT_KIND, DSConfig, states

_DECL_BYTES = MappingProxyType({"u8": 1, "i8": 1, "bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "f64": 8})

LAYER = re.compile(r"layers\.(\d+)\.")
EXPERT = re.compile(r"(layers\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.(weight|scale)")
SHARED = re.compile(r"(layers\.\d+\.ffn\.shared_experts)\.(w[13])\.(weight|scale)")
SKIPPED = ("mtp.", "vision.", "aligner.", "image_")
"""Checkpoint tensors the text model without draft layers does not read."""


def fp8(name: str, out: int, into: int) -> list[WeightDecl]:
    """An FP8 weight ``[out, into]`` and its 32 x 32 block scales, both as the checkpoint's bytes, and
    the scales as a GEMM kernel reads them."""
    return [WeightDecl(name + ".weight", (out, into), "u8"),
            WeightDecl(name + ".scale", (-(-out // 32), -(-into // 32)), "u8"),
            WeightDecl(name + ".packed", formats.packed_shape(out, into), "u8")]


def params(cfg: DSConfig, ranks: int = 1, slots: int = 0) -> list[WeightDecl]:
    """The backbone's parameters, by the checkpoint's names; the routed experts of a layer as
    ``ffn.experts.w13.weight`` ``[experts, 2 * inter, dim / 2]`` (``w1`` over ``w3``) and
    ``ffn.experts.w2.weight``, each with its ``packed`` scales.

    One of ``ranks`` ranks that hold the model between them (``comm.group@1``) has its part of what
    is split in order: of each layer's slots for routed experts (``slots / ranks``; ``slots`` is the
    experts, and spare slots for copies with ``expert.placement@1``), and of each Engram table's rows
    (``shard`` of them, the last rank's part padded)."""
    dim, hd, heads, inter, experts = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads,
                                      cfg.moe_intermediate_size, cfg.n_routed_experts)
    held = (slots or experts) // ranks
    mix = (2 + cfg.hc_mult) * cfg.hc_mult

    def w(name: str, *shape: int, fmt: str = "bf16") -> WeightDecl:
        return WeightDecl(name, shape, fmt)

    out = [w("embed.weight", cfg.vocab_size, dim), w("head.weight", cfg.vocab_size, dim, fmt="f32"), w("norm.weight", dim),
           # 0: the expert stacks' rows are in the checkpoint's order; 1: in the fused MoE kernel's.
           w("experts.layout", 1, fmt="u8"),
           # The expert each slot of each layer holds, over all ranks: as loaded, the checkpoint's
           # order (``expert.placement@1``'s initial layout); EPLB writes it when it moves the rows.
           w("experts.placed", cfg.num_hidden_layers, slots or experts, fmt="i32")]
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        out += [*fp8(p + "attn.wq_a", cfg.q_lora_rank, dim), *fp8(p + "attn.wq_b", heads * hd, cfg.q_lora_rank),
                *fp8(p + "attn.wkv", hd, dim),
                w(p + "attn.wo_a.weight", cfg.o_groups * cfg.o_lora_rank, heads * hd // cfg.o_groups),
                *fp8(p + "attn.wo_b", dim, cfg.o_groups * cfg.o_lora_rank), w(p + "attn.attn_sink", heads, fmt="f32"),
                w(p + "attn.q_norm.weight", cfg.q_lora_rank), w(p + "attn.kv_norm.weight", hd)]
        if i in cfg.kv_source_layer_ids:
            pooled = cfg.ratio(i) > 1  # the softmax pooling runs in fp32, and so do its projections
            out += [w(p + "attn.compressor.wkv.weight", hd, dim, fmt="f32" if pooled else "bf16"),
                    w(p + "attn.compressor.norm.weight", hd), w(p + "attn.indexer.wk.weight", cfg.index_head_dim, hd),
                    w(p + "attn.indexer.k_norm.weight", cfg.index_head_dim)]
            if pooled:
                out.append(w(p + "attn.compressor.wgate.weight", hd, dim, fmt="f32"))
        if i in cfg.index_source_layer_ids:
            out += [*fp8(p + "attn.indexer.wq_b", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
                    w(p + "attn.indexer.weights_proj.weight", cfg.index_n_heads, dim)]
        out += [w(p + "attn_norm.weight", dim), w(p + "ffn_norm.weight", dim),
                w(p + "ffn.gate.weight", experts, dim, fmt="f32"), w(p + "ffn.gate.bias", experts, fmt="f32"),
                # The gate's weight as the checkpoint stores it: a fused gate kernel's product is bf16.
                w(p + "ffn.gate.bf16", experts, dim)]
        for m, (n, k) in (("w13", (2 * inter, dim)), ("w2", (dim, inter))):
            out += [w(p + f"ffn.experts.{m}.weight", held, n, k // 2, fmt="u8"),
                    w(p + f"ffn.experts.{m}.packed", held, *formats.packed_shape(n, k), fmt="u8")]
        # The shared expert's w1 over its w3, as DeepInfer's shared_gate_up_proj (one GEMM).
        out += [*fp8(p + "ffn.shared_experts.w13", 2 * inter, dim), *fp8(p + "ffn.shared_experts.w2", dim, inter)]
        for which in ("attn", "ffn"):
            out += [w(p + f"hc_{which}_fn", mix, cfg.hc_mult * dim, fmt="f32"), w(p + f"hc_{which}_base", mix, fmt="f32"),
                    w(p + f"hc_{which}_scale", 3, fmt="f32")]
        if i in cfg.engram_layer_ids:
            rows, eh = shard(cfg.engram_num_embeddings[cfg.engram_layer_ids.index(i)], ranks), cfg.engram_head_dim
            out += [w(p + "engram.embed.weight", rows, eh, fmt="u8"), w(p + "engram.embed.scale", rows, eh // 32, fmt="u8"),
                    *fp8(p + "engram.wkv", dim * (cfg.hc_mult + 1), cfg.engram_hash_cols * eh),
                    w(p + "engram.q_weight", cfg.hc_mult, dim), w(p + "engram.k_weight", cfg.hc_mult, dim)]
    return out


def shard(rows: int, ranks: int) -> int:
    """Rows of a table of ``rows`` each of ``ranks`` ranks holds: rank ``r`` those from ``r * shard`` on."""
    return -(-rows // ranks)


ENGRAM = re.compile(r"layers\.(\d+)\.engram\.embed\.(weight|scale)")


def checkpoint_name(name: str) -> str:
    """A checkpoint tensor's name as ``convert.py`` reads it (older exports name things as HF does)."""
    name = name.removeprefix("model.").replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    return name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 1:
        raise ValueError(f"a quantized tensor has one byte per element, not {t.dtype}")
    return t.contiguous().view(torch.uint8)


class DeepseekV41ForCausalLM:
    """``model@1``."""

    def __init__(self, cfg: DSConfig, dtype: str, page_size: int, services: Mapping[str, Any],
                 *, symmetric_engram: bool = False, graph_idle: bool = False,
                 max_rows: int = 32, vp_head_enabled: bool = True) -> None:
        if dtype != "bf16":
            raise ValueError(f"model.dtype = {dtype!r}: DS 1017's activations are bf16 (its kernels quantize from it)")
        self.cfg, self._dtype = cfg, dtype
        self.symmetric_engram = symmetric_engram
        self.graph_idle = graph_idle
        self.max_rows = max_rows
        self._vp_head_cfg = vp_head_enabled
        self._vp_head: Any = None
        self._vp_head_failed = False
        s = services
        self._comm_group = s["comm.group@1"]
        self.ranks, self.rank = self._comm_group.size, self._comm_group.rank
        """The ranks that hold the model between them, and which of them this is."""
        if cfg.n_routed_experts % self.ranks:
            raise ValueError(f"comm.size = {self.ranks} does not divide the {cfg.n_routed_experts} routed experts")
        self.placement = s["expert.placement@1"]
        slots = self.placement.slots(cfg.n_routed_experts)
        self.held = slots // self.ranks
        """Slots for routed experts this rank has in each layer."""
        # Slot -> the expert the checkpoint's order puts there; this rank's slots of each expert.
        order = [*range(cfg.n_routed_experts), *(i % cfg.n_routed_experts for i in range(slots - cfg.n_routed_experts))]
        self.order = order
        self.slots_of: dict[int, list[int]] = {}
        for slot in range(self.rank * self.held, (self.rank + 1) * self.held):
            self.slots_of.setdefault(order[slot], []).append(slot - self.rank * self.held)
        self.weights, self.holder = s["state.weights@1"], s["state.holder@1"]
        self.kv, self.slots = s["state.kv.paged@1"], s["state.swa.slots@1"]
        self.embedding, self.layers = s["layers.embedding@1"], s["layers.ds1017@1"]
        self._assembly = Assembly(["layers.ds1017@1"] * cfg.num_hidden_layers, states(cfg, page_size),
                                  params(cfg, self.ranks, slots))
        self._views: list[Any] = []
        self._cached: tuple[tuple[int, int], list[Mapping[str, torch.Tensor]]] = ((-1, -1), [])

    @property
    def assembly(self) -> Assembly:
        return self._assembly

    @property
    def stores(self) -> Mapping[str, Any]:
        return {KV_KIND: self.kv, SLOT_KIND: self.slots}

    @property
    def backends(self) -> Mapping[str, Any]:
        return {"layers.ds1017@1": self.layers}

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    @property
    def dtype(self) -> str:
        return self._dtype

    def _allocate_engram(self, decl: WeightDecl) -> torch.Tensor | None:
        if ENGRAM.fullmatch(decl.name) is None:
            return None
        # Checkpoint loading and refit write the same tensors the peers will map.
        try:
            import torch.distributed._symmetric_memory as symm_mem
            with torch.inference_mode(False):
                return symm_mem.empty(decl.shape, dtype=torch.uint8, device=torch.device("cuda", torch.cuda.current_device())).zero_()
        except torch.OutOfMemoryError:
            raise
        except (ImportError, NotImplementedError, RuntimeError):
            return None  # Ordinary weight storage remains usable by the DeepEP path.

    def start(self) -> None:
        a = self._assembly
        weights_bytes = sum(math.prod(d.shape) * _DECL_BYTES[d.format] for d in a.weights)
        held_bytes = sum(t.numel() * t.element_size() for t in self.weights.bind("").buffers().values())
        kv_decls = [s for s in a.states if s.kind == KV_KIND]
        slot_decls = [s for s in a.states if s.kind == SLOT_KIND]
        if weights_bytes > held_bytes > 0:
            # When weights grow in place (e.g. hot-applying eplb.extra_experts), shrink the state pools
            # first so weights + states never transiently exceed holder.mem_fraction.
            self.holder.size(a.states, weights_bytes)
            self.kv.declare(kv_decls)
            self.slots.declare(slot_decls)
            self.weights.declare(a.weights, allocate=self._allocate_engram if self.symmetric_engram else None)
        else:
            self.weights.declare(a.weights, allocate=self._allocate_engram if self.symmetric_engram else None)
            self.holder.size(a.states, weights_bytes)
            self.kv.declare(kv_decls)
            self.slots.declare(slot_decls)
        self.weights.load()
        self._views = [self.weights.bind(f"layers.{i}.") for i in range(self.cfg.num_hidden_layers)]
        self._views.append(self.weights.bind(""))
        self.layers.set_tables(self._engram_tables)

    def close(self) -> None:
        """Drop cached views of the weight store and hooks into long-lived services on unload."""
        if self._vp_head is not None:
            self._vp_head.close()
            self._vp_head = None
        self._cached = ((-1, -1), [])
        self._views.clear()
        self.placement.stacks = None
        self.layers.set_tables(lambda: ([], []))

    def _engram_tables(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """This rank's rows of the Engram layers' tables, weights and scales, in the layers' order."""
        tables = [self._layer_weights()[i] for i in self.cfg.engram_layer_ids]
        return [t["engram.embed.weight"] for t in tables], [t["engram.embed.scale"] for t in tables]

    # -------------------------------------------------------------- weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], params_dict: Mapping[str, torch.Tensor]
                     ) -> set[str]:
        """``convert.py`` for one rank, into the declared tensors: quantized tensors as their bytes, a
        layer's experts into their stacks, ``wo_a`` dequantized, everything else cast. Of what is split
        over the ranks, this rank's part: its experts, and its rows of the Engram tables (given whole,
        or as the rows ``weight_rows`` names)."""
        loaded: set[str] = set()
        stacked: dict[str, set[tuple[int, str, str]]] = {}
        halves: dict[str, torch.Tensor] = {}
        inter = self.cfg.moe_intermediate_size
        for raw, tensor in weights:
            name = checkpoint_name(raw)
            if name.startswith(SKIPPED) or name.endswith("bias_vl"):
                continue
            expert = EXPERT.fullmatch(name)
            if expert:
                # w1 is the upper half of the ``w13`` stack, w3 the lower; a scale goes in packed.
                matrix, part = expert[3], expert[4]
                stack = f"{expert[1]}.{'w2' if matrix == 'w2' else 'w13'}"
                target = f"{stack}.{'weight' if part == 'weight' else 'packed'}"
                if target in params_dict:
                    rows = slice(inter, 2 * inter) if matrix == "w3" else slice(0, tensor.shape[0])
                    data = as_bytes(tensor)
                    for index in self.slots_of.get(int(expert[2]), ()):  # every slot of this rank that starts with it
                        if part == "weight":
                            params_dict[target][index, rows].copy_(data)
                        else:
                            params_dict[target][index, :, rows].copy_(formats.pack_scales(data)[:, :data.shape[0]])
                        stacked.setdefault(stack, set()).add((index, matrix, part))
                continue
            shared = SHARED.fullmatch(name)
            if shared:  # w1 is the upper half of w13, w3 the lower; a scale's rows and its packed rows too
                target = f"{shared[1]}.w13"
                if target + ".weight" in params_dict:
                    lower = shared[2] == "w3"
                    data = as_bytes(tensor)
                    if shared[3] == "weight":
                        params_dict[target + ".weight"][inter * lower:inter * (lower + 1)].copy_(data)
                    else:
                        blocks = inter // formats.BLOCK
                        params_dict[target + ".scale"][blocks * lower:blocks * (lower + 1)].copy_(data)
                        params_dict[target + ".packed"][:, inter * lower:inter * (lower + 1)].copy_(
                            formats.pack_scales(formats.block_rows(data, inter)))
                    halves[f"{target}.{shared[2]}.{shared[3]}"] = data[:0]
                    if all(f"{target}.{m}.{part}" in halves for m in ("w1", "w3") for part in ("weight", "scale")):
                        loaded.update((target + ".weight", target + ".scale", target + ".packed"))
                continue
            if name.endswith(("attn.wo_a.weight", "attn.wo_a.scale")):
                target = name.rsplit(".", 1)[0] + ".weight"
                if target not in params_dict:
                    continue
                if tensor.element_size() != 1:  # already dequantized
                    params_dict[target].copy_(tensor.to(params_dict[target].dtype))
                    loaded.add(target)
                    continue
                halves[name] = as_bytes(tensor)
                weight, scale = halves.get(target), halves.get(target[:-len("weight")] + "scale")
                if weight is not None and scale is not None:
                    params_dict[target].copy_(formats.dequant_fp8(weight, scale).to(params_dict[target].dtype))
                    loaded.add(target)
                    del halves[target], halves[target[:-len("weight")] + "scale"]
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            rows = self.weight_rows(raw)
            if rows is not None:  # this rank's rows of a table: the last rank's part may be short
                if tensor.shape[0] != rows[1] - rows[0]:
                    tensor = tensor[rows[0]:rows[1]]
                data = as_bytes(tensor)
                param[:data.shape[0]].copy_(data)
                param[data.shape[0]:].zero_()
                loaded.add(name)
                continue
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(f"{raw} is {tuple(tensor.shape)} in the checkpoint, {tuple(param.shape)} in the config")
            param.copy_(as_bytes(tensor) if param.dtype == torch.uint8 else tensor.to(param.dtype))
            loaded.add(name)
            if name.endswith("ffn.gate.weight") and name[:-len("weight")] + "bf16" in params_dict:
                params_dict[name[:-len("weight")] + "bf16"].copy_(tensor.to(torch.bfloat16))
                loaded.add(name[:-len("weight")] + "bf16")
            packed = params_dict.get(name[:-len("scale")] + "packed") if name.endswith(".scale") else None
            if packed is not None:
                packed.copy_(formats.pack_scales(formats.block_rows(param, packed.shape[1])))
                loaded.add(name[:-len("scale")] + "packed")
        if "experts.layout" in params_dict:  # as read: the checkpoint's row order
            params_dict["experts.layout"].zero_()
            loaded.add("experts.layout")
        if "experts.placed" in params_dict:  # as read: every slot holds the expert the order puts there
            params_dict["experts.placed"].copy_(torch.tensor(self.order, dtype=torch.int32).expand(
                self.cfg.num_hidden_layers, -1))
            loaded.add("experts.placed")
        for stack, seen in stacked.items():
            experts = params_dict[stack + ".weight"].shape[0]
            for matrix in (("w2",) if stack.endswith("w2") else ("w1", "w3")):
                count = len({index for index, m, part in seen if m == matrix and part == "weight"}
                            & {index for index, m, part in seen if m == matrix and part == "scale"})
                if count != experts:
                    raise ValueError(f"the checkpoint has {count} of the {experts} experts of "
                                     f"{stack.rsplit('.', 1)[0]}.{matrix}")
            loaded.update((stack + ".weight", stack + ".packed"))
        return loaded

    def wants_weight(self, raw: str) -> bool:
        """Whether ``load_weights`` reads the checkpoint tensor ``raw``: the weight store does not read
        the others (the layers a cut config leaves out, the draft layers, vision)."""
        name = checkpoint_name(raw)
        if name.startswith(SKIPPED) or name.endswith("bias_vl"):
            return False
        layer = LAYER.match(name)
        if layer is not None and int(layer[1]) >= self.cfg.num_hidden_layers:
            return False
        expert = EXPERT.fullmatch(name)
        return expert is None or int(expert[2]) in self.slots_of

    def weight_rows(self, raw: str) -> tuple[int, int] | None:
        """The rows ``load_weights`` takes of the checkpoint tensor ``raw``, ``None`` for all of it:
        of an Engram table split over the ranks, this rank's. The weight store reads those only."""
        table = ENGRAM.fullmatch(checkpoint_name(raw))
        if table is None or self.ranks == 1 or int(table[1]) not in self.cfg.engram_layer_ids:
            return None
        total = self.cfg.engram_num_embeddings[self.cfg.engram_layer_ids.index(int(table[1]))]
        part = shard(total, self.ranks)
        return min(self.rank * part, total), min((self.rank + 1) * part, total)

    def _layer_weights(self) -> list[Mapping[str, torch.Tensor]]:
        # Tensors made again (generation) or filled again from the checkpoint (fills, a refit too):
        # what is derived from their contents below is derived again.
        key = (self.weights.bind("").generation, self.weights.fills)
        if self._cached[0] != key:
            views = [v.buffers() for v in self._views]
            self._cached = (key, views)
            # The expert stacks' rows in the order the routed experts' kernel reads them: a row
            # permutation in place, undone when another backend's kernel reads them again.
            marker, mega = views[-1]["experts.layout"], self.layers.mega_experts()
            if bool(int(marker[0])) != mega:
                for layer in views[:-1]:
                    formats.expert_layout(*(layer[f"ffn.experts.{m}.{part}"] for m in ("w13", "w2")
                                            for part in ("weight", "packed")), mega=mega)
                    formats.shared_layout(layer["ffn.shared_experts.w13.weight"], layer["ffn.shared_experts.w13.packed"],
                                          layer["ffn.shared_experts.w2.packed"], mega=mega)
                marker.fill_(int(mega))
            # Where the experts are, as kept with the weights: the checkpoint's order where they were
            # loaded, where EPLB moved them otherwise (a reload of this unit keeps the weights).
            self.placement.start(self.cfg.num_hidden_layers, self.cfg.n_routed_experts,
                                 lambda layer: [views[layer][f"ffn.experts.{m}.{part}"] for m in ("w13", "w2")
                                                for part in ("weight", "packed")],
                                 views[0]["ffn.gate.weight"].device, views[-1]["experts.placed"])
        return self._cached[1]

    # -------------------------------------------------------------- forward

    @property
    def vp_head_ready(self) -> bool:
        return self._vp_head is not None

    @property
    def vp_head_cap(self) -> int:
        return int(self._vp_head.cap_per_rank) if self._vp_head is not None else 0

    def prepare_vp_head(self) -> bool:
        """Initialize NVLink5 vocab-parallel greedy LM head in lockstep across all ranks."""
        if self._vp_head is not None:
            return True
        if (not self._vp_head_cfg or self._vp_head_failed or not (1 < self.ranks <= 8)
                or self._comm_group.device_group is None or not self.symmetric_engram
                or self.max_rows < 1 or self.cfg.hidden_size % 8 != 0
                or self.cfg.vocab_size % self.ranks != 0
                or (self.cfg.vocab_size // self.ranks) % 16 != 0):
            return False
        if not self.layers.mega_experts():
            return False
        weights = self._layer_weights()
        head_w = weights[self.cfg.num_hidden_layers]["head.weight"]
        if not head_w.is_cuda:
            return False
        import torch.distributed as dist

        pg = self._comm_group.device_group
        cap = min(16, self.max_rows)
        ok = 1
        head: Any = None
        try:
            from kernels.ds1017_nvlink_head import NvlinkVocabHead

            head = NvlinkVocabHead(self._comm_group, self.cfg.vocab_size, self.cfg.hidden_size,
                                   cap, head_w.device)
            if not head.supports_weight(head_w):
                ok = 0
        except Exception as exc:  # noqa: BLE001
            ok = 0
            if self.rank == 0:
                print(f"models.ds1017: NVLink5 vocab-parallel LM head disabled: {exc!r}",
                      file=sys.stderr, flush=True)
        agree = torch.tensor([ok], dtype=torch.int32, device=head_w.device)
        dist.all_reduce(agree, op=dist.ReduceOp.MIN, group=pg)
        if int(agree.item()) != 1 or head is None:
            if head is not None:
                head.close()
            self._vp_head_failed = True
            return False
        self._vp_head = head
        if self.rank == 0:
            print(f"models.ds1017: NVLink5 vocab-parallel greedy LM head active "
                  f"(cap={cap}, shard={head.vocab_shard})", file=sys.stderr, flush=True)
        return True

    def vp_head_step(
        self,
        hidden: torch.Tensor | None,
        *,
        out_host_uva: torch.Tensor | None = None,
        last_token: torch.Tensor | None = None,
        rows_plus1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self._vp_head is not None
        top = self._layer_weights()[self.cfg.num_hidden_layers]
        return self._vp_head.step(hidden, top["head.weight"], out_host_uva=out_host_uva,
                                  last_token=last_token, rows_plus1=rows_plus1)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        if not hidden.shape[0]:
            return torch.empty(0, self.cfg.vocab_size, dtype=torch.float32, device=hidden.device)
        top = self._layer_weights()[self.cfg.num_hidden_layers]
        return self.layers.linear(hidden, top["head.weight"])

    def idle(self) -> None:
        """A step in which this rank has no rows while other ranks of its group have: its part in
        their forward (``Layers.idle``). The executor calls it for a plan without rows."""
        self.layers.idle(self._layer_weights())
        self.placement.step()
        if self._vp_head is not None:
            self.vp_head_step(None)

    def prepare_idle(self) -> None:
        self.layers.idle_fetch(self._layer_weights())

    def capture_idle(self, recorder: Any) -> Any:
        if not self.graph_idle or self.ranks == 1 or not self.layers.mega_experts():
            return None
        weights = self._layer_weights()
        with recorder.warmup(), torch.inference_mode():
            self.layers.idle_fetch(weights)
            self.layers.idle_experts(weights)
        with torch.inference_mode(), recorder.capture() as graph:
            self.layers.idle_experts(weights)
        try:
            with recorder.warmup():
                graph.replay()
        except BaseException:
            graph.close()
            raise
        return graph

    def stepped(self) -> None:
        """A forward of this rank's rows was run (or replayed from a graph): the executor calls it
        after each, outside any graph. The experts' placement counts the step, and places them again
        when it is time."""
        self.placement.step()

    def forward_backbone(self, input_ids: torch.Tensor, positions: torch.Tensor,
                         last_index: torch.Tensor) -> torch.Tensor:
        """All layers through the final RMSNorm: ``[rows, hidden_size]`` bf16 at each row's last token."""
        cfg, layers = self.cfg, self.layers
        weights = self._layer_weights()
        top = weights[cfg.num_hidden_layers]
        layers.begin(input_ids, positions)
        h = self.embedding.embed(input_ids, top["embed.weight"]).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        # Each sublayer's mixes give the next sublayer its input: the first takes the stream's first copy.
        pre = layers.hc_pre0(h.shape[0], h.device)
        # A sublayer's output ``x`` waits to be written back to the stream ``h`` with the ``post`` and
        # ``comb`` computed before it ran: the step before the next sublayer does it (DeepInfer's
        # order, one kernel there), or it is done here where the stream itself is needed.
        x: torch.Tensor | None = None
        post = comb = None
        selected_tail = False
        for i in range(cfg.num_hidden_layers):
            w = weights[i]
            if i in cfg.engram_layer_ids:
                if x is not None:
                    h, x = layers.hc_post(x, h, post, comb), None
                h = layers.engram(i, h, w)
            at_yoco = i == cfg.n_prefill_layer and layers.layout is not None
            stage_attn = not (at_yoco and layers.compacts_yoco_boundary(h.shape[0]))
            y, h, attn_pre, post, comb = layers.hc_step(x, h, pre, post, comb, w, "attn", stage_attn=stage_attn)
            if at_yoco:
                y, h, attn_pre, post, comb, last_index = layers.consumer_inputs(y, h, attn_pre, post, comb,
                                                                              last_index, w)
            next_eng = (i + 1, weights[i + 1]) if (i + 1) in cfg.engram_layer_ids else None
            if i == cfg.num_hidden_layers - 1 and last_index.numel() < input_ids.numel():
                # The last layer still writes all KV. Compute attention outputs for
                # requested positions, retaining one unused lane for the EP collective.
                selected = last_index if last_index.numel() else slice(0, 1)
                x = layers.attention(i, y, w, selected)
                h, attn_pre = h[selected], attn_pre[selected]
                post, comb = post[selected], comb[selected]
                selected_tail = True
            else:
                x = layers.attention(i, y, w, next_engram=next_eng)
            y, h, pre, post, comb = layers.hc_step(x, h, attn_pre, post, comb, w, "ffn")
            if next_eng is not None:
                layers.wait_engram_proj(y.device)
            x = layers.moe(i, y, w)
        layers.end()
        if selected_tail and not last_index.numel():
            return torch.empty(0, cfg.hidden_size, dtype=torch.bfloat16, device=h.device)
        idx = last_index if not selected_tail and (last_index.numel() < h.shape[0] or layers.layout is not None) else None
        if x is not None and post is not None and comb is not None:
            return layers.hc_epilogue_norm(x, h, post, comb, pre, top["norm.weight"], idx)
        if idx is not None:
            h, pre = h[idx], pre[idx]
        return layers.norm(layers.hc_pre(h, pre), top["norm.weight"])

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        """``Transformer.forward``: logits ``[rows, vocab]`` in fp32 at each row's last token."""
        out = self.forward_backbone(input_ids, positions, last_index)
        return self.lm_head(out)


def hf_config(path: str, inline: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    if not inline:
        raise ValueError("models.ds1017 needs model.path or model.hf_config")
    return inline


class Unit:
    SERVICES = ("state.weights@1", "state.holder@1", "state.kv.paged@1", "state.swa.slots@1", "layers.embedding@1",
                "layers.ds1017@1", "comm.group@1", "expert.placement@1")

    def __init__(self, ctx: Any, config: Any) -> None:
        if not config["weights.path"]:
            raise ValueError("weights.path is empty, and DS 1017 takes no dummy weights: they fill floating-point "
                             "tensors only, and its matrices are stored as FP8 and FP4 bytes with their scales "
                             "(agxinfer checkpoint writes random weights in the checkpoint's formats)")
        cfg = DSConfig.from_hf(hf_config(config["model.path"], dict(config["model.hf_config"])))
        cfg.check()
        symmetric = (ctx.use("comm.group@1").device_group is not None and config["dsv4.backend"] in ("auto", "deepinfer")
                     and config["deepinfer.nvlink_engram"])
        self.model = DeepseekV41ForCausalLM(cfg, config["model.dtype"], config["state.kv.page_size"],
                                            {name: ctx.use(name) for name in self.SERVICES}, symmetric_engram=symmetric,
                                            graph_idle=config["dsv4.graph_idle_experts"],
                                            max_rows=int(config["batch.max_rows"]),
                                            vp_head_enabled=bool(config["dsv4.vp_head"]) and int(config["sampler.force_topk"]) == 0)
        ctx.effect(lambda: self.model.close, label="model caches and layer hooks")
        ctx.register("state.weights@1", self.model.load_weights, label="load_weights")
        self.model.start()
        ctx.provide("model@1", self.model)
