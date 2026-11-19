"""``kernels.deepinfer``: the kernel libraries DeepInfer runs DeepSeek-V4.1 on (DeepGEMM, FlashMLA,
TileKernels, DeepSelect), as solutions of the DS definitions (``kernels.ds1017``): the backend
``deepinfer``. They need a GPU of SM 10.0 and the four libraries in the engine's Python environment;
without either the unit registers nothing. Which library call solves which definition, and where its
arithmetic differs from the references, is in ``ops``.

What the layers call between and for the sublayers' own work, ``ds.hc_step@1`` and ``ds.moe@1``, are
DeepGEMM's fused kernels (``mega``): one kernel for a sublayer's step, and for the MoE the fused gate
and the fused MoE kernel over a buffer the ranks of a group share (``comm.group@1``; on one rank too).
The Engram rows of ranks that share the tables still travel through the group's collectives
(``kernels.ds1017_ep``), before the forward."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from core.contracts.kernels import CAPTURE, Implementation, Solution

BACKEND = "deepinfer"
LIBRARIES = ("deep_gemm", "flash_mla", "tile_kernels", "deep_select")

SOLUTIONS: tuple[tuple[str, str, dict[str, list[int]]], ...] = (
    ("ds.fake_quant_fp8@1", "tile_kernels", {}), ("ds.fake_quant_fp4@1", "tile_kernels", {}),
    ("ds.linear_fp8@1", "deep_gemm", {}), ("ds.linear@1", "deep_gemm", {}), ("ds.grouped_linear@1", "deep_gemm", {}),
    ("ds.rms_norm@1", "tile_kernels", {}),
    # FlashMLA's kernels are built for these.
    ("ds.sparse_attn@1", "flash_mla", {"heads": [64, 128], "head_dim": [512]}),
    ("ds.cache_attn@1", "flash_mla", {"heads": [64, 128], "head_dim": [512]}),
    # DeepInfer's own TileLang operators (tilelang_ops), and DeepGEMM's indexer kernels with DeepSelect.
    ("ds.rope@1", "deepinfer", {}), ("ds.transform@1", "deepinfer", {}), ("ds.store_cache@1", "deepinfer", {}),
    ("ds.load_values@1", "deepinfer", {}), ("ds.load_keys@1", "deepinfer", {}),
    ("ds.quant_index_q@1", "deepinfer", {}),
    ("ds.index_topk@1", "deep_gemm", {"index_head_dim": [64, 128]}),
    ("ds.index_topk_paged@1", "deep_gemm", {"index_head_dim": [64, 128], "index_block": [32, 64, 128]}),
    ("ds.compress@1", "deepinfer", {}), ("ds.compress_decode@1", "deepinfer", {}),
    ("ds.shared_expert@1", "tile_kernels", {}), ("ds.engram_hash@1", "tile_kernels", {}),
    ("ds.hc_mixes@1", "tile_kernels", {}), ("ds.hc_pre@1", "tile_kernels", {}), ("ds.hc_post@1", "tile_kernels", {}),
    ("ds.moe_gate@1", "tile_kernels", {}), ("ds.moe_experts@1", "deep_gemm", {}), ("ds.top@1", "deep_select", {}),
    ("ds.engram_gate@1", "tile_kernels", {}),
    ("ds.silu@1", "torch", {}), ("ds.pool@1", "torch", {}), ("ds.index_score@1", "torch", {}),
)
"""(definition, the library its kernel is from, the shapes it is built for); the function in ``ops``
has the definition's name. A CUDA graph may record every one of them (their launches are on torch's
current stream; tests/gpu/test_ds1017_deepinfer.py replays each) but ``HOST_DEPENDENT``."""

HOST_DEPENDENT = frozenset({"ds.moe_experts@1", "ds.index_topk@1", "ds.compress@1"})
"""Which read the device back: the routed experts size their rows to the step's tokens (a captured
step runs ``FIXED`` instead); the indexer and the compressor of a prefill (no graph records one)."""
FIXED = "deepinfer.fixed.ds.moe_experts@1"


def library(name: str) -> str:
    try:
        return f"{name}=={version(name)}"
    except PackageNotFoundError:
        return name


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if (not devices or "sm_100" not in devices[0].capabilities
                or any(importlib.util.find_spec(name) is None for name in LIBRARIES)):
            return
        from kernels.deepinfer import ops

        ops.align_experts()
        for definition, source, shapes in SOLUTIONS:
            if definition == "ds.linear@1":
                continue
            solution = f"{BACKEND}.{definition}"
            name = definition.split(".", 1)[1].split("@")[0]
            implementation = getattr(ops, name)
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{name}", library=library(source), shapes=shapes,
                         capturable=definition not in HOST_DEPENDENT, requires=["cuda", "sm_100"]), implementation),
                label=f"solution {solution}")
        fn = ops.moe_experts_fixed
        ctx.register("kernels.catalog@1", Implementation(
            Solution(FIXED, "ds.moe_experts@1", f"{ops.__name__}:{fn.__name__}", library=library("deep_gemm"),
                     capturable=True, requires=["cuda", "sm_100", CAPTURE]), fn), label=f"solution {FIXED}")
        # DeepInfer's fused kernels, with this instance's part of the ranks' shared MoE buffer: a
        # sublayer's step and the MoE, where the model's shapes are what the kernels are built for.
        # Elsewhere, and for comparison, the same from the unfused kernels above ("unfused": after
        # the fused ones in the catalog's order, so a model the fused kernels fit runs those).
        from kernels.deepinfer.mega import Mega

        mega = Mega(
            config["sched.token_budget"] + config["batch.max_rows"],
            config["deepinfer.force_random_gate"],
            nvlink_engram=bool(config["deepinfer.nvlink_engram"]),
        )
        ctx.effect(lambda: mega.close, label="fused MoE and Engram buffers")
        fused: dict[str, list[int]] = {"mega": [1]}
        fused_attn: dict[str, list[int]] = {"heads": [64, 128], "head_dim": [512], "mega": [1], "mega_attn": [1]}
        fused_engram: dict[str, list[int]] = {"mega": [1], "mega_engram": [1]}
        has_deep_ep = importlib.util.find_spec("deep_ep") is not None
        has_triton = importlib.util.find_spec("triton") is not None
        entries: list[tuple[str, str, Any, bool, list[str], dict[str, list[int]], str]] = [
            (f"{BACKEND}.ds.hc_step@1", "ds.hc_step@1", mega.hc_step, True, [], fused, "deep_gemm"),
            (f"{BACKEND}.ds.moe@1", "ds.moe@1", mega.moe, True, [], fused, "deep_gemm"),
            (f"{BACKEND}.ds.attn_qkv@1", "ds.attn_qkv@1", mega.attn_qkv, True, [], fused_attn, "deep_gemm"),
            (f"{BACKEND}.ds.cache_attn_out@1", "ds.cache_attn_out@1", mega.cache_attn_out, True, [], fused_attn,
             "flash_mla"),
            (f"{BACKEND}.ds.sparse_attn_out@1", "ds.sparse_attn_out@1", mega.sparse_attn_out, True, [], fused_attn,
             "flash_mla"),
            (f"{BACKEND}.ds.engram@1", "ds.engram@1", mega.engram, True, [], fused_engram, "deep_gemm"),
            (f"{BACKEND}.unfused.ds.hc_step@1", "ds.hc_step@1", ops.hc_step, True, [], {}, "deep_gemm"),
            (f"{BACKEND}.unfused.ds.moe@1", "ds.moe@1", ops.moe, False, [], {}, "deep_gemm"),
            (f"{BACKEND}.unfused.fixed.ds.moe@1", "ds.moe@1", ops.moe_fixed, True, [CAPTURE], {}, "deep_gemm"),
            (f"{BACKEND}.unfused.ds.attn_qkv@1", "ds.attn_qkv@1", ops.attn_qkv, True, [], {}, "deep_gemm"),
            (f"{BACKEND}.unfused.ds.cache_attn_out@1", "ds.cache_attn_out@1", ops.cache_attn_out, True, [],
             {"heads": [64, 128], "head_dim": [512]}, "flash_mla"),
            (f"{BACKEND}.unfused.ds.sparse_attn_out@1", "ds.sparse_attn_out@1", ops.sparse_attn_out, True, [],
             {"heads": [64, 128], "head_dim": [512]}, "flash_mla"),
            (f"{BACKEND}.unfused.ds.engram@1", "ds.engram@1", ops.engram, True, [], {}, "tile_kernels"),
            (f"{BACKEND}.ds.ep_experts@1", "ds.ep_experts@1", ops.ep_experts, False, [], {}, "deep_gemm"),
        ]
        entries.append((f"{BACKEND}.ds.linear@1", "ds.linear@1", mega.linear, True, [], {}, "deep_gemm"))
        if has_deep_ep or (has_triton and bool(config["deepinfer.nvlink_engram"])):
            entries.append((f"{BACKEND}.ds.engram_fetch@1", "ds.engram_fetch@1", mega.engram_fetch, True, [],
                            fused_engram, "deep_ep" if has_deep_ep else "triton"))
        entries.append((f"{BACKEND}.unfused.ds.engram_fetch@1", "ds.engram_fetch@1", ops.engram_fetch, False, [],
                        {}, "deep_gemm"))
        for solution, definition, fn, capturable, requires, shapes, pkg in entries:
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{fn.__module__}:{fn.__qualname__}", library=library(pkg),
                         shapes=shapes, capturable=capturable, requires=["cuda", "sm_100", *requires]), fn),
                label=f"solution {solution}")

        from kernels.deepinfer import sampling
        ctx.register("kernels.catalog@1", Implementation(
            Solution("deepinfer.sample.seeded@1", "sample.seeded@1", "kernels.deepinfer.sampling:CachedSampler.sample",
                     library=library("triton"), shapes={}, capturable=False,
                     requires=["cuda", "sm_100"]), sampling.CachedSampler().sample), label="fused stateless sampling")

        ctx.register("kernels.catalog@1", Implementation(
            Solution("deepinfer.sample.topk@1", "sample.topk@1", "kernels.deepinfer.sampling:force_topk",
                     library=library("deep_select"), capturable=True, requires=["cuda", "sm_100"]),
            sampling.force_topk), label="sampler vocabulary cap")
