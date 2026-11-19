"""DS 1017 (DeepSeek-V4.1) through the worker units on the CPU: a tiny config with every kind of
layer, a random checkpoint in the real checkpoint's formats, the reference kernels. The engine
lays a step out in ways the reference implementation shipped with the checkpoint never does (a
prompt in chunks, rows at different positions, a bucket of padded lanes); what holds those together
here is that the reference kernels give a token the same result however its step is laid out, so
every run below must agree bit for bit. ``tests/gpu/test_ds1017.py`` holds the model itself to that
implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from core.contracts.columns import Column
from core.contracts.kernels import Implementation, Solution
from core.contracts.plan import GRAPH, PlanKey
from core.contracts.records import StateCopy
from deploy.models.hf import cut
from kernels import ds1017_formats as formats
from layers.ds1017.unit import Layers
from models.ds1017_config import DSConfig, engram_layout, states
from tests.units.support.ds1017 import DSWorker, checkpoint, driver, tiny_config
from tests.units.support.graphs import with_graphs

PROMPTS = {0: [5, 9, 17, 33, 2, 77, 100, 3, 8, 21, 60, 61, 62, 90, 91, 92, 93, 14, 15, 16, 101, 102, 103],
           1: [7, 7, 8, 120, 44, 45, 46, 47, 48, 49, 50, 51, 52],
           2: [3, 111, 12]}
STEPS = 5


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    cfg = tiny_config()
    return cfg, checkpoint(tmp_path_factory.mktemp("ds1017"), cfg)


@pytest.fixture
def worker(model: tuple[dict[str, Any], Path]) -> Iterator[DSWorker]:
    w = DSWorker(model[0], **{"weights.path": str(model[1])})
    try:
        yield w
    finally:
        w.close()


def generate(w: DSWorker, rows: list[int], chunk: int | None = None, prompts: dict[int, list[int]] | None = None
             ) -> tuple[dict[int, list[int]], dict[int, list[torch.Tensor]]]:
    d = driver(w)
    for row in rows:
        d.add(row, (prompts or PROMPTS)[row])
    logits = d.generate(rows, STEPS, chunk)
    return {row: list(d.requests[row].outputs) for row in rows}, logits


def same(a: dict[int, list[torch.Tensor]], b: dict[int, list[torch.Tensor]], rows: list[int]) -> None:
    for row in rows:
        assert len(a[row]) == len(b[row]) == STEPS
        for step, (x, y) in enumerate(zip(a[row], b[row], strict=True)):
            assert torch.equal(x, y), f"row {row}, step {step}: differs by {float((x - y).abs().max())}"


def fresh(model: tuple[dict[str, Any], Path], **config: Any) -> DSWorker:
    return DSWorker(model[0], **{"weights.path": str(model[1]), **config})


def test_the_tiny_model_has_every_kind_of_layer_and_generates(worker: DSWorker) -> None:
    cfg = DSConfig.from_hf(tiny_config())
    assert [cfg.ratio(i) for i in range(6)] == [0, 2, 2, 1, 1, 1]
    assert [cfg.kv_source(i) for i in range(6)] == [-1, 1, 1, 3, 3, 3]
    assert [cfg.index_source(i) for i in range(6)] == [-1, 1, 1, 3, 3, 5]
    tokens, logits = generate(worker, [0])
    assert len(tokens[0]) == STEPS and logits[0][0].dtype == torch.float32 and logits[0][0].shape == (128,)
    assert bool(torch.isfinite(torch.stack(logits[0])).all()) and len(set(tokens[0])) > 1
    # The prompt is longer than the window, has more latents than an indexer keeps, and more blocks
    # of latents than the candidate layer keeps: every selection is in play.
    assert len(PROMPTS[0]) > cfg.sliding_window and len(PROMPTS[0]) > cfg.index_topk
    assert len(PROMPTS[0]) > cfg.candidate_topk_blocks * cfg.candidate_block_size


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_a_prompt_run_in_chunks_generates_what_it_does_in_one_piece(model: tuple[dict[str, Any], Path],
                                                                   worker: DSWorker, chunk: int) -> None:
    """Chunks of 3 and 7 cut compression groups of 2 in two; chunks of 1 are steps of one token each."""
    whole = generate(worker, [0])
    w = fresh(model)
    try:
        pieces = generate(w, [0], chunk)
    finally:
        w.close()
    assert pieces[0] == whole[0]
    same(pieces[1], whole[1], [0])


def test_a_row_generates_the_same_alone_and_beside_other_rows(model: tuple[dict[str, Any], Path], worker: DSWorker) -> None:
    together = generate(worker, [0, 1, 2])  # prompts of other lengths: decode steps and prefill chunks mix
    for row in (0, 1, 2):
        w = fresh(model)
        try:
            alone = generate(w, [row])
        finally:
            w.close()
        assert alone[0][row] == together[0][row]
        same(alone[1], together[1], [row])


def test_a_decoded_token_is_computed_as_the_same_token_of_a_prompt(model: tuple[dict[str, Any], Path],
                                                                  worker: DSWorker) -> None:
    """The prompt and its first outputs given as one prompt end in the logits the last decode step
    gave: a decode step and a prefill lay the same token out the same way once it is past the
    window's width, and every state a decode step left is what a prefill leaves."""
    tokens, logits = generate(worker, [0])
    w = fresh(model)
    try:
        d = driver(w)
        d.add(0, PROMPTS[0] + tokens[0][:-1])
        forced = d.generate([0], 1)
    finally:
        w.close()
    assert torch.equal(forced[0][0], logits[0][-1])


def test_a_step_from_a_graph_bucket_generates_what_an_eager_step_does(model: tuple[dict[str, Any], Path],
                                                                     worker: DSWorker) -> None:
    """Through a stand-in recorder: a bucket's fixed rows, padded lanes, the indexer over the whole
    context, the experts run on every token."""
    eager = generate(worker, [0, 1, 2])
    w = fresh(model)
    try:
        replays = with_graphs(w, [1, 4])
        graphs = generate(w, [0, 1, 2])
    finally:
        w.close()
    assert graphs[0] == eager[0]
    same(graphs[1], eager[1], [0, 1, 2])
    assert replays and set(replays) == {4}  # three rows: every decode step replayed the bucket of 4


def test_the_checkpoint_formats_are_loaded_as_the_layers_read_them(model: tuple[dict[str, Any], Path],
                                                                  worker: DSWorker) -> None:
    saved: dict[str, torch.Tensor] = {}
    for file in sorted(model[1].glob("*.safetensors")):
        saved |= load_file(str(file))
    weights = worker.service("state.weights@1").bind("").buffers()
    # A layer's experts in two stacks of the checkpoint's bytes: each expert's w1 over its w3, and w2;
    # their scales as a GEMM kernel reads them, which unpacks to the checkpoint's.
    stack, both = weights["layers.2.ffn.experts.w2.weight"], weights["layers.2.ffn.experts.w13.weight"]
    assert stack.dtype == torch.uint8 and stack.shape == (4, 64, 32) and both.shape == (4, 128, 32)
    packed2, packed13 = weights["layers.2.ffn.experts.w2.packed"], weights["layers.2.ffn.experts.w13.packed"]
    assert packed2.shape == (4, 1, 64, 4) and packed13.shape == (4, 1, 128, 4)
    for e in range(4):
        def of(name: str) -> torch.Tensor:
            return saved[f"layers.2.ffn.experts.{e}.{name}"].view(torch.uint8)

        assert torch.equal(stack[e], of("w2.weight")) and torch.equal(both[e], torch.cat([of("w1.weight"), of("w3.weight")]))
        assert torch.equal(formats.unpack_scales(packed2[e], 64, 2), of("w2.scale"))
        assert torch.equal(formats.unpack_scales(packed13[e], 128, 2), torch.cat([of("w1.scale"), of("w3.scale")]))
    # FP8 weights as bytes with their scales, and the scales by row, packed; wo_a dequantized; the head
    # and the pooling compressor in fp32.
    assert torch.equal(weights["layers.0.attn.wq_b.weight"], saved["layers.0.attn.wq_b.weight"].view(torch.uint8))
    scale, packed = weights["layers.0.attn.wq_b.scale"], weights["layers.0.attn.wq_b.packed"]
    n = weights["layers.0.attn.wq_b.weight"].shape[0]
    assert torch.equal(formats.unpack_scales(packed, n, scale.shape[1]), scale.repeat_interleave(32, 0)[:n])
    words = formats.packed_words(packed, n)
    assert words.dtype == torch.int32 and words.shape == (n, packed.shape[0]) and words.stride(0) == 1
    # The shared expert's w1 over its w3, with their block scales and those scales packed.
    w13 = weights["layers.2.ffn.shared_experts.w13.weight"]
    one = saved["layers.2.ffn.shared_experts.w1.weight"].view(torch.uint8)
    three = saved["layers.2.ffn.shared_experts.w3.weight"].view(torch.uint8)
    assert torch.equal(w13, torch.cat([one, three]))
    s1 = saved["layers.2.ffn.shared_experts.w1.scale"].view(torch.uint8)
    s3 = saved["layers.2.ffn.shared_experts.w3.scale"].view(torch.uint8)
    assert torch.equal(weights["layers.2.ffn.shared_experts.w13.scale"], torch.cat([s1, s3]))
    both = torch.cat([formats.block_rows(s1, one.shape[0]), formats.block_rows(s3, three.shape[0])])
    assert torch.equal(formats.unpack_scales(weights["layers.2.ffn.shared_experts.w13.packed"], w13.shape[0], both.shape[1]),
                       both)
    wo_a = formats.dequant_fp8(saved["layers.3.attn.wo_a.weight"].view(torch.uint8), saved["layers.3.attn.wo_a.scale"].view(torch.uint8))
    assert torch.equal(weights["layers.3.attn.wo_a.weight"], wo_a.to(torch.bfloat16)) and "layers.3.attn.wo_a.scale" not in weights
    assert weights["head.weight"].dtype == torch.float32 and torch.equal(weights["head.weight"], saved["head.weight"].float())
    assert weights["layers.1.attn.compressor.wkv.weight"].dtype == torch.float32      # two tokens to one: pooled in fp32
    assert weights["layers.3.attn.compressor.wkv.weight"].dtype == torch.bfloat16     # one to one: a plain projection
    assert "layers.3.attn.compressor.wgate.weight" not in weights and "layers.5.attn.indexer.wk.weight" not in weights


def test_a_checkpoint_missing_an_expert_is_refused(model: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    from safetensors.torch import save_file

    for file in sorted(model[1].glob("*.safetensors")):
        tensors = {k: v for k, v in load_file(str(file)).items() if not k.startswith("layers.4.ffn.experts.3.w1.")}
        save_file(tensors, str(tmp_path / file.name))
    for name in ("config.json", "engram_token_map.json"):
        (tmp_path / name).write_text((model[1] / name).read_text())
    with pytest.raises(Exception, match="3 of the 4 experts of layers.4.ffn.experts.w1"):
        fresh((model[0], tmp_path))


def test_engram_without_its_token_map_fails_the_start(model: tuple[dict[str, Any], Path], tmp_path: Path,
                                                      worker: DSWorker) -> None:
    """And the map may be in a directory of its own (``dsv4.token_map_path``): a model served from its
    config alone (cut to its first layers) with dummy or separate weights, a tokenizer kept elsewhere."""
    bare, maps = tmp_path / "bare", tmp_path / "maps"
    bare.mkdir()
    maps.mkdir()
    for file in model[1].glob("*"):
        (maps if file.name == "engram_token_map.json" else bare).joinpath(file.name).write_bytes(file.read_bytes())
    with pytest.raises(Exception, match="compressed token map"):
        fresh((model[0], bare))
    w = fresh((model[0], bare), **{"dsv4.token_map_path": str(maps)})
    try:
        assert generate(w, [2])[0] == generate(worker, [2])[0]
    finally:
        w.close()


def test_dummy_weights_are_refused(model: tuple[dict[str, Any], Path]) -> None:
    """The weights are the checkpoint's bytes: ``initialize_dummy_weights`` fills floating-point tensors
    only, which would leave every quantized matrix and its scales zero."""
    with pytest.raises(Exception, match="weights.path is empty, and DS 1017 takes no dummy weights"):
        DSWorker(model[0], **{"dsv4.token_map_path": str(model[1])})


def test_what_the_layers_store(worker: DSWorker) -> None:
    cfg = DSConfig.from_hf(tiny_config())
    decls = states(cfg, 4)
    paged = {d.layer: d for d in decls if d.kind == "kv.paged"}
    slots = {d.layer: d for d in decls if d.kind == "swa.slots"}
    assert sorted(paged) == [1, 3] and sorted(slots) == [0, 1, 2, 3, 4, 5]
    # Bytes in DeepInfer's cache formats: an FP4 latent of 32 is 16 bytes and 2 scales, an FP4 key 16 and 1,
    # an FP8 window KV 32 and 1. A page of 4 tokens holds 2 latents at a ratio of 2, 4 at a ratio of 1.
    assert paged[1].tensors == {"kv": ((2, 18), "u8"), "ik": ((2, 17), "u8")} and paged[1].tokens_per_item == 4
    assert paged[3].tensors == {"kv": ((4, 18), "u8"), "ik": ((4, 17), "u8")}
    assert slots[0].tensors == {"ring": ((8, 33), "u8")}
    assert set(slots[1].tensors) == {"ring", "cstate", "tail"} and slots[1].tensors["tail"] == ((3,), "i64")
    assert slots[1].tensors["cstate"] == ((2, 2, 32), "f32")  # each member's score, then its KV
    assert set(slots[3].tensors) == {"ring"}  # one latent per token: no group to fill
    model = worker.service("model@1")
    assert set(model.stores) == {"kv.paged", "swa.slots"} and list(model.backends) == ["layers.ds1017@1"]
    assert worker.service("executor@1").capacity().items == {"kv.paged": 64, "swa.slots": 6}
    kv = worker.service("state.kv.paged@1")
    assert sorted(kv.tensors) == [1, 3] and kv.k_buffer == {} and kv.bind(3).buffers()["ik"].shape == (65 * 4, 17)
    assert kv.bind(1).buffers()["kv"].shape == (65 * 2, 18)


def test_a_config_the_layers_cannot_run_is_refused() -> None:
    def text(**kw: Any) -> DSConfig:
        return DSConfig.from_hf(tiny_config(**kw))

    with pytest.raises(ValueError, match="no KV source before it"):
        text(kv_source_layer_ids=[3]).check()
    with pytest.raises(ValueError, match="no indexer between"):
        text(index_source_layer_ids=[1]).check()
    with pytest.raises(ValueError, match="multiples of 32"):
        text(q_lora_rank=48).check()
    with pytest.raises(ValueError, match="sqrtsoftplus"):
        text(scoring_func="sigmoid")
    with pytest.raises(ValueError, match="candidate_source_layer_id = 4: that layer runs no indexer"):
        text(candidate_source_layer_id=4).check()
    with pytest.raises(ValueError, match="candidate_source_layer_id = 1: the indexer of layer 3 scores latents of another ratio"):
        text(candidate_source_layer_id=1).check()


def test_a_model_cut_to_its_first_layers_reads_only_their_tensors(model: tuple[dict[str, Any], Path]) -> None:
    """The real checkpoint is hundreds of GiB: a config cut to a few layers must not read the rest."""
    from core.state.weights.unit import safetensors_files, safetensors_weights_iterator

    w = DSWorker(cut(model[0], 3), **{"weights.path": str(model[1])})
    try:
        wants = w.service("model@1").wants_weight
        assert wants("layers.2.ffn.experts.3.w1.weight") and wants("embed.weight") and wants("layers.0.attn.wo_a.scale")
        assert not wants("layers.3.attn.wq_a.weight") and not wants("layers.5.ffn.experts.0.w2.scale")
        assert not wants("mtp.0.attn.wq_a.weight") and not wants("vision.blocks.0.mlp.w1.weight")
        assert not wants("layers.1.ffn.gate.bias_vl") and wants("model.layers.1.self_attn.wkv.weight")
        read = [name for name, _ in safetensors_weights_iterator(safetensors_files(str(model[1])), wants)]
        assert read and all(wants(name) for name in read) and not any(name.startswith(("layers.3.", "layers.5.")) for name in read)
        every = [name for name, _ in safetensors_weights_iterator(safetensors_files(str(model[1])))]
        assert len(read) < len(every)
        tokens, _ = generate(w, [2])
        assert len(tokens[2]) == STEPS
    finally:
        w.close()


def test_a_config_cut_to_its_first_layers_keeps_their_engram_layers_only(model: tuple[dict[str, Any], Path]) -> None:
    """``--layers``: the Engram layers past the cut are not the model's (no state, no hash), and those
    kept hash as they do in the whole model, whose layout draws each layer's primes in the list's order."""
    whole, three = DSConfig.from_hf(model[0]), DSConfig.from_hf(cut(model[0], 3))
    assert whole.engram_layer_ids == (1, 4) and three.engram_layer_ids == (1,)
    assert three.engram_num_embeddings == whole.engram_num_embeddings[:1]
    assert [part[:1] for part in engram_layout(whole)] == list(engram_layout(three))
    three.check()
    for bad in ([4, 1], [1, 1], [-1, 1]):
        with pytest.raises(ValueError, match="engram_layer_ids .* ascend"):
            DSConfig.from_hf(tiny_config(engram_layer_ids=bad))
        with pytest.raises(ValueError, match="engram_layer_ids .* ascend"):
            DSConfig.from_hf(cut(tiny_config(engram_layer_ids=bad), 3))
    # One layer: no Engram layer is left, and no compressed KV either.
    one = cut(model[0], 1)
    assert DSConfig.from_hf(one).engram_layer_ids == () and not any(d.kind == "kv.paged" for d in states(DSConfig.from_hf(one), 4))
    w = DSWorker(one, **{"weights.path": str(model[1])})
    try:
        tokens, _ = generate(w, [0, 2])
        assert len(tokens[0]) == len(tokens[2]) == STEPS
    finally:
        w.close()


@pytest.mark.parametrize("backend", ["nope", "deepinfer", "tilelang"])
def test_a_named_backend_that_cannot_run_here_fails_the_start(model: tuple[dict[str, Any], Path], backend: str) -> None:
    """stage2-contract.md §5.5.3. Neither kernel library of this model runs on a CPU, and nothing
    registers under a misspelt name: the start fails rather than run the references in their place."""
    with pytest.raises(Exception, match=rf"dsv4.backend = '{backend}': backend '{backend}' has no solution"):
        fresh(model, **{"dsv4.backend": backend})


def test_a_captured_step_runs_the_named_backend_s_kernels_or_is_not_recorded(worker: DSWorker) -> None:
    """A step replayed from a graph takes no other backend's kernel in the named one's place, whatever
    else a graph could record. The references bring the routed experts' fixed-shape form under
    their own name (in the MoE's composition); a backend whose kernels a graph cannot record has no
    solution there (``""``:
    ``core.plan`` records no graph and says which kernels are why)."""
    catalog, layers = worker.service("kernels.catalog@1"), worker.service("layers.ds1017@1")
    declared = worker.service("model@1").assembly.states
    resolved = worker.service("plan@1").resolved
    names = sorted({name for phase in ("prefill", "decode") for name in resolved[phase]["layers.ds1017@1"]} - {"backend"})
    for name in names:  # two more backends: one a graph records, first by id, and one it does not
        for backend, capturable in (("another", True), ("eager", False)):
            catalog.register(Implementation(Solution(f"{backend}.{name}", name, "tests:kernel", capturable=capturable),
                                            catalog.resolve(f"reference.{name}")))
    graph = layers.resolve(declared)[GRAPH]
    assert graph.pop("backend") == "reference" and graph["ds.moe@1"] == "reference.fixed.ds.moe@1"
    assert all(solution.startswith("reference.") for solution in graph.values()), graph
    config = {"dsv4.backend": "eager", "dsv4.prefill_backend": "", "dsv4.decode_backend": "",
              "dsv4.yoco_mode": "off", "dsv4.yoco_keep_tokens": 140}
    eager = Layers(catalog, config, layers.cfg, layers.kv, layers.slots, [0], 64, layers.group,
                   layers.placement).resolve(declared)
    assert all(solution == f"eager.{name}" for phase in ("prefill", "decode")
               for name, solution in eager[phase].items() if name != "backend"), eager
    assert {solution for name, solution in eager[GRAPH].items() if name != "backend"} == {""}, eager[GRAPH]


def test_a_decode_step_reads_no_page_past_a_row_s_own(model: tuple[dict[str, Any], Path], worker: DSWorker) -> None:
    """A row's table keeps what a longer request left after the row's own pages (as SGLang's pool
    does), which may be pages of a pool since made smaller. The indexer of a decode step is given
    every row's table as wide as the longest row (a graph bucket: as the longest context) and reads
    only the latents each row reaches."""
    clean = generate(worker, [0, 2])
    for buckets in ([], [4]):
        w = fresh(model)
        try:
            if buckets:
                with_graphs(w, buckets)
            # Before any request has its pages: whatever a row is not given stays outside the pool.
            table = w.service("worker.batch@1").binding.buffers()["req_to_token"]
            table[1:] = 1000 * w.service("state.kv.paged@1").page_size
            stale = generate(w, [0, 2])
        finally:
            w.close()
        assert stale[0] == clean[0]
        same(stale[1], clean[1], [0, 2])


def test_speculation_is_refused_at_start_for_a_model_that_drafts_nothing(model: tuple[dict[str, Any], Path]) -> None:
    with pytest.raises(Exception, match="spec.algorithm = 'eagle': the model drafts no tokens"):
        fresh(model, **{"spec.algorithm": "eagle"})

def test_siblings_restored_from_the_same_window_generate_their_cold_logits(worker: DSWorker) -> None:
    prefix = PROMPTS[0][:12]  # three pages, after the first sliding-window wrap
    prompts = {1: prefix + PROMPTS[1], 2: prefix + PROMPTS[2]}
    cold_ids, cold_logits = generate(worker, [1, 2], prompts=prompts)

    d = driver(worker)
    parent = d.add(0, prefix + [55, 54, 53, 52])
    snapshot = d.next_slot
    d.next_slot += 1
    d.step({0: len(prefix)}, [StateCopy("swa.slots", "after", Column.of("i32", [parent.slot]),
                                        Column.of("i32", [snapshot]))])
    d.step({0: 4})  # overwrite the parent's ring and Engram tail after taking the snapshot
    for row, prompt in prompts.items():
        child = d.add(row, prompt)
        child.pages, child.computed = parent.pages[:len(prefix) // d.page_size], len(prefix)
        restore = StateCopy("swa.slots", "before", Column.of("i32", [snapshot]), Column.of("i32", [child.slot]))
        first = d.step({row: len(prompt) - len(prefix)}, [restore]).logits[0].clone()
        logits = [first, *d.generate([row], STEPS)[row]]
        assert child.outputs == cold_ids[row]
        same({row: logits}, cold_logits, [row])


def test_forward_backbone_and_lm_head_match_forward(model: tuple[dict[str, Any], Path]) -> None:
    w = fresh(model, **{"sampler.force_topk": 8, "dsv4.decode_overlap": True})
    try:
        m = w.service("model@1")
        assert not m.vp_head_ready
        assert not m._vp_head_cfg
        d = driver(w)
        d.add(0, PROMPTS[0][:10])
        delta, plan, ops = d.records({0: 10})
        batch = w.service("worker.batch@1")
        batch.apply(delta, ops)
        layers = w.service("layers.ds1017@1")
        input_ids = torch.tensor(PROMPTS[0][:10], dtype=torch.int64)
        positions = torch.arange(10, dtype=torch.int64)
        last_index = torch.tensor([9], dtype=torch.int64)
        layers.prepare(batch, plan, PlanKey("prefill", 0))
        ref = m.forward(input_ids, positions, last_index)
        layers.prepare(batch, plan, PlanKey("prefill", 0))
        hidden = m.forward_backbone(input_ids, positions, last_index)
        split = m.lm_head(hidden)
        torch.testing.assert_close(split, ref, rtol=0.0, atol=0.0)
    finally:
        w.close()
