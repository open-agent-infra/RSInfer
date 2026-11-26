"""Snapshot isolation, whole-state validation, bounded lookup, and invalidation."""

from __future__ import annotations

import hashlib
from typing import Any

import msgspec
import pytest
import torch

from core.contracts.cache import CacheRef
from core.contracts.state import StateDecl
from state.cache.tensors import Image, dump, restore
from state.cache.unit import Cache


class Binding:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def buffers(self) -> dict[str, torch.Tensor]:
        return self.tensors


class Pool:
    def __init__(self, tensors: dict[int, dict[str, torch.Tensor]]) -> None:
        self.tensors = tensors

    def bind(self, layer: int) -> Binding:
        return Binding(self.tensors[layer])

    def capacity(self) -> int:
        return 4


def pools() -> tuple[list[StateDecl], dict[str, Pool]]:
    decls = [StateDecl("kv.paged", 0, {"k": ((4, 2), "bf16"), "v": ((4, 2), "bf16")}, "request", 4),
             StateDecl("gdn.slots", 1, {"conv": ((3, 2), "f32"), "temporal": ((2, 3), "f32")}, "slot")]
    generator = torch.Generator().manual_seed(41)
    kv = {name: torch.randn(20, 2, generator=generator).bfloat16() for name in ("k", "v")}
    gdn = {name: torch.randn(5, *shape, generator=generator) for name, (shape, _) in decls[1].tensors.items()}
    return decls, {"kv.paged": Pool({0: kv}), "gdn.slots": Pool({1: gdn})}


@pytest.mark.covers(edges=("data", "state"), kinds=("state",))
def test_all_state_restores_to_private_ids_and_partial_page_tail_is_zero() -> None:
    decls, stores = pools()
    ref = CacheRef("test", 5)
    raw = dump(ref, decls, stores, {"kv.paged": [0, 2], "gdn.slots": [0]}, 1 << 20)
    before = {kind: {name: t.clone() for name, t in pool.bind(0 if kind == "kv.paged" else 1).buffers().items()}
              for kind, pool in stores.items()}
    restore(raw, ref, decls, stores, {"kv.paged": [1, 3], "gdn.slots": [2]}, 1 << 20)
    for name, tensor in stores["kv.paged"].bind(0).buffers().items():
        assert torch.equal(tensor[8:12], before["kv.paged"][name][4:8])
        assert torch.equal(tensor[16], before["kv.paged"][name][12])
        assert torch.count_nonzero(tensor[17:20]) == 0
        assert torch.equal(tensor[:8], before["kv.paged"][name][:8])
    for name, tensor in stores["gdn.slots"].bind(1).buffers().items():
        assert torch.equal(tensor[3], before["gdn.slots"][name][1])
        assert torch.equal(tensor[1], before["gdn.slots"][name][1])


@pytest.mark.covers(edges=("data", "state"), kinds=("state",))
@pytest.mark.parametrize("damage", ["checksum", "layout", "missing", "key"])
def test_bad_snapshot_is_rejected_before_any_destination_is_written(damage: str) -> None:
    decls, stores = pools()
    ref = CacheRef("test", 4)
    ids = {"kv.paged": [0], "gdn.slots": [0]}
    raw = dump(ref, decls, stores, ids, 1 << 20)
    image = msgspec.msgpack.decode(raw, type=Image)
    if damage == "checksum":
        image.digest = "broken"
    elif damage == "key":
        image.ref = CacheRef("another-model", 4)
    else:
        if damage == "layout":
            image.tensors[-1].dtype = "torch.float64"
        else:
            image.tensors.pop()
        image.digest = hashlib.sha256(msgspec.msgpack.encode(image.tensors)).hexdigest()
    before = dump(ref, decls, stores, ids, 1 << 20)
    with pytest.raises(ValueError):
        restore(msgspec.msgpack.encode(image), ref, decls, stores, ids, 1 << 20)
    assert dump(ref, decls, stores, ids, 1 << 20) == before
    with pytest.raises(ValueError, match="exceeds"):
        dump(ref, decls, stores, ids, 1)


class MemoryStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.queries: list[int] = []

    def call(self, op: str, value: Any) -> Any:
        if op == "exists":
            self.queries.append(len(value))
            return [int(key in self.values) for key in value]
        raise AssertionError(op)


@pytest.mark.covers(edges=("data", "state"), kinds=("state",))
def test_lookup_longest_prefix_namespace_and_post_change_guard() -> None:
    store = MemoryStore()
    identity = ["v1"]
    cache = Cache(store, "model-one", 4, 1024, lambda: identity[0])
    prompt = list(range(2100))
    shorter, longest = cache.ref(prompt, 4), cache.ref(prompt, 12)
    store.values.update({shorter.key: b"a", longest.key: b"b"})
    assert cache.lookup(prompt) == longest
    assert max(store.queries) <= 256
    assert cache.lookup(prompt[:12]) == shorter  # retain a token for logits
    assert Cache(store, "another-model", 4, 1024).lookup(prompt) is None
    assert cache.lookup([777, *prompt[1:]]) is None
    identity[0] = "v2"
    assert cache.lookup(prompt) is None
    with pytest.raises(RuntimeError, match="reload"):
        cache.load(longest, None, None, 0)


@pytest.mark.covers(edges=("data", "state"), kinds=("state",))
def test_namespace_hash_uses_the_active_snapshot_even_when_the_module_was_not_reimported(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import state.cache
    from state.cache.unit import source_root

    active = tmp_path / "new-snapshot" / "engine"
    cache = Cache(MemoryStore(), "unseen-before-edit", 4, 1024, source=source_root())
    monkeypatch.setattr(state.cache, "__path__", [str(active / "state" / "cache")])
    assert source_root() == active
    assert not cache.active()
