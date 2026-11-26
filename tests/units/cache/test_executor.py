"""A real hybrid worker restores every state and isolates one failed import in a mixed batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest
import torch

from core.contracts.cache import CacheRef
from core.contracts.records import Resync
from state.cache.unit import Cache
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config


class Store:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def call(self, operation: str, value: Any) -> Any:
        if operation == "put":
            self.data[value[0]] = value[1]
            return None
        if operation == "get":
            return self.data.get(value, b"")
        raise AssertionError(operation)


@pytest.mark.covers(edges=("data", "call", "state"), kinds=("logic", "state"))
def test_imported_gdn_and_kv_match_full_prefill_and_a_miss_does_not_poison_other_rows(tmp_path: Path) -> None:
    cfg = tiny_config()
    weights = save_checkpoint(hf_model(cfg, seed=9), tmp_path / "weights")
    worker = Worker(cfg.to_dict(), **{"weights.path": str(weights), "state.gdn.slots": 8})
    try:
        driver = Driver(worker)
        prompt = [4, 7, 13, 2, 9, 21, 6, 17, 12, 5, 8, 16]
        driver.add(0, prompt)
        expected = driver.step({0: 12}).logits.clone()
        driver.add(1, prompt[:8])
        driver.step({1: 8})
        store = Store()
        cache = Cache(store, "test-model", 4, 1 << 20)
        model, batch = worker.service("model@1"), worker.service("worker.batch@1")
        ref = cache.ref(prompt, 8)
        cache.save(ref, model, batch, 1)
        executor = worker.service("executor@1")
        executor.cache, executor.model = cache, model
        driver.add(2, prompt).computed = 8
        driver.add(3, prompt).computed = 8
        driver.add(4, prompt)
        delta, plan, state = driver.records({2: 4, 3: 4, 4: 12})
        delta = msgspec.structs.replace(delta, added=[msgspec.structs.replace(
            a, cache=ref if a.row == 2 else CacheRef("evicted", 8) if a.row == 3 else None) for a in delta.added])
        result = executor.submit(3, delta, plan, state)
        assert not isinstance(result, Resync)
        sampled = result.finish()
        assert list(sampled.rows.values()) == [2, 3, 4]
        assert list(sampled.counts.values()) == [1, 0, 1]
        assert set(sampled.errors) == {3} and "evicted" in sampled.errors[3]
        assert list(sampled.tokens.values()) == [int(expected.argmax()), int(expected.argmax())]
        torch.testing.assert_close(result.logits[0], expected[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result.logits[1], expected[0], atol=1e-4, rtol=1e-4)
    finally:
        worker.close()
