"""Opt-in REAL Mooncake integration, on CPU or one GPU; no fake store and no environment skips.

Run this file explicitly with RSINFER_MOONCAKE_MASTER and RSINFER_MOONCAKE_HOST set.
RSINFER_TEST_DEVICE is cpu by default, or cuda for the same test on a single GPU.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.scenarios.deploy import post, served
from tests.units.support.tokenizer import save_tokenizer
from tests.units.support.worker import hf_model, save_checkpoint, tiny_config


@pytest.mark.timeout(300)  # cold CUDA extension compilation is part of this explicit integration test
@pytest.mark.covers(edges=("data", "call", "state"), kinds=("logic", "state"))
def test_real_mooncake_reuses_state_across_independent_engines(tmp_path: Path) -> None:
    master, hostname = os.environ["RSINFER_MOONCAKE_MASTER"], os.environ["RSINFER_MOONCAKE_HOST"]
    device = os.environ.get("RSINFER_TEST_DEVICE", "cpu")
    cfg = tiny_config(vocab_size=300, max_position_embeddings=256)
    checkpoint = save_checkpoint(hf_model(cfg, seed=3), tmp_path / "weights")
    (checkpoint / "config.json").write_text(json.dumps({**cfg.to_dict(), "architectures": ["Qwen3_5ForCausalLM"],
                                                       "model_type": "qwen3_5_text", "eos_token_id": 0}))
    tokenizer = save_tokenizer(tmp_path / "tokenizer", 300, eos_in_config=False)
    args = ["--model", str(checkpoint), "--tokenizer", str(tokenizer), "--device", device, "--dtype", "f32",
            "--conv-dtype", "f32", "--page-size", "4", "--max-rows", "8", "--max-context", "128",
            "--token-budget", "16", "--cpu-bytes", str(16 << 20), "--mem-fraction", "0.15", "--timeout", "120",
            "--disable-radix-cache", "--set", "state.kv.pages=512", "--set", "state.gdn.slots=16"]
    cache = ["--set", "cache.backend=mooncake", "--set", f"cache.mooncake_master={master}",
             "--set", f"cache.hostname={hostname}"]
    prompts = [" ".join(f"w{i % 60}" for i in range(n)) for n in (5, 17, 35)]

    def body(prompt: str) -> dict[str, object]:
        return {"model": "Qwen3.8-27B", "prompt": prompt, "max_tokens": 8, "temperature": 0.0, "ignore_eos": True}

    with served(tmp_path / "baseline", *args) as base:
        expected = [post(base + "/v1/completions", body(p))["choices"] for p in prompts]
    with served(tmp_path / "writer", *args, *cache) as writer, served(tmp_path / "reader", *args, *cache) as reader:
        assert [post(writer + "/v1/completions", body(p))["choices"] for p in prompts] == expected
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda p: post(reader + "/v1/completions", body(p)), prompts))
        assert [r["choices"] for r in results] == expected
        logs = "\n".join(p.read_text() for p in (tmp_path / "reader" / "host-a" / "snapshots" / ".run").glob("*/log"))
        assert "kv.cache restored:" in logs and "tokens=32" in logs, logs[-4000:]
        assert "cache restore failed" not in logs
