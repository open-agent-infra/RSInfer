"""Shared prefix snapshots in Mooncake Store.

The service follows SGLang HiCache's exists/get/put separation and treats auxiliary recurrent
state as part of a hit. It uses bounded host staging, not SGLang's GPU pointer transfer path.
See doc/specs/2026-11-26-mooncake-pd.md for the reference revision and initial support boundary.
"""

from __future__ import annotations

import hashlib
import json
import sys
import struct
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


from core.contracts.cache import CacheRef

from .io import StoreProcess


class Cache:
    def __init__(self, store: Any, namespace: str, page_size: int, max_bytes: int, identity: Callable[[], str] = lambda: "",
                 source: Path | None = None) -> None:
        self.store, self.namespace, self.page_size, self.max_bytes = store, namespace, page_size, max_bytes
        self.enabled = store is not None
        self.identity, self.source = identity, source
        self.served_identity: str | None = None

    def active(self) -> bool:
        if self.source is not None and source_root() != self.source:
            return False  # Also invalidates a cache never used before the first hot edit.
        current = self.identity()
        if self.served_identity is None:
            self.served_identity = current
        return self.enabled and self.served_identity == current

    def ref(self, prompt: Sequence[int], tokens: int) -> CacheRef:
        if not 0 < tokens <= len(prompt):
            raise ValueError("invalid cache prefix length")
        digest = hashlib.sha256(self.namespace.encode())
        for token in prompt[:tokens]:
            digest.update(struct.pack("<q", int(token)))
        return CacheRef("rsinfer-v1-" + digest.hexdigest(), tokens)

    def lookup(self, prompt: Sequence[int]) -> CacheRef | None:
        if not self.active():
            return None
        refs: list[CacheRef] = []
        digest = hashlib.sha256(self.namespace.encode())
        for n, token in enumerate(prompt[:-1], 1):
            digest.update(struct.pack("<q", int(token)))
            if n % self.page_size == 0:
                refs.append(CacheRef("rsinfer-v1-" + digest.hexdigest(), n))
        refs.reverse()
        try:
            # Bound metadata request size even for very long contexts.
            for start in range(0, len(refs), 256):
                batch = refs[start:start + 256]
                found = self.store.call("exists", [r.key for r in batch])
                if not isinstance(found, list) or len(found) != len(batch):
                    raise ValueError("Mooncake returned malformed existence results")
                for ref, hit in zip(batch, found, strict=True):
                    if hit == 1:
                        return ref
        except Exception as exc:
            self.note("lookup_failed", str(exc))
        return None

    def prefill_end(self, start: int, end: int, prompt_len: int) -> int:
        checkpoint = (prompt_len - 1) // self.page_size * self.page_size
        return min(end, checkpoint) if start < checkpoint else end

    def load(self, ref: CacheRef, model: Any, batch: Any, row: int) -> None:
        from .tensors import restore

        if not self.active():
            raise RuntimeError("cache requires a unit reload after the engine identity changed")
        raw = self.store.call("get", ref.key)
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("cache entry was evicted before restore; retry the request")
        restore(raw, ref, model.assembly.states, model.stores,
                {k: batch.ids(k).get(row, []) for k in model.stores}, self.max_bytes)
        self.note("restored", f"tokens={ref.tokens} bytes={len(raw)}")

    def save(self, ref: CacheRef, model: Any, batch: Any, row: int) -> None:
        from .tensors import dump

        if not self.active():
            raise RuntimeError("cache requires a unit reload after the engine identity changed")
        raw = dump(ref, model.assembly.states, model.stores,
                   {k: batch.ids(k).get(row, []) for k in model.stores}, self.max_bytes)
        self.store.call("put", [ref.key, raw])
        self.note("published", f"tokens={ref.tokens} bytes={len(raw)}")

    def remove(self, ref: CacheRef) -> None:
        self.store.call("remove", ref.key)

    def note(self, event: str, detail: str) -> None:
        print(f"kv.cache {event}: {detail}", file=sys.stderr, flush=True)


def source_root() -> Path:
    # Hot apply retargets package search paths; unchanged modules retain their old __file__.
    assert __package__ is not None
    paths = getattr(sys.modules[__package__], "__path__")
    return Path(paths[0]).resolve().parents[1]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        backend = config["cache.backend"]
        store = None
        namespace = ""
        root: Path | None = None
        if backend:
            if backend != "mooncake":
                raise ValueError("cache.backend supports only mooncake or the empty disabled value")
            if (config["alloc.radix_cache"] or config["spec.algorithm"] or config["sched.overlap"]
                    or len(config["sched.workers"]) != 1 or config["sched.domains"] != 1):
                raise ValueError("Mooncake requires one worker/domain, radix cache off, speculation off and overlap off")
            if not config["cache.mooncake_master"] or not config["cache.hostname"]:
                raise ValueError("Mooncake requires cache.mooncake_master and cache.hostname")
            kinds = {d["kind"] for d in config["alloc.states"]}
            if not kinds or not kinds <= {"kv.paged", "gdn.slots"}:
                raise ValueError("Mooncake snapshots currently support Qwen KV and GDN states only")
            if config["cache.timeout_s"] <= 0 or config["cache.max_bytes"] <= 0:
                raise ValueError("cache timeout and size limit must be positive")
            if config["cache.protocol"] not in ("tcp", "rdma"):
                raise ValueError("cache.protocol must be tcp or rdma")
            # Same code/config and declared weight identity share a namespace; hot edits invalidate it.
            root = source_root()
            digest = hashlib.sha256()
            for path in sorted(root.rglob("*.py")):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
            model_config = config["model.hf_config"]
            if not model_config:
                model_config = json.loads((Path(config["model.path"]) / "config.json").read_text())
            weight_path = config["weights.path"]
            files = sorted(Path(weight_path).glob("*.safetensors")) if weight_path else []
            if not files:
                raise ValueError("Mooncake requires immutable checkpoint weights; dummy weights cannot be shared")
            for path in files:
                with path.open("rb") as file:
                    while chunk := file.read(1 << 20):
                        digest.update(chunk)
            semantic = {"id": config["cache.model_id"], "states": config["alloc.states"], "model": model_config,
                        "dtype": config["model.dtype"], "kv": config["state.kv.dtype"],
                        "conv": config["state.gdn.conv_dtype"], "ssm": config["state.gdn.ssm_dtype"]}
            digest.update(json.dumps(semantic, sort_keys=True).encode())
            namespace = digest.hexdigest()
            settings = {"local_hostname": config["cache.hostname"], "metadata_server": config["cache.metadata_server"],
                        "global_segment_size": str(config["cache.segment_bytes"]),
                        "local_buffer_size": str(config["cache.buffer_bytes"]), "protocol": config["cache.protocol"],
                        "rdma_devices": config["cache.rdma_devices"], "master_server_addr": config["cache.mooncake_master"]}
            store = StoreProcess(settings, config["cache.timeout_s"], config["cache.max_bytes"])
            ctx.effect(lambda: store.close, "Mooncake I/O process")
        control = ctx.use("proc.control@1")
        ctx.provide("kv.cache@1", Cache(store, namespace, config["state.kv.page_size"], config["cache.max_bytes"],
                                         lambda: str(control.identity), source=root))
