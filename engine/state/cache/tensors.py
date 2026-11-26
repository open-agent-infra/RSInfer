"""Portable, validated snapshots of all declared state, with private destination items."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import msgspec
import torch

from core.contracts.cache import CacheRef
from core.contracts.state import StateDecl


class TensorImage(msgspec.Struct):
    kind: str
    layer: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    data: bytes


class Image(msgspec.Struct):
    version: int
    ref: CacheRef
    tensors: list[TensorImage]
    digest: str


def views(decls: Sequence[StateDecl], stores: Mapping[str, Any], ids: Mapping[str, list[int]],
          tokens: int) -> Iterator[tuple[StateDecl, str, torch.Tensor, list[int]]]:
    for d in sorted(decls, key=lambda d: (d.kind, d.layer)):
        count = 1 if d.lifetime == "slot" else -(-tokens // d.tokens_per_item)
        items = ids.get(d.kind, [])[:count]
        if len(items) != count or len(set(items)) != count or any(i < 0 or i >= stores[d.kind].capacity() for i in items):
            raise ValueError("snapshot destination has insufficient or invalid private items")
        for name, buffer in sorted(stores[d.kind].bind(d.layer).buffers().items()):
            shape = d.tensors[name][0]
            yield d, name, buffer.reshape(-1, *shape), items


def dump(ref: CacheRef, decls: Sequence[StateDecl], stores: Mapping[str, Any], ids: Mapping[str, list[int]],
         limit: int) -> bytes:
    tensors: list[TensorImage] = []
    size = 0
    for d, name, buffer, items in views(decls, stores, ids, ref.tokens):
        size += len(items) * buffer[0].numel() * buffer.element_size()
        if size > limit:
            raise ValueError("snapshot exceeds cache.max_bytes")
        index = torch.tensor([i + 1 for i in items], device=buffer.device)
        tensor = buffer.index_select(0, index).to("cpu").contiguous()
        # Never publish the uncomputed tail of a partial last page (it may belong to an old request).
        if d.lifetime == "request" and ref.tokens % d.tokens_per_item:
            per = d.tensors[name][0][0]
            used = -(-(ref.tokens % d.tokens_per_item) * per // d.tokens_per_item)
            tensor[-1, used:] = 0
        data = tensor.view(torch.uint8).numpy().tobytes()
        tensors.append(TensorImage(d.kind, d.layer, name, tuple(tensor.shape), str(tensor.dtype), data))
    digest = hashlib.sha256(msgspec.msgpack.encode(tensors)).hexdigest()
    raw = msgspec.msgpack.encode(Image(1, ref, tensors, digest))
    if len(raw) > limit:
        raise ValueError("snapshot exceeds cache.max_bytes")
    return raw


def restore(raw: bytes, ref: CacheRef, decls: Sequence[StateDecl], stores: Mapping[str, Any],
            ids: Mapping[str, list[int]], limit: int) -> None:
    if not raw or len(raw) > limit:
        raise ValueError("snapshot missing or exceeds cache.max_bytes")
    image = msgspec.msgpack.decode(raw, type=Image)
    if image.version != 1 or image.ref != ref or image.digest != hashlib.sha256(
            msgspec.msgpack.encode(image.tensors)).hexdigest():
        raise ValueError("snapshot version, key or checksum mismatch")
    expected = list(views(decls, stores, ids, ref.tokens))
    if len(expected) != len(image.tensors):
        raise ValueError("snapshot state set mismatch")
    pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    # Validate every tensor before the first destination write.
    for (d, name, buffer, items), saved in zip(expected, image.tensors, strict=True):
        shape = (len(items), *buffer.shape[1:])
        size = len(items) * buffer[0].numel() * buffer.element_size()
        if ((saved.kind, saved.layer, saved.name, saved.shape, saved.dtype) !=
                (d.kind, d.layer, name, shape, str(buffer.dtype)) or len(saved.data) != size):
            raise ValueError("snapshot tensor layout mismatch")
        source = (torch.frombuffer(bytearray(saved.data), dtype=buffer.dtype).reshape(shape)
                  if size else torch.empty(shape, dtype=buffer.dtype))
        index = torch.tensor([i + 1 for i in items], device=buffer.device)
        pending.append((buffer, index, source))
    for buffer, index, source in pending:
        buffer.index_copy_(0, index, source.to(buffer.device))
