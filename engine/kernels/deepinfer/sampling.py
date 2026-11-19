# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# type: ignore
"""Fuse the reference sampler's Murmur/Gumbel draw; keep its Torch ordering and nucleus sums."""
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _mix(h, k):
    k = k * 0xCC9E2D51
    k = (k << 15) | (k >> 17)
    k = k * 0x1B873593
    h = h ^ k
    h = (h << 13) | (h >> 19)
    return h * 5 + 0xE6546B64


@triton.jit
def _draw(P, CDF, IDX, KS, PS, SEEDS, POS, OUT,
          STRIDE: tl.constexpr, CDF_STRIDE: tl.constexpr, IDX_STRIDE: tl.constexpr,
          WIDTH: tl.constexpr, FILTER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    p = tl.load(P + row * STRIDE + col, valid, 0.0)
    if FILTER:
        total = tl.load(CDF + row * CDF_STRIDE + col, valid, 0.0)
        k = tl.load(KS + row)
        cutoff = tl.load(PS + row)
        p = tl.where((col < k) & (total - p <= cutoff), p, 0.0)
        logp = libdevice.log(p.to(tl.float64))
    else:
        # The reference's unfiltered path takes log in FP32 before the FP64 draw.
        logp = libdevice.log(p).to(tl.float64)
    seed = tl.load(SEEDS + row).to(tl.uint64)
    pos = tl.load(POS + row).to(tl.uint32)
    h = _mix(tl.full((), 0, tl.uint32), seed.to(tl.uint32))
    h = _mix(h, (seed >> 32).to(tl.uint32))
    h = _mix(h, pos)
    h = _mix(h, col.to(tl.uint32)) ^ 16
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    h = h * 0xC2B2AE35
    h = h ^ (h >> 16)
    u = h.to(tl.float64) / tl.full((), 4294967295.0, tl.float64)
    logu = libdevice.log(u)
    logu = tl.maximum(logu, tl.full((), -1.7976931348623157e308, tl.float64))
    logu = tl.minimum(logu, tl.full((), -(2.0 ** -32), tl.float64))
    score = logp - libdevice.log(-logu)
    score = tl.where(valid, score, -float('inf'))
    highest = tl.max(score, 0)
    selected = tl.min(tl.where(score == highest, col, 0x7FFFFFFF), 0)
    if FILTER:
        selected = tl.load(IDX + row * IDX_STRIDE + selected).to(tl.int32)
    tl.store(OUT + row, selected)


def _device(values: Sequence[float] | Sequence[int] | torch.Tensor,
            dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype).contiguous()
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _rotl32(x: torch.Tensor, bits: int) -> torch.Tensor:
    return ((x << bits) | (x >> (32 - bits))) & 0xFFFFFFFF


def _murmur_mix_torch(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h = h ^ k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & 0xFFFFFFFF


def _multinomial_seeded_torch(logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows, cols = logprobs.shape
    seed64 = seed.to(torch.int64)
    col = torch.arange(cols, dtype=torch.int64, device=logprobs.device)
    h = torch.zeros((rows, 1), dtype=torch.int64, device=logprobs.device)
    h = _murmur_mix_torch(h, (seed64 & 0xFFFFFFFF)[:, None])
    h = _murmur_mix_torch(h, ((seed64 >> 32) & 0xFFFFFFFF)[:, None])
    h = _murmur_mix_torch(h, (positions.to(torch.int64) & 0xFFFFFFFF)[:, None])
    h = _murmur_mix_torch(h.expand(rows, cols), (col & 0xFFFFFFFF)[None, :]) ^ 16
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    x = h.to(torch.float64) / 0xFFFFFFFF
    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0 ** -32)).neg_()
    x.log_().neg_()
    x.add_(logprobs.to(torch.float64))
    return torch.argmax(x, dim=1, keepdim=True)


def sample(logits: torch.Tensor, temperatures: Sequence[float], top_ps: Sequence[float],
           top_ks: Sequence[int], seeds: Sequence[int], positions: Sequence[int] | torch.Tensor,
           _cache: "CachedSampler | None" = None) -> torch.Tensor:
    ks = [1 if 0 <= t < 1e-6 else ((1 << 30) if k == -1 else int(k))
          for t, k in zip(temperatures, top_ks)]
    if all(k == 1 for k in ks):
        if _cache is not None:
            fast = _cache.greedy(logits)
            if fast is not None:
                return fast
        return torch.argmax(logits, dim=-1).to(torch.int32)
    device = logits.device
    temps = [1.0 if 0 <= t < 1e-6 else float(t) for t in temperatures]
    key = (device, tuple(temps), tuple(top_ps), tuple(ks), tuple(seeds))
    if _cache is not None and _cache.key == key:
        temp, seed, k, p = _cache.params
    else:
        temp = _device(temps, torch.float32, device)
        seed = _device(seeds, torch.int64, device)
        k = _device(ks, torch.int64, device)
        p = _device(top_ps, torch.float32, device)
        if _cache is not None:
            _cache.key, _cache.params = key, (temp, seed, k, p)
    logits = logits.float().div_(temp.view(-1, 1))
    probs = torch.softmax(logits, dim=-1)
    pos = _device(positions, torch.int64, device)
    filtered = not (all(k == 1 << 30 for k in ks) and all(p == 1.0 for p in top_ps))
    if probs.shape[1] > 1024:
        if filtered:
            probs, indices = probs.sort(dim=-1, descending=True)
            sums = torch.cumsum(probs, dim=-1)
            probs[torch.arange(0, probs.shape[-1], device=device).view(1, -1) >= k.view(-1, 1)] = 0.0
            probs[(sums - probs) > p.view(-1, 1)] = 0.0
            picked = _multinomial_seeded_torch(probs.to(torch.float64).log_(), seed, pos)
            return torch.gather(indices.to(torch.int32), dim=1, index=picked).view(-1)
        return _multinomial_seeded_torch(torch.log(probs), seed, pos).view(-1).to(torch.int32)
    if filtered:
        probs, indices = probs.sort(dim=-1, descending=True)
        sums = torch.cumsum(probs, dim=-1)
    else:
        sums, indices, k, p = probs, seed, seed, seed
    out = torch.empty(probs.shape[0], dtype=torch.int32, device=device)
    _draw[(probs.shape[0],)](probs, sums, indices, k, p, seed, pos, out,
                           probs.stride(0), sums.stride(0), indices.stride(0),
                           probs.shape[1], filtered, triton.next_power_of_2(probs.shape[1]),
                           num_warps=4, enable_fp_fusion=False)
    return out


def force_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_select
    alignment = deep_select.get_stride_requirement()[0] // logits.element_size()
    if logits.stride(0) % alignment or logits.stride(1) != 1:
        padded = torch.empty(logits.shape[0], -(-logits.shape[1] // alignment) * alignment,
                             dtype=logits.dtype, device=logits.device)
        logits = padded[:, :logits.shape[1]].copy_(logits)
    return deep_select.topk(logits, k, sorted=False, indices_type=torch.int32)


class CachedSampler:
    """Keep unchanged request parameters on the device; positions remain a per-step input."""

    def __init__(self) -> None:
        self.key: Any = None
        self.params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._fast_argmax: Any = None

    def greedy(self, logits: torch.Tensor) -> torch.Tensor | None:
        if self._fast_argmax is None:
            from kernels.deepinfer.fast_argmax import FastArgmax

            self._fast_argmax = FastArgmax()
        return self._fast_argmax.run(logits)

    def sample(self, logits: torch.Tensor, temperatures: Sequence[float], top_ps: Sequence[float],
               top_ks: Sequence[int], seeds: Sequence[int], positions: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return sample(logits, temperatures, top_ps, top_ks, seeds, positions, _cache=self)
