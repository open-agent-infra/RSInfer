"""``worker.comm``: this worker's place in its communication group (``comm.group@1``;
doc/specs/v1-ds1017.md stage 3). The ranks of a group compute each step together: with expert
parallelism every rank holds part of each layer's experts, and a token's experts are run by the ranks
that hold them.

It is ``torch.distributed``, set up as DeepInfer does (``worker/base.py``,
``parallel_utils/parallel_state.py``, a679bb8): the default group over gloo, found through
``comm.rendezvous``, and with ``comm.device_backend = nccl`` a group of the same ranks for tensors on
the cards. DeepInfer's kernel libraries take that ``ProcessGroup`` (``Group.device_group``).

A process joins once. ``torch.distributed``'s default group is the process's, not this unit's: a
reload of this unit finds it joined and keeps it, and a change of the group's size or rendezvous
needs new processes. The group for the cards is this unit's: made when it loads, given up when it
unloads; a change reloads this unit on every rank at once, so they make the new one together. A rank
that dies takes the group with it: the others' next step fails, and the engine has to be started
again.

``Group.gather`` and ``Group.reduce`` are the two collectives the reference kernels are written
with (``kernels.ds1017_ep``). Every rank calls them in the same order: a step is one forward on
every rank, an empty one where a rank has no rows (``model.idle``).
"""

from __future__ import annotations

import datetime
import math
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

BACKENDS = ("gloo", "nccl")


class Group:
    """``comm.group@1``."""

    def __init__(self, size: int = 1, rank: int = 0, device_group: Any = None) -> None:
        self.size, self.rank = size, rank
        self.device_group = device_group
        self.uniform_prefill = False
        """The ``ProcessGroup`` for tensors where they are (NCCL); ``None``: they go through the host."""
        self.collectives = 0
        """How many ``gather`` and ``reduce`` calls this rank made: equal on every rank after a step."""

    def gather(self, tensors: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        """Each tensor ``[n, ...]`` of this rank with the other ranks' (each rank its own ``n``, 0
        too) as ``[sum of n, ...]``, in rank order; and every rank's ``n``."""
        n = tensors[0].shape[0]
        if self.size == 1:
            return list(tensors), [n]
        self.collectives += 1
        counts = self._all_gather(torch.tensor([n], dtype=torch.int64)).tolist()
        width = max(counts)
        out: list[torch.Tensor] = []
        for t in tensors:
            if t.shape[0] != n:
                raise ValueError("the tensors gathered together have one row count")
            # As bytes: the collectives of gloo take few dtypes (no bf16). A rank without rows knows
            # a row's size too.
            size = t.element_size() * math.prod(t.shape[1:])
            padded = torch.zeros(width, size, dtype=torch.uint8, device=t.device)
            padded[:n] = t.contiguous().view(torch.uint8).reshape(n, size)
            every = self._all_gather(padded)
            kept = torch.cat([every[r * width:r * width + c] for r, c in enumerate(counts)])
            out.append(kept.view(t.dtype).reshape(sum(counts), *t.shape[1:]))
        return out, counts

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """The sum of the ranks' ``tensor`` (the same shape on every rank; fp32 or an integer type)."""
        if self.size == 1:
            return tensor
        self.collectives += 1
        if self.device_group is not None and tensor.is_cuda:
            out = tensor.clone()
            dist.all_reduce(out, group=self.device_group)
            return out
        out = tensor.cpu().clone()
        dist.all_reduce(out)
        return out.to(tensor.device)

    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """``[size * n, ...]``: every rank's ``tensor`` ``[n, ...]``, in rank order."""
        if self.device_group is not None and tensor.is_cuda:
            out = torch.empty(self.size * tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(out, tensor.contiguous(), group=self.device_group)
            return out
        host = tensor.cpu().contiguous()
        out = torch.empty(self.size * host.shape[0], *host.shape[1:], dtype=host.dtype)
        dist.all_gather_into_tensor(out, host)
        return out.to(tensor.device)


def join(size: int, rank: int, rendezvous: str, device_backend: str, timeout_s: float) -> Group:
    """This process's place in a group of ``size`` ranks; it waits for the others to come."""
    if not 0 <= rank < size:
        raise ValueError(f"rank {rank} of a group of comm.size = {size}")
    if device_backend not in BACKENDS:
        raise ValueError(f"comm.device_backend = {device_backend!r}: one of {BACKENDS}")
    # Every rank is a process that sees its one card as device 0. torch's symmetric memory (the buffer
    # DeepGEMM's fused MoE kernel shares between the ranks) takes ranks with the same device index for
    # ranks on one card unless told otherwise; DeepInfer sets the same (deepinfer/__init__.py).
    os.environ.setdefault("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "1")
    if dist.is_initialized():  # a reload of this unit: the process joined already
        if dist.get_world_size() != size or dist.get_rank() != rank:
            raise ValueError(f"this process is rank {dist.get_rank()} of {dist.get_world_size()}, and a group is "
                             f"joined once: rank {rank} of {size} needs a new process")
    else:
        if not rendezvous:
            raise ValueError(f"comm.size = {size} needs comm.rendezvous (tcp://host:port or file:///path)")
        dist.init_process_group(backend="gloo", init_method=rendezvous, rank=rank, world_size=size,
                                timeout=datetime.timedelta(seconds=timeout_s))
    device_group = None
    if device_backend == "nccl":
        device_group = dist.new_group(list(range(size)), backend="nccl")
        # NCCL builds its communicator at the first collective: now, not inside a step.
        dist.all_reduce(torch.zeros(1, device="cuda"), group=device_group)
    return Group(size, rank, device_group)


def leave(group: Group) -> None:
    """Give up the group for the cards; the default group stays, as the process's."""
    if group.device_group is not None:
        dist.destroy_process_group(group.device_group)
        group.device_group = None


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        size = config["comm.size"]
        if size < 1:
            raise ValueError(f"comm.size = {size} must be at least 1")
        group = Group() if size == 1 else join(size, ctx.instance.proc.index, config["comm.rendezvous"],
                                               config["comm.device_backend"], config["comm.timeout_s"])
        ctx.effect(lambda: lambda: leave(group), label="device group")
        ctx.provide("comm.group@1", group)
