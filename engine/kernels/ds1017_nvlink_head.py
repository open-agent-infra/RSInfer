"""NVLink5 vocab-parallel greedy LM head + fused Split-K argmax + last_token + UVA pinned host write.

When multiple GPUs share an NVLink5 domain (e.g., 4x SM100 DP4/EP4), each rank slices its
1/R vocab shard of the lossless BF16 ``head.weight`` (e.g., 331 MB instead of 1.324 GB per GPU),
pushes its local final hidden states directly into peers' double-buffered symmetric memory,
runs one 1/R-sized ``deep_gemm.bf16_gemm_nt``, and fuses local Split-K FP32 argmax, R-rank
top-1 reduction, ``last_token`` scatter, and direct UVA pinned-host output into one CUDA kernel.

Produces bit-exact identical greedy tokens to full-vocab ``deep_gemm.bf16_gemm_nt`` + ``torch.argmax``
(including NaN propagation and lowest-index tie-breaking), and supports arbitrary per-rank batch
counts including idle (0-row) ranks and mixed greedy/non-greedy steps in lockstep.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from functools import cache
from typing import Any

import torch

from kernels.cuda_build import build_cuda_extension

SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kMaxRanks = 8;
constexpr int kSplits = 16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ void st_release_sys(uint32_t* addr, uint32_t val) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* addr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
  return val;
}

__device__ __forceinline__ bool is_better(float nv, int ni, float ov, int oi) {
  if (isnan(ov)) return isnan(nv) && (ni < oi);
  if (isnan(nv)) return true;
  return (nv > ov) || (nv == ov && ni < oi);
}

__device__ __forceinline__ void warp_reduce(float& v, int& idx) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, v, offset);
    int oi = __shfl_down_sync(0xffffffffu, idx, offset);
    if (is_better(ov, oi, v, idx)) {
      v = ov;
      idx = oi;
    }
  }
}

// Kernel 1 (Direct Push + Barrier):
// Grid: (max(1, n_local)), Block: (256).
// Pushes local_h[row] directly into every peer's symm all_h[slot, rank * cap + row],
// writes n_local into every peer's symm counts[slot, rank], and runs an NVLink5 barrier.
__global__ void vp_push_hidden_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    int64_t all_h_off,
    int64_t all_h_slot_bytes,
    int64_t counts_off,
    int64_t sig1_off,
    const __nv_bfloat16* __restrict__ local_h,
    int64_t local_h_stride_m,
    int n_local,
    int cap_per_rank,
    int K,
    int num_ranks,
    int rank,
    uint32_t seq,
    uint32_t* __restrict__ state_buf) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int slot = static_cast<int>(seq & 1u);
  const int grid_blocks = n_local > 0 ? n_local : 1;

  if (row < n_local) {
    const int n_vec = K >> 3;
    const float4* src = reinterpret_cast<const float4*>(local_h + static_cast<int64_t>(row) * local_h_stride_m);
    const int64_t row_off = all_h_off + slot * all_h_slot_bytes +
                            static_cast<int64_t>(rank * cap_per_rank + row) * K * sizeof(__nv_bfloat16);
    float4* dsts[kMaxRanks];
    for (int p = 0; p < num_ranks; ++p) {
      dsts[p] = reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(symm_ptrs[p]) + row_off);
    }
    for (int i = tid; i < n_vec; i += kThreads) {
      const float4 v = src[i];
      for (int p = 0; p < num_ranks; ++p) {
        dsts[p][i] = v;
      }
    }
  }
  if (row == 0 && tid < num_ranks) {
    int* peer_cnt = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(symm_ptrs[tid]) + counts_off);
    peer_cnt[slot * num_ranks + rank] = n_local;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence_system();
    const unsigned int done = atomicAdd(&state_buf[0], 1u) + 1u;
    if (done == static_cast<unsigned int>(grid_blocks)) {
      state_buf[0] = 0u;
      if (grid_blocks > 1) {
        __threadfence_system();
      }
      for (int p = 0; p < num_ranks; ++p) {
        if (p != rank) {
          uint32_t* peer_sig = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(symm_ptrs[p]) + sig1_off);
          st_release_sys(&peer_sig[rank], seq);
        }
      }
      const uint32_t* my_sig = reinterpret_cast<const uint32_t*>(reinterpret_cast<uint8_t*>(symm_ptrs[rank]) + sig1_off);
      for (int p = 0; p < num_ranks; ++p) {
        if (p != rank) {
          while (ld_acquire_sys(&my_sig[p]) < seq) {}
        }
      }
    }
  }
}

// Kernel 2 (Fused Split-K Local Argmax + Direct Top-1 Push + Barrier + Final Reduce + last_token + UVA):
// Grid: (kSplits, num_ranks * cap_per_rank), Block: (256).
// Inactive rows exit immediately on instruction 1 (zero atomic / fence overhead!).
__global__ void vp_argmax_reduce_kernel(
    const float* __restrict__ logits,
    int64_t stride_m,
    int V_shard,
    int vocab_offset,
    int cap_per_rank,
    int num_ranks,
    int rank,
    int n_local,
    uint32_t seq,
    const uint64_t* __restrict__ symm_ptrs,
    int64_t counts_off,
    int64_t top1_off,
    int64_t sig2_off,
    float* __restrict__ part_v,
    int* __restrict__ part_i,
    unsigned int* __restrict__ row_counters,
    uint32_t* __restrict__ state_buf,
    int* __restrict__ out_dev,
    int* __restrict__ out_host_uva,
    int64_t* __restrict__ last_token,
    const int64_t* __restrict__ rows_plus1) {
  const int global_row = blockIdx.y;
  const int peer = global_row / cap_per_rank;
  const int local_row = global_row - peer * cap_per_rank;
  const int split = blockIdx.x;
  const int tid = threadIdx.x;
  const int slot = static_cast<int>(seq & 1u);

  const uint8_t* my_base = reinterpret_cast<const uint8_t*>(symm_ptrs[rank]);
  const int* my_counts = reinterpret_cast<const int*>(my_base + counts_off) + slot * num_ranks;
  const int n_p = my_counts[peer];

  int total_active = 0;
  for (int p = 0; p < num_ranks; ++p) {
    total_active += my_counts[p];
  }

  if (total_active == 0) {
    if (global_row != 0 || split != 0) return;
    if (tid == 0) {
      for (int p = 0; p < num_ranks; ++p) {
        if (p != rank) {
          uint32_t* peer_sig = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(symm_ptrs[p]) + sig2_off);
          st_release_sys(&peer_sig[rank], seq);
        }
      }
      const uint32_t* my_sig = reinterpret_cast<const uint32_t*>(my_base + sig2_off);
      for (int p = 0; p < num_ranks; ++p) {
        if (p != rank) {
          while (ld_acquire_sys(&my_sig[p]) < seq) {}
        }
      }
    }
    return;
  }

  if (local_row >= n_p) return;

  const int chunk = (V_shard + kSplits - 1) / kSplits;
  const int start = split * chunk;
  const int end = start + chunk < V_shard ? start + chunk : V_shard;
  const float* row_ptr = logits + static_cast<int64_t>(global_row) * stride_m;

  float bv = -INFINITY;
  int bi = 0x7fffffff;
  int i = start;
  if (start < end && (((reinterpret_cast<uintptr_t>(row_ptr) | (static_cast<uintptr_t>(start) * sizeof(float))) & 15) == 0)) {
    const int vec_start = start >> 2;
    const int vec_end = end >> 2;
    const float4* ptr4 = reinterpret_cast<const float4*>(row_ptr);
    for (int vi = vec_start + tid; vi < vec_end; vi += kThreads) {
      const float4 v4 = ptr4[vi];
      const int base_i = vi << 2;
      if (is_better(v4.x, base_i, bv, bi)) { bv = v4.x; bi = base_i; }
      if (is_better(v4.y, base_i + 1, bv, bi)) { bv = v4.y; bi = base_i + 1; }
      if (is_better(v4.z, base_i + 2, bv, bi)) { bv = v4.z; bi = base_i + 2; }
      if (is_better(v4.w, base_i + 3, bv, bi)) { bv = v4.w; bi = base_i + 3; }
    }
    i = vec_end << 2;
  }
  for (int j = i + tid; j < end; j += kThreads) {
    const float v = row_ptr[j];
    if (is_better(v, j, bv, bi)) { bv = v; bi = j; }
  }

  warp_reduce(bv, bi);
  __shared__ float sm_v[kWarps];
  __shared__ int sm_i[kWarps];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    sm_v[warp] = bv;
    sm_i[warp] = bi;
  }
  __syncthreads();

  if (warp == 0) {
    bv = (lane < kWarps) ? sm_v[lane] : -INFINITY;
    bi = (lane < kWarps) ? sm_i[lane] : 0x7fffffff;
    warp_reduce(bv, bi);
    if (lane == 0) {
      const int poff = global_row * kSplits + split;
      part_v[poff] = bv;
      part_i[poff] = bi;
      __threadfence();
      const unsigned int done = atomicAdd(&row_counters[global_row], 1u) + 1u;
      sm_i[0] = (done == kSplits) ? 1 : 0;
    }
  }
  __syncthreads();

  if (!sm_i[0]) return;

  __shared__ int is_last_block;
  if (warp == 0) {
    float fv = (lane < kSplits) ? part_v[global_row * kSplits + lane] : -INFINITY;
    int fi = (lane < kSplits) ? part_i[global_row * kSplits + lane] : 0x7fffffff;
    warp_reduce(fv, fi);
    if (lane == 0) {
      row_counters[global_row] = 0u;
      if (fi == 0x7fffffff) fi = 0;
      const int global_idx = fi + vocab_offset;
      uint8_t* peer_top1 = reinterpret_cast<uint8_t*>(symm_ptrs[peer]) + top1_off;
      const int elem = (slot * num_ranks + rank) * cap_per_rank + local_row;
      const int total_top1 = 2 * num_ranks * cap_per_rank;
      reinterpret_cast<float*>(peer_top1)[elem] = fv;
      reinterpret_cast<int*>(peer_top1 + total_top1 * sizeof(float))[elem] = global_idx;

      __threadfence_system();
      const unsigned int gdone = atomicAdd(&state_buf[1], 1u) + 1u;
      if (gdone == static_cast<unsigned int>(total_active)) {
        state_buf[1] = 0u;
        if (total_active > 1) {
          __threadfence_system();
        }
        for (int p = 0; p < num_ranks; ++p) {
          if (p != rank) {
            uint32_t* peer_sig = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(symm_ptrs[p]) + sig2_off);
            st_release_sys(&peer_sig[rank], seq);
          }
        }
        const uint32_t* my_sig = reinterpret_cast<const uint32_t*>(my_base + sig2_off);
        for (int p = 0; p < num_ranks; ++p) {
          if (p != rank) {
            while (ld_acquire_sys(&my_sig[p]) < seq) {}
          }
        }
        is_last_block = 1;
      } else {
        is_last_block = 0;
      }
    }
  }
  __syncthreads();

  if (is_last_block && tid < n_local) {
    const uint8_t* my_top1 = my_base + top1_off;
    const int total_top1 = 2 * num_ranks * cap_per_rank;
    const float* my_val = reinterpret_cast<const float*>(my_top1);
    const int* my_idx = reinterpret_cast<const int*>(my_top1 + total_top1 * sizeof(float));

    float rv = -INFINITY;
    int ri = 0x7fffffff;
    for (int p = 0; p < num_ranks; ++p) {
      const int elem = (slot * num_ranks + p) * cap_per_rank + tid;
      const float v = my_val[elem];
      const int idx = my_idx[elem];
      if (is_better(v, idx, rv, ri)) {
        rv = v;
        ri = idx;
      }
    }
    if (ri == 0x7fffffff) ri = 0;
    if (out_dev != nullptr) out_dev[tid] = ri;
    if (out_host_uva != nullptr) out_host_uva[tid] = ri;
    if (last_token != nullptr && rows_plus1 != nullptr) {
      last_token[rows_plus1[tid]] = static_cast<int64_t>(ri);
    }
  }
}
} // namespace

void vp_push_hidden(
    torch::Tensor symm_ptrs, int64_t all_h_off, int64_t all_h_slot_bytes, int64_t counts_off, int64_t sig1_off,
    torch::Tensor local_h, int n_local, int cap_per_rank, int K, int num_ranks, int rank, int64_t seq,
    torch::Tensor state_buf) {
  const c10::cuda::CUDAGuard guard(symm_ptrs.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* lh_ptr = n_local > 0 ? reinterpret_cast<const __nv_bfloat16*>(local_h.data_ptr()) : nullptr;
  const int64_t stride_m = n_local > 0 ? local_h.stride(0) : K;
  const int grid_blocks = n_local > 0 ? n_local : 1;
  vp_push_hidden_kernel<<<grid_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(symm_ptrs.data_ptr<int64_t>()),
      all_h_off, all_h_slot_bytes, counts_off, sig1_off,
      lh_ptr, stride_m, n_local, cap_per_rank, K, num_ranks, rank,
      static_cast<uint32_t>(seq), reinterpret_cast<uint32_t*>(state_buf.data_ptr<int32_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void vp_argmax_reduce(
    torch::Tensor logits, int vocab_offset, int cap_per_rank, int num_ranks, int rank, int n_local, int64_t seq,
    torch::Tensor symm_ptrs, int64_t counts_off, int64_t top1_off, int64_t sig2_off,
    torch::Tensor part_v, torch::Tensor part_i, torch::Tensor row_counters, torch::Tensor state_buf,
    c10::optional<torch::Tensor> out_dev, c10::optional<torch::Tensor> out_host_uva,
    c10::optional<torch::Tensor> last_token, c10::optional<torch::Tensor> rows_plus1) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int V_shard = static_cast<int>(logits.size(1));
  dim3 grid(kSplits, num_ranks * cap_per_rank);
  int* od = out_dev.has_value() ? out_dev->data_ptr<int>() : nullptr;
  int* oh = out_host_uva.has_value() ? out_host_uva->data_ptr<int>() : nullptr;
  int64_t* lt = last_token.has_value() ? last_token->data_ptr<int64_t>() : nullptr;
  const int64_t* rp = rows_plus1.has_value() ? rows_plus1->data_ptr<int64_t>() : nullptr;
  vp_argmax_reduce_kernel<<<grid, kThreads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), V_shard, vocab_offset, cap_per_rank, num_ranks, rank, n_local,
      static_cast<uint32_t>(seq), reinterpret_cast<const uint64_t*>(symm_ptrs.data_ptr<int64_t>()),
      counts_off, top1_off, sig2_off,
      part_v.data_ptr<float>(), part_i.data_ptr<int>(),
      reinterpret_cast<unsigned int*>(row_counters.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(state_buf.data_ptr<int32_t>()),
      od, oh, lt, rp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vp_push_hidden", &vp_push_hidden);
  m.def("vp_argmax_reduce", &vp_argmax_reduce);
}
"""


@cache
def operator() -> Any:
    return build_cuda_extension("agx_nvlink_head", "nvlink_head.cu", SOURCE)


class NvlinkVocabHead:
    """Single-node NVLink5 vocab-parallel greedy LM head + Split-K argmax."""

    def __init__(self, group: Any, vocab_size: int, hidden_size: int, cap_per_rank: int, device: torch.device) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if group.size <= 1 or group.size > 8 or group.device_group is None:
            raise ValueError("NvlinkVocabHead requires 2..8 NVLink ranks with an active NCCL device_group")
        if hidden_size % 8 != 0 or vocab_size % group.size != 0 or (vocab_size // group.size) % 16 != 0:
            raise ValueError(
                f"Unsupported vocab/hidden for VP head: vocab={vocab_size}, hidden={hidden_size}, ranks={group.size}"
            )
        if not 1 <= cap_per_rank <= 256:
            raise ValueError(f"cap_per_rank must be in 1..256, got {cap_per_rank}")

        op = operator()
        if op is None:
            raise RuntimeError("failed to build agx_nvlink_head CUDA extension")
        self._op = op
        self.num_ranks = int(group.size)
        self.rank = int(group.rank)
        self.vocab_size = vocab_size
        self.vocab_shard = vocab_size // self.num_ranks
        self.vocab_offset = self.rank * self.vocab_shard
        self.hidden_size = hidden_size
        self.cap_per_rank = cap_per_rank
        self.total_rows = self.num_ranks * cap_per_rank
        self.device = device
        self._seq = 0
        self._shard_cache: tuple[int, int, torch.Tensor | None] = (0, -1, None)

        all_h_slot_bytes = ((self.total_rows * hidden_size * 2 + 255) // 256) * 256
        self._all_h_off = 0
        self._all_h_slot_bytes = all_h_slot_bytes
        self._counts_off = 2 * all_h_slot_bytes
        self._sig1_off = self._counts_off + 256
        self._top1_off = self._sig1_off + 256
        top1_bytes = ((2 * self.total_rows * 8 + 255) // 256) * 256
        self._sig2_off = self._top1_off + top1_bytes
        total_bytes = self._sig2_off + 256

        torch.cuda.synchronize(device)
        with torch.inference_mode(False):
            self._sym_buf = symm_mem.empty((total_bytes,), dtype=torch.uint8, device=device).zero_()
            self._handle = symm_mem.rendezvous(self._sym_buf, group=group.device_group)
            self._symm_ptrs = torch.tensor(list(self._handle.buffer_ptrs), dtype=torch.int64, device=device)
            slot_elems = self.total_rows * hidden_size
            self._all_h_slots = (
                self._sym_buf[self._all_h_off:self._all_h_off + slot_elems * 2]
                .view(torch.bfloat16)
                .view(self.total_rows, hidden_size),
                self._sym_buf[self._all_h_off + all_h_slot_bytes:self._all_h_off + all_h_slot_bytes + slot_elems * 2]
                .view(torch.bfloat16)
                .view(self.total_rows, hidden_size),
            )
            self._all_logits = torch.empty((self.total_rows, self.vocab_shard), dtype=torch.float32, device=device)
            self._part_v = torch.empty((self.total_rows, 16), dtype=torch.float32, device=device)
            self._part_i = torch.empty((self.total_rows, 16), dtype=torch.int32, device=device)
            self._row_counters = torch.zeros((self.total_rows,), dtype=torch.int32, device=device)
            self._state_buf = torch.zeros((2,), dtype=torch.int32, device=device)
            self._out_dev = torch.empty((cap_per_rank,), dtype=torch.int32, device=device)
            self._empty_h = torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize(device)

    def _head_shard(self, weight: torch.Tensor) -> torch.Tensor | None:
        base = weight._base if weight._base is not None else weight
        ver = -1 if base.is_inference() else base._version
        ptr = weight.data_ptr()
        if self._shard_cache[0] == ptr and self._shard_cache[1] == ver:
            return self._shard_cache[2]
        with torch.inference_mode(False):
            sl = weight[self.vocab_offset:self.vocab_offset + self.vocab_shard]
            if sl.dtype == torch.bfloat16:
                packed: torch.Tensor | None = sl.detach().contiguous()
            elif sl.dtype == torch.float32:
                cand = sl.detach().to(torch.bfloat16).contiguous()
                packed = cand if torch.equal(sl, cand.float()) else None
            else:
                packed = None
        self._shard_cache = (ptr, ver, packed)
        return packed

    def supports_weight(self, weight: torch.Tensor) -> bool:
        return self._head_shard(weight) is not None

    def step(
        self,
        hidden: torch.Tensor | None,
        head_weight: torch.Tensor,
        *,
        out_host_uva: torch.Tensor | None = None,
        last_token: torch.Tensor | None = None,
        rows_plus1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one lockstep vocab-parallel greedy step across all ranks.

        ``hidden`` is ``[n_local, hidden_size]`` BF16 (``0 <= n_local <= cap_per_rank``),
        or ``None`` / 0-row tensor for idle or non-greedy ranks participating in the collective.
        Returns device ``int32`` tokens ``[n_local]`` and writes ``out_host_uva`` and
        ``last_token[rows_plus1]`` when supplied.
        """
        import deep_gemm

        w_shard = self._head_shard(head_weight)
        if w_shard is None:
            raise RuntimeError("NvlinkVocabHead requires lossless BF16-convertible head.weight")
        if hidden is None or hidden.shape[0] == 0:
            lh = self._empty_h
            n_local = 0
        else:
            lh = (hidden if hidden.dtype == torch.bfloat16 and hidden.stride(-1) == 1
                  else hidden.to(torch.bfloat16).contiguous())
            n_local = int(lh.shape[0])
            if n_local > self.cap_per_rank:
                raise ValueError(f"n_local={n_local} exceeds NvlinkVocabHead cap_per_rank={self.cap_per_rank}")

        self._seq += 1
        seq = self._seq
        slot = seq & 1
        self._op.vp_push_hidden(
            self._symm_ptrs, self._all_h_off, self._all_h_slot_bytes, self._counts_off, self._sig1_off,
            lh, n_local, self.cap_per_rank, self.hidden_size, self.num_ranks, self.rank, seq, self._state_buf,
        )
        deep_gemm.bf16_gemm_nt(self._all_h_slots[slot], w_shard, self._all_logits, None)
        out_dev = self._out_dev[:n_local] if n_local > 0 else None
        self._op.vp_argmax_reduce(
            self._all_logits, self.vocab_offset, self.cap_per_rank, self.num_ranks, self.rank, n_local, seq,
            self._symm_ptrs, self._counts_off, self._top1_off, self._sig2_off,
            self._part_v, self._part_i, self._row_counters, self._state_buf,
            out_dev, out_host_uva if n_local > 0 else None,
            last_token if n_local > 0 else None, rows_plus1 if n_local > 0 else None,
        )
        return out_dev if out_dev is not None else self._out_dev[:0]

    def close(self) -> None:
        self._shard_cache = (0, -1, None)
        if hasattr(self, "_handle"):
            torch.cuda.synchronize(self.device)
            del self._handle
        if hasattr(self, "_sym_buf"):
            del self._sym_buf
