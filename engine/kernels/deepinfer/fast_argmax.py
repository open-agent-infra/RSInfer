"""1-kernel CUDA C++ Split-K FP32 greedy argmax with optional fused last_token write."""
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
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSplits = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ bool is_better(float nv, int ni, float ov, int oi) {
  // Match torch.argmax: NaN wins if earlier or propagates; otherwise higher value, ties to lower index.
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

__global__ void splitk_argmax_kernel(
    const float* __restrict__ logits,
    int64_t stride_m,
    int N,
    float* __restrict__ part_v,
    int* __restrict__ part_i,
    unsigned int* __restrict__ counters,
    int* __restrict__ out,
    int64_t* __restrict__ last_token,
    const int64_t* __restrict__ rows_plus1) {
  const int row = blockIdx.y;
  const int split = blockIdx.x;
  const int tid = threadIdx.x;

  const int chunk = (N + kSplits - 1) / kSplits;
  const int start = split * chunk;
  const int end = start + chunk < N ? start + chunk : N;
  const float* row_ptr = logits + row * stride_m;

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
      const int poff = row * kSplits + split;
      part_v[poff] = bv;
      part_i[poff] = bi;
      __threadfence();
      const unsigned int done = atomicAdd(&counters[row], 1u) + 1u;
      sm_i[0] = (done == kSplits) ? 1 : 0;
    }
  }
  __syncthreads();

  if (sm_i[0] && warp == 0) {
    const int poff = row * kSplits + lane;
    float fv = part_v[poff];
    int fi = part_i[poff];
    warp_reduce(fv, fi);
    if (lane == 0) {
      counters[row] = 0u;
      if (fi == 0x7fffffff) fi = 0;
      out[row] = fi;
      if (last_token != nullptr && rows_plus1 != nullptr) {
        last_token[rows_plus1[row]] = static_cast<int64_t>(fi);
      }
    }
  }
}
} // namespace

void fast_argmax(
    torch::Tensor logits,
    torch::Tensor part_v,
    torch::Tensor part_i,
    torch::Tensor counters,
    torch::Tensor out,
    c10::optional<torch::Tensor> last_token,
    c10::optional<torch::Tensor> rows_plus1) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.dtype() == torch::kFloat32 && logits.stride(1) == 1);
  const int B = static_cast<int>(logits.size(0));
  const int N = static_cast<int>(logits.size(1));
  if (B == 0 || N == 0) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(kSplits, B);
  int64_t* lt_ptr = last_token.has_value() ? last_token->data_ptr<int64_t>() : nullptr;
  const int64_t* rp_ptr = rows_plus1.has_value() ? rows_plus1->data_ptr<int64_t>() : nullptr;
  splitk_argmax_kernel<<<grid, kThreads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), N,
      part_v.data_ptr<float>(), part_i.data_ptr<int>(),
      reinterpret_cast<unsigned int*>(counters.data_ptr<int>()),
      out.data_ptr<int>(), lt_ptr, rp_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fast_argmax", &fast_argmax);
}
"""


@cache
def operator() -> Any:
    return build_cuda_extension("agx_fast_argmax", "fast_argmax.cu", SOURCE)


class FastArgmax:
    """Per-device reusable scratch buffers for 1-kernel Split-K FP32 greedy argmax."""

    def __init__(self) -> None:
        self._bufs: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._op: Any = None
        self._failed = False

    def run(
        self,
        logits: torch.Tensor,
        last_token: torch.Tensor | None = None,
        rows_plus1: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if (self._failed or not logits.is_cuda or logits.ndim != 2 or logits.dtype != torch.float32
                or logits.stride(-1) != 1 or logits.shape[1] < 4096 or logits.shape[0] > 256):
            return None
        if self._op is None:
            try:
                self._op = operator()
            except Exception:  # noqa: BLE001
                self._failed = True
                return None
            if self._op is None:
                self._failed = True
                return None
        b = logits.shape[0]
        cap = max(128, b)
        dev = logits.device
        bufs = self._bufs.get((dev, cap))
        if bufs is None:
            with torch.inference_mode(False):
                bufs = (
                    torch.empty((cap, 32), dtype=torch.float32, device=dev),
                    torch.empty((cap, 32), dtype=torch.int32, device=dev),
                    torch.zeros((cap,), dtype=torch.int32, device=dev),
                    torch.empty((cap,), dtype=torch.int32, device=dev),
                )
            self._bufs[(dev, cap)] = bufs
        pv, pi, cnt, out_buf = bufs
        out = out_buf[:b]
        self._op.fast_argmax(logits, pv, pi, cnt, out, last_token, rows_plus1)
        return out
