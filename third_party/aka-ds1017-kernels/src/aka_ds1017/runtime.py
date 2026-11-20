from functools import lru_cache
import logging
import os

import deep_gemm
from flash_mla import fused_norm_rope_attn_rope_cast as flash_mla
import torch

from .loader import load
from .indexer import index

logger = logging.getLogger(__name__)


class Kernels:
    index = staticmethod(index)

    def __init__(self, policy):
        self.policy = policy
        self.log_route = logger.warning if os.getenv("AKA_DS1017_TRACE") == "1" else logger.info
        self.pick = lru_cache(maxsize=256)(self._pick)
        if torch.cuda.get_device_capability() != (10, 0) or "B200" not in torch.cuda.get_device_name():
            raise ValueError("The DS1017 v1 profile requires NVIDIA B200")
        if torch.version.cuda != "13.2" or torch.__version__.split("+")[0] != "2.13.0":
            raise ValueError("The DS1017 v1 profile requires torch 2.13.0 with CUDA 13.2")

    def _pick(self, op, rows, path):
        name = self.policy.select(op, rows, path)
        self.log_route("DS1017 profile=%s budget=%d op=%s rows=%d path=%s impl=%s",
                    self.policy.profile, self.policy.budget, op, rows, path, name or "official")
        return load(name) if name else None

    def mhc(self, x, *args, **kw):
        path = "attn" if kw.get("y_gemm_sf") is not None else "moe" if kw.get("y_routed_sf") is not None else "bf16"
        m = self.pick("mhc", x.shape[0], path) if x.shape[-1] == 5120 and kw["hc_mult"] == 4 else None
        return (m.native if m else deep_gemm).mega_mhc(x, *args, **kw)

    def gemm(self, a, b, out, bias, **kw):
        rows, cols = out.shape
        path = {(1792,5120): "qkv", (36864,1280): "qb", (5120,8192): "wo_b"}.get((cols,a[0].shape[-1]))
        m = self.pick("projection", rows, path) if path and bias is None and kw.get("recipe") == (1,1,32) else None
        if m and path == "qkv" and rows in (2240,32768,65536):
            return m.qkv(a,b,out)
        return m.native.agx_gemm(a,b,out) if m else deep_gemm.fp8_fp4_gemm_nt(a,b,out,bias,**kw)

    def einsum(self, eq, a, b, out, **kw):
        m = self.pick("projection", a[0].shape[0], "wo_a") if eq == "bhr,hdr->bhd" and tuple(a[0].shape[1:]) == (8,4096) and kw.get("recipe") == (1,1,32) else None
        return m.native.agx_wo(a,b,out) if m else deep_gemm.fp8_einsum(eq,a,b,out,**kw)

    def prefill(self, **kw):
        q = kw["q"]
        path = "sparse640" if kw["indices"].shape[-1] == 640 else "prefill"
        m = self.pick("mla",q.shape[0],path) if q.shape[1:] == (64,512) and kw["rope_dim"] == 64 else None
        return (m.prefill if m else flash_mla.prefill)(**kw)

    def paged(self, **kw):
        q = kw["q"]
        m = self.pick("mla",q.shape[0],"paged") if q.shape[1:] == (64,512) and kw["rope_dim"] == 64 else None
        if m is None:
            return flash_mla.decode(**kw)
        return m.run(q,kw["k_cache"],kw["indices_in_kvcache"],kw["topk_length"],kw["attn_sink"],
                     kw["extra_k_cache"],kw["extra_indices_in_kvcache"],kw["extra_topk_length"],
                     kw["token_positions"],kw["cos_sin_cache"],kw["sm_scale"])

    def block(self, size, slots, cap, rows, top, fmt, *, uniform):
        m = self.pick("megamoe",rows,"uniform-ep4") if uniform and (size,slots,top,fmt) == (4,416,6,"fp8xfp4") else None
        return (m.native if m else deep_gemm).get_block_m_for_mega_moe(size,slots,cap,rows,top,fmt)

    def moe(self,y,l1,l2,buf,*,uniform,**kw):
        m = self.pick("megamoe",y.shape[0],"uniform-ep4") if uniform and (buf.group.size(),buf.num_experts,buf.num_topk)==(4,416,6) else None
        if m is None:
            return deep_gemm.fp8_fp4_mega_moe(y,l1,l2,buf,**kw)
        return m.native.fp8_fp4_mega_moe(y,l1,l2,kw["shared_l1_weights"],kw["shared_l2_weights"],
            kw["cumulative_local_expert_recv_stats"],buf.buffer,buf.handle.buffer_ptrs,buf.group.rank(),
            buf.num_max_tokens_per_rank,buf.num_experts,buf.num_topk,(1,1,32),kw["activation"],kw["activation_clamp"],kw["fast_math"])
