"""Production projection GEMM cores, compiled from the supplied DeepGEMM source."""
import torch
from functools import lru_cache
from .source_build import build as build_main
from .official_qkv.source_build import build as build_official_qkv

@lru_cache(maxsize=1)
def build():
    # Initialize each JIT while its independently keyed cache is active.
    build_official_qkv().warm_jit()
    main = build_main()
    main.get_jit()
    return main

def run(kind, a, b, n):
    # DS1017 supplies packed UE8M0 scales for both prequantized operands.
    if a[1].dtype != torch.int32 or b[1].dtype != torch.int32:
        raise TypeError("DS1017 projection requires packed int32 UE8M0 scales")
    dg = build()
    m = a[0].shape[0]
    if kind == 'wo_a':
        out = torch.empty((m, 8, n), dtype=torch.float8_e4m3fn, device=a[0].device)
        sf = torch.empty_strided((m, 8 * n // 128), (1, (m + 3) // 4 * 4),
                                dtype=torch.int32, device=a[0].device)
        dg.fp8_einsum('bhr,hdr->bhd', a, b, (out, sf), recipe=(1, 1, 32))
        return out, sf
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a[0].device)
    if kind == 'qkv' and m in (32768, 65536) and n == 1792 and a[0].shape[-1] == 5120:
        build_official_qkv().run(a, b, out)
    else:
        dg.fp8_fp4_gemm_nt(a, b, out, None, recipe=(1, 1, 32))
    return out
