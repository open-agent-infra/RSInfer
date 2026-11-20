"""Bind a startup-selected set of kernels to a single DeepInfer unit instance."""
from __future__ import annotations

from importlib import import_module
from typing import Any

import deep_gemm  # pyright: ignore[reportMissingImports]
from flash_mla import fused_norm_rope_attn_rope_cast  # pyright: ignore[reportMissingImports]


class Official:
    mhc = staticmethod(deep_gemm.mega_mhc)
    gemm = staticmethod(deep_gemm.fp8_fp4_gemm_nt)
    einsum = staticmethod(deep_gemm.fp8_einsum)
    prefill = staticmethod(fused_norm_rope_attn_rope_cast.prefill)
    paged = staticmethod(fused_norm_rope_attn_rope_cast.decode)

    @staticmethod
    def block(*args: Any, uniform: bool = False) -> Any:
        return deep_gemm.get_block_m_for_mega_moe(*args)

    @staticmethod
    def moe(*args: Any, uniform: bool = False, **kwargs: Any) -> Any:
        return deep_gemm.fp8_fp4_mega_moe(*args, **kwargs)


def configure(profile: str, budget: int, disabled: str) -> Any:
    if profile == "official":
        return Official
    if profile not in ("validated", "expanded"):
        raise ValueError(f"Unknown DeepInfer AKA profile {profile!r}")
    package = import_module("aka_ds1017")
    if package.__version__ != "0.1.0":
        raise RuntimeError(f"DeepInfer requires aka-ds1017-kernels 0.1.0, found {package.__version__}")
    return package.create(profile, budget, disabled)
