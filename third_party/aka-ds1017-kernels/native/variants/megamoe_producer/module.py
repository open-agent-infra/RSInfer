"""Editable source-built MegaMoE entry; no installed vendor operator imports."""
from .source_build import build


def __getattr__(name):
    if name in {'SymmBuffer','transform_weights_for_mega_moe','get_symm_buffer_for_mega_moe'}:
        from . import host
        return getattr(host,name)
    if name.startswith('__'):raise AttributeError(name)
    return getattr(build(),name)


def run(y,l1_weights,l2_weights,buffer,shared_l1_weights,shared_l2_weights,stats):
    import os
    os.environ.pop('AKA_MEGAMOE_COMPILE_ONLY',None)
    from . import host
    return host.fp8_fp4_mega_moe(y,l1_weights,l2_weights,buffer,
        shared_l1_weights=shared_l1_weights,shared_l2_weights=shared_l2_weights,
        cumulative_local_expert_recv_stats=stats,activation='swiglu',activation_clamp=10.,fast_math=True)


def prepare(y,l1_weights,l2_weights,buffer,shared_l1_weights,shared_l2_weights,stats):
    import os
    from . import host
    previous=os.environ.get('AKA_MEGAMOE_COMPILE_ONLY')
    os.environ['AKA_MEGAMOE_COMPILE_ONLY']='1'
    try:
        from contextlib import nullcontext
        if os.environ.get('AKA_BUILD_CACHE_DIR'):
            from .aka_cache import jit_compile
            fence=jit_compile()
        else:fence=nullcontext()
        with fence:
            host.fp8_fp4_mega_moe(y,l1_weights,l2_weights,buffer,shared_l1_weights=shared_l1_weights,
                shared_l2_weights=shared_l2_weights,cumulative_local_expert_recv_stats=stats,
                activation='swiglu',activation_clamp=10.,fast_math=True)
    finally:
        if previous is None:os.environ.pop('AKA_MEGAMOE_COMPILE_ONLY',None)
        else:os.environ['AKA_MEGAMOE_COMPILE_ONLY']=previous


def fp8_fp4_mega_moe(y,l1_weights,l2_weights,sym_buffer,shared_l1_weights=None,shared_l2_weights=None,
                    cumulative_local_expert_recv_stats=None,**kwargs):
    from . import host
    return host.fp8_fp4_mega_moe(y,l1_weights,l2_weights,sym_buffer,
        shared_l1_weights=shared_l1_weights,shared_l2_weights=shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,**kwargs)
