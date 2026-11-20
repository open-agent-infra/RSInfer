"""Complete fused prefill ABI over source-built SM100 CUDA kernels."""
import ctypes,hashlib,os,shutil,subprocess
from pathlib import Path
import torch
_library=None

def build():
    global _library
    if _library is not None:return _library
    root=Path(__file__).resolve().parent
    from torch.utils.cpp_extension import CUDA_HOME
    from .aka_nvcc_cache import load
    cuda=Path(CUDA_HOME)
    files=[root/'module.py',root/'aka_nvcc_cache.py',*sorted(root.rglob('*.cu')),*sorted((root/'include').rglob('*'))]
    command=[str(cuda/'bin/nvcc'),'-Rbridge.cu','-o','{output}','-shared','-Xcompiler=-fPIC','-O3','-std=c++20','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--ptxas-options=-v,--register-usage-level=10','-DFLASH_MLA_IS_BUILD_ON_CUDA','-DKERUTILS_IS_BUILD_ON_CUDA','-DNDEBUG','-gencode=arch=compute_100a,code=sm_100a','-IRinclude','-I'+str(cuda/'include/cccl'),'-lcuda']
    _library=load(root,files,command)
    _library.launch_prefill.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int]*7+[ctypes.c_double,ctypes.c_void_p]
    _library.launch_prefill.restype=ctypes.c_int
    return _library

def prefill(enable_q_norm,rms_norm_eps,token_positions,is_rope_neox_style,rope_dim,cos_sin_cache,n_wv_group,num_per_channels,use_tma_aligned_col_major_sf,round_sf,use_packed_ue8m0,q,kv,indices,sm_scale,d_v=512,attn_sink=None,topk_length=None):
    assert not enable_q_norm and not is_rope_neox_style and rope_dim==64
    assert n_wv_group==8 and num_per_channels==32 and d_v==512
    assert use_tma_aligned_col_major_sf and round_sf and use_packed_ue8m0
    assert q.shape[1:]==(64,512) and kv.shape[1:]==(1,512)
    with torch.cuda.device(q.device):
        lib=build();n=len(q);aligned=(n+3)//4*4
        out=torch.empty((n,8,4096),device=q.device,dtype=torch.float8_e4m3fn)
        scales=torch.empty((8,32,aligned),device=q.device,dtype=torch.int32).transpose(1,2)[:,:n].transpose(0,1)
        maximum=torch.empty((n,64),device=q.device,dtype=torch.float32);lse=torch.empty_like(maximum)
        if n:
            ptr=lambda t:0 if t is None else t.data_ptr()
            tensors=[q,kv,indices,attn_sink,topk_length,out,maximum,lse,token_positions,cos_sin_cache,scales]
            code=lib.launch_prefill(*map(ptr,tensors),n,len(kv),indices.shape[-1],q.stride(0),kv.stride(0),indices.stride(0),aligned,sm_scale,torch.cuda.current_stream().cuda_stream)
            if code:raise RuntimeError(f'CUDA launch failed: {code}')
        return out,scales,maximum,lse

run = prefill
