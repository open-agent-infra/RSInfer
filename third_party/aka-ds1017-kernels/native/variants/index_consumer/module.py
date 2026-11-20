"""Extracted production indexer: source-built DeepGEMM logits + DeepSelect + candidate helpers."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import torch
from .source_build import build as build_gemm

_select = None

def build():
    global _select
    dg = build_gemm()
    if _select is None:
        root = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for p in [root/'select_bridge.cu', *sorted((root/'select').rglob('*'))]:
            if p.is_file():digest.update(p.read_bytes())
        cache = root/'.aka-build'/('select-'+digest.hexdigest()[:16]);cache.mkdir(parents=True,exist_ok=True)
        library=cache/'select.so'
        if not library.exists():
            from torch.utils.cpp_extension import CUDA_HOME
            subprocess.run([str(Path(CUDA_HOME)/'bin/nvcc'),str(root/'select_bridge.cu'),'-shared','-Xcompiler=-fPIC',
                '-O3','-std=c++20','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','--ftz=false',
                '-lineinfo','-DDEEP_SELECT_IS_BUILD_ON_CUDA','-DKERUTILS_IS_BUILD_ON_CUDA',
                '-gencode=arch=compute_100a,code=sm_100a','-I'+str(root),'-I'+str(root/'select'),
                '-I'+str(root/'select_include'),'-I'+str(root/'include'),
                '-I'+str(Path(CUDA_HOME)/'include/cccl'),'-lcuda','-o',str(library)],check=True)
        _select=ctypes.CDLL(str(library))
        P,I=ctypes.c_void_p,ctypes.c_int
        for name,args in [('agx_ruibo_topk_i32',[P]*4+[I]*5+[P]),('agx_ruibo_topk_bs_i32',[P]*6+[I]*6+[P]),('launch_topk',[P]*5+[I]*4+[P]),('candidate_scores',[P]*4+[I]*5+[P]),
                          ('candidate_lengths',[P]*3+[I]*3+[P]),('restore_indices_i32',[P]*3+[I]*4+[P]),('restore_indices',[P]*3+[I]*4+[P]),
                          ('launch_topk_bs',[P]*7+[I]*5+[P]),('select_candidates8',[P]*5+[I]*5+[P])]:
            fn=getattr(_select,name);fn.argtypes=args;fn.restype=I
    return dg

def _call(name, *args):
    values=[x.data_ptr() if isinstance(x,torch.Tensor) else 0 if x is None else x for x in args]
    code=getattr(_select,name)(*values,torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError((name,code))

def _top(scores,k,end,offset=None,out=None):
    if out is None:
        out=torch.empty((len(scores),k),device=scores.device,dtype=torch.int32)
    _call('agx_ruibo_topk_i32',scores,end,offset,out,len(scores),scores.shape[1],scores.stride(0),k,out.stride(0))
    return out

def _top_bs(scores,k,end,offset,bs,bs_valid,bs_stride,out):
    _call('agx_ruibo_topk_bs_i32',scores,end,offset,out,bs,bs_valid,len(scores),scores.shape[1],scores.stride(0),k,out.stride(0),bs_stride)
    return out

def run(q,kv,weights,ks,ke,longest,mode,candidates,topk=512,topk_blocks=2048,block=8):
    dg=build(); n=len(q[0]); result=torch.empty(n,topk,device='cuda',dtype=torch.int32)
    picked=torch.empty(n,topk_blocks,device='cuda',dtype=torch.int32) if mode=='source' else None
    chunk=max(8,min(8192,(2<<30)//(max(longest,1)*2)//8*8))
    for start in range(0,n,chunk):
        c=slice(start,min(start+chunk,n)); qc=(q[0][c],q[1][c])
        if mode=='user':
            cand=candidates[c].to(torch.int32).contiguous()
            valid=torch.empty_like(ks[c])
            meta=dg.get_sparse_mqa_logits_metadata(ks[c],ke[c],len(kv[0]),cand,
                                                  qc[0].dtype,block,use_unaligned_ks=True,candidate_valid=valid,headers_only=True)
            logits=dg.fp8_fp4_sparse_mqa_logits(qc,kv,weights[c],meta,cand.shape[1],block,use_unaligned_ks=True,candidates=cand)
            if block == 8 and logits.shape[1] <= 16384 and topk == 512:
                _call('select_candidates8',logits,valid,cand,ks[c],result[c],len(logits),logits.shape[1],logits.stride(0),topk)
            else:
                _top(logits,topk,valid,out=result[c])
                _call('restore_indices_i32',result[c],cand,ks[c],result[c].shape[0],topk,cand.shape[1],block)
        else:
            # Balanced static schedule (cost-weighted KV splits per SM); PDL overlaps the metadata
            # tail with the logits kernel prologue.
            smeta=dg.get_mqa_logits_metadata(ks[c],ke[c],len(kv[0]),q[0].shape[1])
            logits=dg.fp8_fp4_mqa_logits(qc,kv,weights[c],ks[c],ke[c],max_seqlen_k=longest,schedule_meta=smeta)
            if picked is not None:
                cols=(logits.shape[1]+block-1)//block; stride=(cols+511)//512*512
                scores=torch.empty((len(logits),stride),device='cuda',dtype=torch.bfloat16)
                valid=torch.empty_like(ks[c])
                # Fused: token TopK also produces the candidate block scores, then block TopK runs.
                _top_bs(logits,topk,ke[c],ks[c],scores,valid,stride,result[c])
                _top(scores,topk_blocks,valid,out=picked[c])
            else:
                _top(logits,topk,ke[c],ks[c],out=result[c])
    return result,picked
