"""Direct-output Indexer path with per-step metadata reuse."""
import torch

def index(m,q,kv,w,ks,ke,klen,offset,width,mode,candidates,out,picked,cache):
    n,topk=out.shape
    consumer=n<4096
    dg=m.native
    chunk=max(8,min(8192,(2<<30)//(2*(16384 if mode=='user' else width))//8*8))
    if not consumer and mode=='dense' and picked is None:
        chunk=min(n,32768)
    for a in range(0,n,chunk):
        b=min(a+chunk,n); c=slice(a,b); qc=(q[0][c],q[1][c])
        off=None if offset is None else offset[c]
        if mode=='user':
            if consumer:
                key=('r1720-user',candidates.data_ptr(),a,b)
                entry=cache.get(key)
                if entry is None:
                    cand=candidates[c].to(torch.int32).contiguous()
                    valid=torch.empty_like(ks[c])
                    entry=cache[key]=(cand,valid,candidates)
                cand,valid,_=entry
                logits=dg.fp8_fp4_sparse_mqa_logits(qc,kv,w[c],None,cand.shape[1],8,use_unaligned_ks=True,candidates=cand,row_bounds=(ks[c],ke[c],valid))
                m._call('select_candidates8',logits,valid,cand,off,out[c],b-a,logits.shape[1],logits.stride(0),topk,out.stride(0))
                del logits
                continue
            key=('ab-user',ks.data_ptr(),ke.data_ptr(),candidates.data_ptr(),len(kv[0]),a,b)
            entry=cache.get(key)
            if entry is None:
                cand=candidates[c].to(torch.int32).contiguous()
                valid=torch.empty_like(ks[c])
                options={'headers_only':True} if consumer else {}
                meta=dg.get_sparse_mqa_logits_metadata(ks[c],ke[c],len(kv[0]),cand,qc[0].dtype,
                    8,use_unaligned_ks=True,candidate_valid=valid,**options)
                entry=cache[key]=(cand,valid,meta,candidates)
            cand,valid,meta,_=entry
            options={'candidates':cand} if consumer else {}
            logits=dg.fp8_fp4_sparse_mqa_logits(qc,kv,w[c],meta,cand.shape[1],8,use_unaligned_ks=True,**options)
            args=(logits,valid,cand,off,out[c],b-a,logits.shape[1],logits.stride(0),topk,out.stride(0))
            if consumer:m._call('select_candidates8',*args)
            else:m._call('topk_restore8',*args,cand.stride(0))
        else:
            key=('ab-dense',ks.data_ptr(),ke.data_ptr(),len(kv[0]),q[0].shape[1],a,b)
            if key not in cache:cache[key]=dg.get_mqa_logits_metadata(ks[c],ke[c],len(kv[0]),q[0].shape[1])
            logits=dg.fp8_fp4_mqa_logits(qc,kv,w[c],ks[c],ke[c],max_seqlen_k=width,schedule_meta=cache[key])
            if picked is None and not consumer:
                from deep_select import interface as select
                select.topk(logits,topk,sorted=False,end=klen[c],indices_type=torch.int32,sorted_index=True,
                            output_idx=out[c],output_idx_offset=off,idx_oob_fill_value=-1,return_value=False)
            elif picked is None:m._top(logits,topk,klen[c],off,out[c])
            else:
                stride=(logits.shape[1]+4095)//4096*512
                scores=torch.empty((b-a,stride),device=out.device,dtype=torch.bfloat16)
                valid=torch.empty_like(ks[c])
                m._top_bs(logits,topk,klen[c],off,scores,valid,stride,out[c])
                m._top(scores,picked.shape[1],valid,out=picked[c])
        del logits
    return out,picked
