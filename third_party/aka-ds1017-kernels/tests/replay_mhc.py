"""Run directly on a B200 after installing the wheel and official DeepInfer dependencies."""
import torch
import deep_gemm
from aka_ds1017 import create


def outputs(rows, layout):
    device = "cuda"
    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)
    out = dict(new_residual=empty(rows,4,5120), new_prev_mix=empty(rows,4,1,dtype=torch.float32),
               new_post_mix=empty(rows,4,1,dtype=torch.float32), new_comb_res_mix=empty(rows,4,4,dtype=torch.float32),
               y_bf16=empty(rows,5120), y_fp8=empty(rows,5120,dtype=torch.float8_e4m3fn))
    if layout == "attn":
        out["y_gemm_sf"] = empty(40,rows,dtype=torch.int32).T
    else:
        out["y_routed_sf"] = empty(rows,40,dtype=torch.int32)
        aligned = (rows+127)//128*128
        out["y_shared_sf"] = empty(40,aligned,dtype=torch.int32).T[:rows]
    return out


def check(rows,layout):
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape,device="cuda",dtype=dtype)
    kw=dict(fn=rand(24,20480,dtype=torch.float32)*0.001, mix_scales=rand(3,dtype=torch.float32),
            mix_bases=rand(24,dtype=torch.float32), hc_mult=4,hc_norm_eps=1e-20,hc_pre_eps=1e-6,hc_post_scale=2.,
            sinkhorn_eps=1e-6,num_sinkhorn_iters=20,rmsnorm_weight=rand(5120),rmsnorm_eps=1e-20,
            rmsnorm_scale=1.,shared_sf_block_m=128 if layout=="moe" else 0)
    call=create("validated",rows//140*4096).mhc
    capture=torch.cuda.Stream()
    lanes=[]
    for _ in range(2):
        args=(rand(rows,5120),rand(rows,4,5120),rand(rows,4,1,dtype=torch.float32),
              rand(rows,4,1,dtype=torch.float32),rand(rows,4,4,dtype=torch.float32))
        candidate=outputs(rows,layout)
        capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture):
            call(*args,**kw,**candidate)
        capture.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=capture):
            call(*args,**kw,**candidate)
        lanes.append((args,candidate,outputs(rows,layout),graph,torch.cuda.Stream()))
    logical=torch.arange(rows,device="cuda")
    physical=logical//128*128+(logical%32)*4+(logical%128)//32
    for _ in range(64):
        for args,candidate,reference,graph,stream in lanes:
            args[0].neg_()
            args[1].mul_(0.99)
            deep_gemm.mega_mhc(*args,**kw,**reference)
        for args,candidate,reference,graph,stream in lanes:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for args,candidate,reference,graph,stream in lanes:
            torch.cuda.current_stream().wait_stream(stream)
            for name,a in candidate.items():
                b=reference[name]
                if name=="y_shared_sf":
                    # The swizzle can place live rows beyond the logical last row.
                    a=a.as_strided((a.stride(1),40),a.stride())[physical]
                    b=b.as_strided((b.stride(1),40),b.stride())[physical]
                assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),(rows,layout,name)
    print("PASS", rows,layout,"128 replays",flush=True)


if __name__=="__main__":
    torch.manual_seed(1017)
    for rows in (1120,2240):
        for layout in ("attn","moe"):
            check(rows,layout)
