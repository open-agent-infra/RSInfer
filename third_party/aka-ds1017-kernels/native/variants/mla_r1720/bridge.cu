#include <cuda_runtime.h>
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh"
extern "C" int launch_prefill(void* q, void* kv, void* indices, void* sink,
    void* lengths, void* out, void* maximum, void* lse, void* positions,
    void* rope, void* scales, int rows, int kv_rows, int topk,
    int q_stride, int kv_stride, int index_stride, int scale_rows,
    double scale, void* stream) {
    using namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
    ParamT<SparseAttnFwdMode::Prefill> p{};
    p.s_q=rows;p.s_kv=kv_rows;p.h_q=64;p.h_kv=1;p.d_qk=512;p.d_v=512;p.topk=topk;
    p.sm_scale=scale;p.sm_scale_div_log2=float(double(scale)*1.4426950408889634);
    p.q=(bf16*)q;p.kv=(bf16*)kv;p.indices=(int*)indices;
    p.attn_sink=(float*)sink;p.topk_length=(int*)lengths;
    p.stride_q_s_q=q_stride;p.stride_q_h_q=512;
    p.stride_kv_s_kv=kv_stride;p.stride_kv_h_kv=512;
    p.stride_indices_s_q=index_stride;p.stride_indices_h_kv=topk;
    p.out=nullptr;p.max_logits=(float*)maximum;p.lse=(float*)lse;
    p.mode=SparseAttnFwdMode::Prefill;
    int device;cudaGetDevice(&device);cudaDeviceGetAttribute(&p.num_sm,cudaDevAttrMultiProcessorCount,device);
    p.stream=(cudaStream_t)stream;p.enable_q_norm=false;p.rms_norm_eps=0;
    p.token_positions=(uint32_t*)positions;p.is_rope_neox_style=false;p.rope_dim=64;
    p.cos_sin_cache=(float*)rope;p.n_wv_group=8;p.wv_group_size=8;p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true;p.round_sf=true;p.use_packed_ue8m0=true;
    p.out_fp8=(fp8_e4m3*)out;p.out_sf=(uint32_t*)scales;
    p.stride_out_sf_head_dim=scale_rows;p.stride_out_sf_wv_group=scale_rows*32;
    // Short jobs (<= 2 KV blocks of 64) have no long softmax tail for the Q-load burst to stall, so they
    // benefit from issuing the whole Q token as one burst (GROUP_Q_LOADS_CFG=false). Longer jobs keep the
    // tiled, dependency-chained Q load that protects the softmax warpgroup's shared-memory stores.
    if (topk <= 128) {
        run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Prefill,ModelType::V41,ModelType::V41,64,false,false}>(p);
    } else {
        run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Prefill,ModelType::V41,ModelType::V41,64,false,true}>(p);
    }
    return int(cudaGetLastError());
}
