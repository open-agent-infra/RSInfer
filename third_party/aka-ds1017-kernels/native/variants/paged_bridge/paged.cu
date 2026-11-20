#include <cuda_runtime.h>
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh"
extern "C" int paged(void** t,const int* v,double scale,void* stream) {
    using namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
    int sms,dev;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
    auto mt=v[13]==288?ModelType::V41_FP4:ModelType::V41;
    SparseAttnDecodeParams base{1,v[0],64,1,512,512,(float)scale,(float)(scale*1.4426950408889634),
       v[1],v[2],v[3],ModelType::V41,mt,(bf16*)t[0],(bf16*)t[1],(int*)t[2],(int*)t[3],(float*)t[4],
       (float*)t[5],nullptr,v[4],v[5],v[6],(bf16*)t[6],(int*)t[7],(int*)t[8],0,v[7],512,
       v[8],v[9],0,v[10],0,64,0,0,0,v[11],v[12],0,v[14],sms,(cudaStream_t)stream,false,nullptr,nullptr,0,0,0,0,0,nullptr,nullptr,0};
    ParamT<SparseAttnFwdMode::Decode> p{base,false,0,(uint32_t*)t[9],false,64,(float*)t[10],8,8,32,true,true,true,
       (fp8_e4m3*)t[11],(uint32_t*)t[12],(uint32_t)(v[15]*32),(uint32_t)v[15]};
    if(mt==ModelType::V41_FP4)
        run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode,ModelType::V41,ModelType::V41_FP4,64,false,true}>(p);
    else
        run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode,ModelType::V41,ModelType::V41,64,false,true}>(p);
    return (int)cudaGetLastError();
}
