// Thin C ABI for the unmodified DeepSelect bfa4507 BF16 kernels.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "structs.h"
#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.cuh"

extern "C" int launch_topk(void* input, void* end, void* offset, void* output,
    void* begin, int rows, int width, int stride, int topk, int output_stride, void* stream) {
    int device; cudaGetDevice(&device);
    cudaDeviceProp props{}; cudaGetDeviceProperties(&props, device);
    TopkSelectArgs args{};
    args.batch_size=rows; args.vocab_size=width; args.topk=topk;
    args.input=input; args.output_index=output; args.end_ptr=(int*)end;
    args.begin_ptr=(int*)begin;
    args.output_idx_offset=(int*)offset; args.stride_input_batch=stride;
    args.stride_output_index_batch=topk; args.sorted_index=true;
    args.return_value=false; args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY; args.abort_when_nan_found=true;
    args.shared_memory_size_per_sm=props.sharedMemPerBlockOptin; args.stream=(cudaStream_t)stream;
    if(topk > 1024) {
        // Exact-size survivor/sort buffers for k=2048 block TopK. occ2 small-smem config wins the
        // cold-L2 sweep (162.4us vs 252.5us occ1 vs 271.0us original); its window-delta table is
        // sized for vocab <= 65536, so wider inputs fall back to the occ1 config.
        if(width <= 8192) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,2048,256,3,4096,2048,2,512,1,false,8192>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,2048,256,2,4096,2048,2,512,1,false,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,2048,512,1,8192,4096,2>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    } else if(rows <= props.multiProcessorCount) {
        using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,512,1,8192,4096,5>;
        topk_select_bf16_normal::run_topk_select_kernel<C>(args);
    } else {
        // occ3 (regs 96->80, tiny spill) hides the 2-barrier/round + TMA latency of the streaming
        // scan: 6 warps/scheduler vs 4. Sweep on real source logits: 367us vs 443us occ2 (-17%).
        // max_vocab=65536 sizes the index-sort window-delta table for the workload widths; wider
        // inputs fall back to the occ2 config (default 2^23 table).
        if(width <= 8192) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,4,4096,2048,2,512,1,false,8192>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 16384) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,3,4096,2048,2,512,1,false,16384>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,3,4096,2048,2,512,1,false,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,2,4096,4096,4>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    }
    return int(cudaGetLastError());
}

// Token TopK with fused candidate block-max side outputs (source mode). Writes int64 indices and
// produces the BF16 block scores + valid block counts that the standalone block_scores kernel made.
extern "C" int launch_topk_bs(void* input, void* end, void* offset, void* output,
    void* block_scores, void* block_scores_valid, void* begin,
    int rows, int width, int stride, int topk, int block_scores_stride, void* stream) {
    int device; cudaGetDevice(&device);
    cudaDeviceProp props{}; cudaGetDeviceProperties(&props, device);
    TopkSelectArgs args{};
    args.batch_size=rows; args.vocab_size=width; args.topk=topk;
    args.input=input; args.output_index=output; args.end_ptr=(int*)end;
    args.begin_ptr=(int*)begin;
    args.output_idx_offset=(int*)offset; args.stride_input_batch=stride;
    args.stride_output_index_batch=topk; args.sorted_index=true;
    args.return_value=false; args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY; args.abort_when_nan_found=true;
    args.shared_memory_size_per_sm=props.sharedMemPerBlockOptin; args.stream=(cudaStream_t)stream;
    args.block_scores=block_scores; args.block_scores_valid=(int*)block_scores_valid;
    args.block_scores_stride=block_scores_stride; args.block_size=8;
    if(rows <= props.multiProcessorCount) {
        using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,512,1,8192,4096,5,512,1,true>;
        topk_select_bf16_normal::run_topk_select_kernel<C>(args);
    } else {
        if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,3,4096,2048,2,512,1,true,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int64_t,false,true,false,512,256,2,4096,4096,4,512,1,true>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    }
    return int(cudaGetLastError());
}

// CUDA translations of DeepInfer a679bb8's three candidate-block helpers.
__global__ void block_scores(const nv_bfloat16* x,const int* len,nv_bfloat16* out,int* valid,
                            int width,int stride,int out_stride,int block) {
    int row=blockIdx.x, n=(len[row]+block-1)/block;
    if(threadIdx.x==0)valid[row]=n;
    for(int b=threadIdx.x;b<(width+block-1)/block;b+=blockDim.x) {
        float value=-INFINITY;
        for(int j=0;j<block;++j)if(b*block+j<len[row])
            value=fmaxf(value,__bfloat162float(x[row*stride+b*block+j]));
        value=fminf(value,3.3895313892515355e38f);
        if(b==n-1)value=INFINITY;
        out[row*out_stride+b]=__float2bfloat16(value);
    }
}
__global__ void compact_lens(const int* cand,const int* len,int* out,int cols,int block) {
    int row=blockIdx.x, sum=0, l=len[row];
    const int* crow=cand+(size_t)row*cols;
    if(cols%4==0) {
        const int4* c4=reinterpret_cast<const int4*>(crow);
        for(int j=threadIdx.x;j<cols/4;j+=blockDim.x) {
            int4 v=c4[j];
            if(v.x>=0)sum+=max(0,min(block,l-v.x*block));
            if(v.y>=0)sum+=max(0,min(block,l-v.y*block));
            if(v.z>=0)sum+=max(0,min(block,l-v.z*block));
            if(v.w>=0)sum+=max(0,min(block,l-v.w*block));
        }
    } else {
        for(int j=threadIdx.x;j<cols;j+=blockDim.x) {
            int b=crow[j];if(b>=0)sum+=max(0,min(block,l-b*block));
        }
    }
    __shared__ int partial[256];partial[threadIdx.x]=sum;__syncthreads();
    for(int s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)partial[threadIdx.x]+=partial[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)out[row]=partial[0];
}
__device__ __forceinline__ int64_t restore_one(int64_t c,const int* crow,const int* offset,int row,int ccols,int block) {
    int c32=(int)c;
    if(c32<0 || c32/block>=ccols)return -1;
    int b=crow[c32/block];
    return b<0?-1:(int64_t)b*block+c32%block+(offset?offset[row]:0);
}
__global__ void restore(int64_t* idx,const int* cand,const int* offset,int cols,int ccols,int block) {
    int row=blockIdx.x;
    const int* crow=cand+(size_t)row*ccols;
    int64_t* irow=idx+(size_t)row*cols;
    if(cols%2==0) {
        for(int j=threadIdx.x*2;j<cols;j+=blockDim.x*2) {
            longlong2 v=*reinterpret_cast<const longlong2*>(irow+j);
            v.x=restore_one(v.x,crow,offset,row,ccols,block);
            v.y=restore_one(v.y,crow,offset,row,ccols,block);
            *reinterpret_cast<longlong2*>(irow+j)=v;
        }
    } else {
        for(int j=threadIdx.x;j<cols;j+=blockDim.x)
            irow[j]=restore_one(irow[j],crow,offset,row,ccols,block);
    }
}
extern "C" int candidate_scores(void* x,void* len,void* out,void* valid,int m,int width,int stride,int os,int b,void* s){
    block_scores<<<m,256,0,(cudaStream_t)s>>>((nv_bfloat16*)x,(int*)len,(nv_bfloat16*)out,(int*)valid,width,stride,os,b);
    return int(cudaGetLastError());
}
extern "C" int candidate_lengths(void* c,void* len,void* out,int m,int cols,int b,void* s){
    compact_lens<<<m,256,0,(cudaStream_t)s>>>((int*)c,(int*)len,(int*)out,cols,b);return int(cudaGetLastError());
}
extern "C" int restore_indices(void* idx,void* cand,void* offset,int m,int cols,int ccols,int b,void* s){
    restore<<<m,256,0,(cudaStream_t)s>>>((int64_t*)idx,(int*)cand,(int*)offset,cols,ccols,b);return int(cudaGetLastError());
}

// Thin C ABI for the unmodified DeepSelect bfa4507 BF16 kernels.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "structs.h"
#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.cuh"

extern "C" int agx_ruibo_topk_i32(void* input, void* end, void* offset, void* output,
    int rows, int width, int stride, int topk, int output_stride, void* stream) {
    int device; cudaGetDevice(&device);
    cudaDeviceProp props{}; cudaGetDeviceProperties(&props, device);
    TopkSelectArgs args{};
    args.batch_size=rows; args.vocab_size=width; args.topk=topk;
    args.input=input; args.output_index=output; args.end_ptr=(int*)end;
    args.begin_ptr=nullptr;
    args.output_idx_offset=(int*)offset; args.stride_input_batch=stride;
    args.stride_output_index_batch=output_stride; args.sorted_index=true;
    args.return_value=false; args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY; args.abort_when_nan_found=true;
    args.shared_memory_size_per_sm=props.sharedMemPerBlockOptin; args.stream=(cudaStream_t)stream;
    if(topk > 1024) {
        // Exact-size survivor/sort buffers for k=2048 block TopK. occ2 small-smem config wins the
        // cold-L2 sweep (162.4us vs 252.5us occ1 vs 271.0us original); its window-delta table is
        // sized for vocab <= 65536, so wider inputs fall back to the occ1 config.
        if(width <= 8192) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,2048,256,3,4096,2048,2,512,1,false,8192>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,2048,256,2,4096,2048,2,512,1,false,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,2048,512,1,8192,4096,2>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    } else if(rows <= props.multiProcessorCount) {
        using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,512,1,8192,4096,5>;
        topk_select_bf16_normal::run_topk_select_kernel<C>(args);
    } else {
        // occ3 (regs 96->80, tiny spill) hides the 2-barrier/round + TMA latency of the streaming
        // scan: 6 warps/scheduler vs 4. Sweep on real source logits: 367us vs 443us occ2 (-17%).
        // max_vocab=65536 sizes the index-sort window-delta table for the workload widths; wider
        // inputs fall back to the occ2 config (default 2^23 table).
        if(width <= 8192) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,4,4096,2048,2,512,1,false,8192>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 16384) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,3,4096,2048,2,512,1,false,16384>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,3,4096,2048,2,512,1,false,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,2,4096,4096,4>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    }
    return int(cudaGetLastError());
}

// Token TopK with fused candidate block-max side outputs (source mode). Writes int64 indices and
// produces the BF16 block scores + valid block counts that the standalone block_scores kernel made.
extern "C" int agx_ruibo_topk_bs_i32(void* input, void* end, void* offset, void* output,
    void* block_scores, void* block_scores_valid,
    int rows, int width, int stride, int topk, int output_stride, int block_scores_stride, void* stream) {
    int device; cudaGetDevice(&device);
    cudaDeviceProp props{}; cudaGetDeviceProperties(&props, device);
    TopkSelectArgs args{};
    args.batch_size=rows; args.vocab_size=width; args.topk=topk;
    args.input=input; args.output_index=output; args.end_ptr=(int*)end;
    args.begin_ptr=nullptr;
    args.output_idx_offset=(int*)offset; args.stride_input_batch=stride;
    args.stride_output_index_batch=output_stride; args.sorted_index=true;
    args.return_value=false; args.idx_oob_fill_value=-1;
    args.value_oob_fill_value=-INFINITY; args.abort_when_nan_found=true;
    args.shared_memory_size_per_sm=props.sharedMemPerBlockOptin; args.stream=(cudaStream_t)stream;
    args.block_scores=block_scores; args.block_scores_valid=(int*)block_scores_valid;
    args.block_scores_stride=block_scores_stride; args.block_size=8;
    if(rows <= props.multiProcessorCount) {
        using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,512,1,8192,4096,5,512,1,true>;
        topk_select_bf16_normal::run_topk_select_kernel<C>(args);
    } else {
        if(width <= 65536) {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,3,4096,2048,2,512,1,true,65536>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        } else {
            using C=TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,256,2,4096,4096,4,512,1,true>;
            topk_select_bf16_normal::run_topk_select_kernel<C>(args);
        }
    }
    return int(cudaGetLastError());
}


__global__ void restore_i32(int* idx,const int* cand,const int* offset,int cols,int ccols,int block) {
    int row=blockIdx.x;
    for(int j=threadIdx.x;j<cols;j+=blockDim.x) {
        int i=row*cols+j,c=idx[i];
        if(c<0 || c/block>=ccols){idx[i]=-1;continue;}
        int b=cand[row*ccols+c/block];idx[i]=b<0?-1:b*block+c%block+(offset?offset[row]:0);
    }
}
extern "C" int restore_indices_i32(void* idx,void* cand,void* offset,int m,int cols,int ccols,int b,void* s){
    restore_i32<<<m,256,0,(cudaStream_t)s>>>((int*)idx,(int*)cand,(int*)offset,cols,ccols,b);return int(cudaGetLastError());
}

struct CandidateSelect8 : TopkSelectConfig<nv_bfloat16,int32_t,false,true,false,512,512,2,8192,2048,2,512,1,false,16384> {
    static constexpr bool restore_blocks = true;
};

using WindowRadix = topk_select_common::TopkSelectKernelBF16Base<CandidateSelect8>;

__global__ __launch_bounds__(CandidateSelect8::num_threads, CandidateSelect8::target_occupancy)
void select_register_window(__grid_constant__ const TopkSelectArgs args) {
    constexpr uint32_t span = CandidateSelect8::max_vocab / CandidateSelect8::num_threads;
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint32_t length = args.end_ptr[row];
    const uint32_t topk = args.topk;
    const int offset = (args.output_idx_offset ? args.output_idx_offset[row] : 0);
    auto result = static_cast<int*>(args.output_index) + size_t(row) * args.stride_output_index_batch;
    if (length <= topk) {
        if (tid < topk) result[tid] = tid < length ? WindowRadix::EpilogueT::output_index(args, row, tid, offset) : -1;
        return;
    }

    struct Storage {
        alignas(16) uint32_t reconstruct_bucket_counter[2][WindowRadix::NUM_RECONSTRUCT_BUCKET_SLOTS];
        uint32_t reconstruct_pivot_bucket, reconstruct_num_should_select;
        uint32_t selected[CandidateSelect8::max_topk];
    };
    __shared__ Storage storage;
    for (uint32_t bucket = tid; bucket < WindowRadix::NUM_RECONSTRUCT_BUCKET_SLOTS; bucket += blockDim.x) {
        storage.reconstruct_bucket_counter[0][bucket] = 0;
        storage.reconstruct_bucket_counter[1][bucket] = 0;
    }
    const auto scores = static_cast<const nv_bfloat16*>(args.input) + size_t(row) * args.stride_input_batch;
    alignas(16) nv_bfloat162 values[span / 2];
    const uint32_t begin = tid * span;
    #pragma unroll
    for (uint32_t vec = 0; vec < span / 8; ++vec) {
        const uint32_t col = begin + vec * 8;
        if (col + 8 <= length) {
            reinterpret_cast<uint4*>(values)[vec] = *reinterpret_cast<const uint4*>(scores + col);
        } else {
            #pragma unroll
            for (uint32_t elem = 0; elem < 8; ++elem)
                reinterpret_cast<nv_bfloat16*>(values)[vec * 8 + elem] = col + elem < length ? scores[col + elem] : __ushort_as_bfloat16(0xff80);
        }
    }
    nv_bfloat162 maximum = values[0];
    #pragma unroll
    for (uint32_t pair = 1; pair < span / 2; ++pair) maximum = __hmax2(maximum, values[pair]);
    const uint32_t max_key = topk_select_common::distort(__bfloat16_as_ushort(__hmax(maximum.x, maximum.y)));
    const uint32_t warp_max = __reduce_max_sync(0xffffffffu, max_key);
    if (tid % 32 == 0) storage.selected[tid / 32] = warp_max;
    __syncthreads();
    const uint32_t lane = tid % 32;
    const uint32_t row_max = __reduce_max_sync(0xffffffffu, lane < CandidateSelect8::num_threads / 32 ? storage.selected[lane] : 0u);
    const uint32_t base = max(255u, row_max) - 255u;
    const uint32_t bias = base * 0x10001u;
    #pragma unroll
    for (uint32_t pair = 0; pair < span / 2; ++pair) {
        const uint32_t raw = topk_select_common::bf16x2_to_u32(values[pair]);
        uint32_t keys;
        topk_select_common::distort_x2<uint16_t>((uint16_t*)&keys, (const uint16_t*)&raw);
        const uint32_t bins = __vsubus2(keys, bias);
        atomicAdd(storage.reconstruct_bucket_counter[0] + (bins & 255u), 1u);
        atomicAdd(storage.reconstruct_bucket_counter[0] + ((bins >> 16) & 255u), 1u);
    }
    __syncthreads();
    if (tid < 32) WindowRadix::find_pivot_in_histogram<false>(storage, storage.reconstruct_bucket_counter[0], topk, tid);
    __syncthreads();
    const uint32_t window_bin = storage.reconstruct_pivot_bucket;
    uint32_t pivot_key = base + window_bin;
    if (window_bin == 0) {
        if (tid < 256) storage.reconstruct_bucket_counter[0][tid] = 0;
        __syncthreads();
        WindowRadix::histogram_radix_msb(storage.reconstruct_bucket_counter[0], values, span / 2);
        __syncthreads();
        if (tid < 32) WindowRadix::find_pivot_in_histogram<false>(storage, storage.reconstruct_bucket_counter[0], topk, tid);
        __syncthreads();
        const uint32_t high = storage.reconstruct_pivot_bucket;
        const uint32_t rank = storage.reconstruct_num_should_select;
        WindowRadix::histogram_radix_lsb_for_pivot_msb<false>(storage.reconstruct_bucket_counter[1], high, values, span / 2);
        __syncthreads();
        if (tid < 32) WindowRadix::find_pivot_in_histogram<false>(storage, storage.reconstruct_bucket_counter[1], rank, tid);
        __syncthreads();
        pivot_key = high * 256 + storage.reconstruct_pivot_bucket;
    }
    const uint32_t cutoff = 0x10001u * topk_select_common::un_distort(uint16_t(pivot_key));
    static_assert(span == 32);
    uint32_t greater = 0, equal = 0, has_nan = 0;
    #pragma unroll
    for (uint32_t pair = 0; pair < span / 2; ++pair) {
        const uint32_t bits = topk_select_common::bf16x2_to_u32(values[pair]);
        uint32_t gt, eq;
        asm("set.gt.u32.bf16x2 %0, %1, %2;" : "=r"(gt) : "r"(bits), "r"(cutoff));
        asm("set.eq.u32.bf16x2 %0, %1, %2;" : "=r"(eq) : "r"(bits), "r"(cutoff));
        greater |= ((gt & 1u) | ((gt >> 15) & 2u)) << (2 * pair);
        equal   |= ((eq & 1u) | ((eq >> 15) & 2u)) << (2 * pair);
        has_nan |= __vcmpgtu2(bits & 0x7fff7fffu, 0x7f807f80u);
    }
    if (__syncthreads_or(has_nan != 0)) {
        WindowRadix::take_action_when_have_nan(args, row);
        return;
    }
    auto prefix = WindowRadix::compute_equal_quota_and_prefix(__popc(greater), __popc(equal), topk,
                                                              tid / 32, tid % 32, storage.reconstruct_bucket_counter[0]);
    for (uint32_t quota = prefix.eq_quota; equal && quota; --quota) {
        greater |= equal & -equal;
        equal &= equal - 1;
    }
    while (greater) {
        storage.selected[prefix.start_pos_in_collector++] = begin + __ffs(greater) - 1;
        greater &= greater - 1;
    }
    __syncthreads();
    if (tid < topk) result[tid] = WindowRadix::EpilogueT::output_index(args, row, storage.selected[tid], offset);
}

extern "C" int select_candidates8(const void* input, const int* end, const int* candidates,
    const int* starts, int* output, int rows, int width, int stride, int topk, int output_stride, void* stream) {
    if (width > CandidateSelect8::max_vocab || topk > CandidateSelect8::max_topk || stride % 8) return int(cudaErrorInvalidValue);
    TopkSelectArgs args{};
    args.batch_size = rows;
    args.vocab_size = width;
    args.topk = topk;
    args.input = const_cast<void*>(input);
    args.output_index = output;
    args.end_ptr = const_cast<int*>(end);
    args.output_idx_offset = const_cast<int*>(starts);
    args.stride_input_batch = stride;
    args.stride_output_index_batch = output_stride;
    args.sorted_index = true;
    args.idx_oob_fill_value = -1;
    args.value_oob_fill_value = -INFINITY;
    args.abort_when_nan_found = true;
    args.stream = static_cast<cudaStream_t>(stream);
    args.candidate_blocks = candidates;
    select_register_window<<<rows, CandidateSelect8::num_threads, 0, args.stream>>>(args);
    return int(cudaGetLastError());
}
