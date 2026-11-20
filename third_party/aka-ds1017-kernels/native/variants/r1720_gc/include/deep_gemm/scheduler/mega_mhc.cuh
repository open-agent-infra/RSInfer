#pragma once

#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched::mega_mhc {

using namespace layout::mega_mhc;

// Every split arrives once; wait observes all split writes through the
// terminal release sequence. wait_init prevents a producer from racing the grid tag.
template <uint32_t kBarrierIndex>
struct SplitBarrier {
    static CUTLASS_DEVICE uint64_t* get_ptr(uint64_t* gmem_split_barriers, const uint32_t m_block_idx) {
        const auto line_idx = static_cast<uint64_t>(m_block_idx) * kNumSplitBarriers + kBarrierIndex;
        return math::advance_ptr<uint64_t>(gmem_split_barriers, line_idx * kSplitBarrierLineBytes);
    }

    static CUTLASS_DEVICE uint64_t encode_state(const uint64_t grid_idx, const uint32_t num_arrived_splits = 0) {
        return (grid_idx + 1) * kNumSplitBarriers * kNumMaxSplits + num_arrived_splits;
    }

    static CUTLASS_DEVICE void init(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        ptx::st_rel(get_ptr(gmem_split_barriers, m_block_idx), encode_state(ptx::get_grid_idx()));
    }

    template <uint32_t kNumSplits, bool kTags = false>
    static CUTLASS_DEVICE void wait_init(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        if constexpr (kTags) return;
        const auto current_grid_barrier_base = encode_state(ptx::get_grid_idx());
        const auto split_barrier_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        // NOTES: unsigned distance rejects stale grid tags but accepts any current-grid arrival count.
        while (ptx::ld_acq_gpu(split_barrier_ptr) - current_grid_barrier_base >= kNumSplits);
    }

    template <uint32_t kNumSplits, bool kTags = false>
    static CUTLASS_DEVICE void arrive(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        const auto state_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        if constexpr (kTags) {
            static_assert(1 + kNumSplits <= kSplitBarrierLineBytes / sizeof(uint64_t));
            ptx::st_rel(state_ptr + 1 + blockIdx.x % kNumSplits, ptx::get_grid_idx() + 1);
            return;
        }
        // NOTES: SM100 can retire the release reduction asynchronously while this CTA starts its next task.
        ptx::red_async_inc_rel(state_ptr);
    }

    template <uint32_t kNumSplits, bool kTags = false>
    static CUTLASS_DEVICE void wait(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        if constexpr (kTags) {
            const auto address = get_ptr(gmem_split_barriers, m_block_idx) + 1 + ptx::get_lane_idx() % kNumSplits;
            static_assert(kNumSplits <= 32);
            const auto tag = ptx::get_grid_idx() + 1;
            while (not __all_sync(0xffffffff, ptx::ld_acq_gpu(address) == tag));
            __syncwarp();
            return;
        }
        const auto all_splits_arrived_state = encode_state(ptx::get_grid_idx(), kNumSplits);
        const auto split_barrier_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        while (ptx::ld_acq_gpu(split_barrier_ptr) != all_splits_arrived_state);
    }
};

// Norm is used only by Shifted after materializing X1 and its statistics.
using Norm = SplitBarrier<0>;
using Mix = SplitBarrier<1>;

} // namespace deep_gemm::sched::mega_mhc
