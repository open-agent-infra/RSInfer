#pragma once

#include <cutlass/arch/memory_sm80.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched::sparse_mqa_logits {

using namespace layout::sparse_mqa_logits;

template <uint32_t BLOCK_Q, uint32_t NUM_MAX_SPARSE_BLOCKS,
          uint32_t kNumKVBlocksPerSplit, uint32_t kNumKVSplitsPerEntry,
          uint32_t kNumThreads>
struct SharedStorage {
    static constexpr uint32_t kNumMaxMergedKVBlocks = BLOCK_Q * NUM_MAX_SPARSE_BLOCKS;
    static constexpr uint32_t kNumMaxKVSplits = math::constexpr_ceil_div(kNumMaxMergedKVBlocks, kNumKVBlocksPerSplit);
    static constexpr uint32_t kNumWarps = kNumThreads / 32;

    uint32_t logical_kv_block_indices[2][NUM_MAX_SPARSE_BLOCKS];
    uint32_t packed_slots_by_merged_kv_block[kNumMaxMergedKVBlocks];
    uint32_t packed_slot_bases_by_kv_split[kNumMaxKVSplits];
    uint32_t warp_sums[kNumWarps];
    uint32_t num_waves;
    uint32_t kv_split_histograms[kNumWarps][kNumKVSplitsPerEntry + 1];
    struct alignas(16) {
        uint32_t q_token_base;
        uint32_t num_q_tokens;
        uint32_t num_kv_blocks[2];
        uint32_t kv_split_base;
        uint32_t kv_lens[2];
    } q_block;
    uint32_t is_last_cta;
    uint32_t valid_sums[2];
};

// Divide KV splits evenly, then split each SM range at Q-block boundaries.
template <uint32_t BLOCK_Q, uint32_t kNumKVBlocksPerSplit, uint32_t kNumSMs, typename smem_t>
CUTLASS_DEVICE uint32_t build_contiguous_schedule(
    ScheduleEntry* schedule_entries, const KVSplit<kNumKVBlocksPerSplit>* kv_splits,
    const QBlockInfo* q_block_infos, const uint32_t num_q_tokens,
    const uint32_t total_kv_splits, smem_t& smem
) {
    if (threadIdx.x == 0)
        smem.num_waves = 1;
    __syncthreads();

    uint32_t num_sm_entries = 0;
    if (threadIdx.x < kNumSMs) {
        uint32_t kv_split_idx = static_cast<uint32_t>(math::ceil_div<uint64_t>(
            static_cast<uint64_t>(total_kv_splits) * threadIdx.x, kNumSMs));
        const uint32_t kv_split_end = static_cast<uint32_t>(math::ceil_div<uint64_t>(
            static_cast<uint64_t>(total_kv_splits) * (threadIdx.x + 1), kNumSMs));
        while (kv_split_idx < kv_split_end) {
            const uint32_t q_token_base = kv_splits[kv_split_idx].header.q_token_base;
            const auto q_block_info = q_block_infos[q_token_base];
            const uint32_t entry_kv_split_end = cute::min(kv_split_end, q_block_info.kv_split_base + q_block_info.num_kv_splits);
            schedule_entries[num_sm_entries * kNumSMs + threadIdx.x] = ScheduleEntry(
                kv_split_idx, entry_kv_split_end, q_token_base, cute::min(BLOCK_Q, num_q_tokens - q_token_base));
            kv_split_idx = entry_kv_split_end;
            ++ num_sm_entries;
        }
        atomicMax(&smem.num_waves, num_sm_entries);
    }
    __syncthreads();

    const uint32_t num_waves = smem.num_waves;
    if (threadIdx.x < kNumSMs) {
        for (uint32_t wave_idx = num_sm_entries; wave_idx < num_waves; ++ wave_idx)
            schedule_entries[wave_idx * kNumSMs + threadIdx.x] = ScheduleEntry(0, 0, 0, 0);
    }
    return num_waves;
}

// Sort each wave by KV-split count and rotate heavy entries across SMs.
template <uint32_t kNumSMs, uint32_t kNumKVSplitsPerEntry, typename smem_t>
CUTLASS_DEVICE void balance_wave_entries(ScheduleEntry* schedule_entries, const uint32_t num_waves, smem_t& smem) {
    __syncthreads();
    if (num_waves == 1)
        return;
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();

    for (uint32_t wave_idx = warp_idx; wave_idx < num_waves; wave_idx += smem_t::kNumWarps) {
        constexpr uint32_t kNumEntriesPerLane = math::constexpr_ceil_div(kNumSMs, 32u);
        const auto histogram = smem.kv_split_histograms[warp_idx];
        const auto wave_entries = schedule_entries + wave_idx * kNumSMs;
        ScheduleEntry entries[kNumEntriesPerLane];
        uint32_t ranks[kNumEntriesPerLane];

        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++ entry_idx)
            entries[entry_idx] = wave_entries[sm_idx];
        for (uint32_t num_kv_splits = lane_idx; num_kv_splits <= kNumKVSplitsPerEntry; num_kv_splits += 32)
            histogram[num_kv_splits] = 0;
        __syncwarp();

        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++ entry_idx)
            ranks[entry_idx] = atomicAdd(histogram + entries[entry_idx].kv_split_end - entries[entry_idx].kv_split_begin, 1u);
        __syncwarp();

        if (cute::elect_one_sync()) {
            uint32_t rank_begin = 0;
            for (uint32_t num_kv_splits = 0; num_kv_splits <= kNumKVSplitsPerEntry; ++ num_kv_splits) {
                const uint32_t frequency = histogram[num_kv_splits];
                histogram[num_kv_splits] = rank_begin;
                rank_begin += frequency;
            }
        }
        __syncwarp();

        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++ entry_idx) {
            const uint32_t rank = histogram[entries[entry_idx].kv_split_end - entries[entry_idx].kv_split_begin] + ranks[entry_idx];
            // Reverse the two-wave tail to make active SM sets complementary.
            const uint32_t dst_sm_idx = num_waves == 2 and wave_idx == 1 ? kNumSMs - 1 - rank :
                                        (rank + kNumSMs - wave_idx * kNumSMs / num_waves) % kNumSMs;
            wave_entries[dst_sm_idx] = entries[entry_idx];
        }
    }
}

// Split each Q block into bounded entries, then balance each wave.
template <uint32_t BLOCK_Q, uint32_t kNumKVSplitsPerEntry,
          uint32_t kNumSMs, uint32_t kNumThreads, typename smem_t>
CUTLASS_DEVICE uint32_t build_paged_schedule(
    ScheduleEntry* schedule_entries, const QBlockInfo* q_block_infos,
    const uint32_t num_q_tokens, const uint32_t* indices, smem_t& smem
) {
    uint32_t num_entries = 0;
    for (uint32_t q_token_begin = 0; q_token_begin < num_q_tokens; q_token_begin += kNumThreads) {
        const uint32_t q_token_idx = q_token_begin + threadIdx.x;
        const auto q_block_info = q_token_idx < num_q_tokens ? q_block_infos[q_token_idx] : QBlockInfo(0, 0);
        const uint32_t num_kv_splits = q_block_info.num_kv_splits;
        const uint32_t num_q_entries = math::ceil_div(num_kv_splits, kNumKVSplitsPerEntry);
        uint32_t num_batch_entries;
        const uint32_t entry_begin = num_entries + math::cta_exclusive_sum<kNumThreads>(
            num_q_entries, smem.warp_sums, num_batch_entries);

        const uint32_t kv_splits_per_entry = num_q_entries == 0 ? 0 : num_kv_splits / num_q_entries;
        const uint32_t num_larger_entries = num_q_entries == 0 ? 0 : num_kv_splits % num_q_entries;
        const uint32_t num_q_block_tokens = q_token_idx + 1 < num_q_tokens and indices[q_token_idx + 1] == indices[q_token_idx] ? BLOCK_Q : 1;
        uint32_t kv_split_begin = q_block_info.kv_split_base;
        for (uint32_t q_entry_idx = 0; q_entry_idx < num_q_entries; ++ q_entry_idx) {
            const uint32_t kv_split_end = kv_split_begin + kv_splits_per_entry + (q_entry_idx < num_larger_entries);
            schedule_entries[entry_begin + q_entry_idx] = ScheduleEntry(
                kv_split_begin, kv_split_end, q_token_idx, num_q_block_tokens);
            kv_split_begin = kv_split_end;
        }
        num_entries += num_batch_entries;
        __syncthreads();
    }

    const uint32_t num_waves = cute::max(1u, math::ceil_div(num_entries, kNumSMs));
    for (uint32_t entry_idx = num_entries + threadIdx.x; entry_idx < num_waves * kNumSMs; entry_idx += kNumThreads)
        schedule_entries[entry_idx] = ScheduleEntry(0, 0, 0, 0);

    balance_wave_entries<kNumSMs, kNumKVSplitsPerEntry>(schedule_entries, num_waves, smem);
    return num_waves;
}

template <bool kIsPaged, bool kUseUnalignedKs, uint32_t BLOCK_Q,
          uint32_t SPLIT_KV, uint32_t SPARSE_BLOCK_KV,
          uint32_t NUM_MAX_SPARSE_BLOCKS, uint32_t PAGE_KV,
          uint32_t kNumKVSplitsPerEntry, uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 4)
void sm100_sparse_mqa_logits_metadata(
    const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
    const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
    const uint32_t* context_lens, const uint32_t* block_table, const uint32_t block_table_stride,
    const uint32_t* indices,
    const uint32_t* sparse_kv_block_indices, uint8_t* metadata, uint8_t* workspace,
    int* candidate_valid
) {
    // Shared memory sizes
    constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    using smem_t = SharedStorage<
        BLOCK_Q, NUM_MAX_SPARSE_BLOCKS, kNumKVBlocksPerSplit, kNumKVSplitsPerEntry, kNumThreads>;

    // Template checks
    DG_STATIC_ASSERT((BLOCK_Q == 1 or BLOCK_Q == 2) and (kNumThreads == 256 or kNumThreads == 512) and kNumSMs <= kNumThreads, "Unsupported metadata shape");
    DG_STATIC_ASSERT(SPLIT_KV % SPARSE_BLOCK_KV == 0 and SPLIT_KV % 128 == 0, "Invalid sparse split shape");
    DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS % 4 == 0, "Sparse index rows must be 16-byte aligned");
    DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS <= 4096, "Too many sparse KV blocks");
    DG_STATIC_ASSERT(kNumKVBlocksPerSplit <= kInvalidSparseSlot, "Sparse slot offset overflow");
    DG_STATIC_ASSERT(kNumKVSplitsPerEntry > 0, "Invalid number of KV splits per entry");
    DG_STATIC_ASSERT(not kIsPaged or PAGE_KV % SPARSE_BLOCK_KV == 0, "Invalid page shape");
    DG_STATIC_ASSERT(not kIsPaged or not kUseUnalignedKs, "Paged sparse MQA does not use ks");
    DG_DEVICE_ASSERT(candidate_valid == nullptr or not kIsPaged);

    const uint32_t thread_idx = threadIdx.x;
    cudaGridDependencySynchronize();

    // Valid-token contribution of one candidate block (mirrors the compact_lens kernel).
    const auto block_contrib = [](uint32_t b, uint32_t kl) -> uint32_t {
        int bb = (int)b;
        if (bb < 0) return 0;
        int t = (int)kl - bb * (int)SPARSE_BLOCK_KV;
        return t <= 0 ? 0 : (t >= (int)SPARSE_BLOCK_KV ? SPARSE_BLOCK_KV : (uint32_t)t);
    };

    const auto workspace_state = reinterpret_cast<WorkspaceState*>(workspace);
    const auto q_block_infos = reinterpret_cast<QBlockInfo*>(workspace + sizeof(WorkspaceState));
    const auto kv_splits = reinterpret_cast<KVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(MetadataHeader));

    // Shared memory
    extern __shared__ __align__(16) uint8_t storage[];
    auto& smem = *reinterpret_cast<smem_t*>(storage);

    // Build Q-block metadata
    uint32_t q_token_idx = blockIdx.x * (kIsPaged ? 1 : BLOCK_Q);
    while (true) {
        if (thread_idx == 0) {
            smem.q_block.num_q_tokens = 0;
            while (q_token_idx < num_q_tokens) {
                uint32_t num_q_block_tokens = cute::min(BLOCK_Q, num_q_tokens - q_token_idx);
                if constexpr (kIsPaged) {
                    // Keep Q blocks within one request
                    const uint32_t request_idx = indices[q_token_idx];
                    // Requests contain few Q tokens, so use a linear scan
                    uint32_t request_q_token_base = q_token_idx;
                    while (request_q_token_base > 0 and indices[request_q_token_base - 1] == request_idx)
                        --request_q_token_base;
                    if ((q_token_idx - request_q_token_base) % BLOCK_Q != 0) {
                        q_token_idx = gridDim.x + ptx::atomic_add(&workspace_state->next_q_offset, 1u);
                        continue;
                    }
                    num_q_block_tokens = q_token_idx + 1 < num_q_tokens and
                        indices[q_token_idx + 1] == request_idx ? BLOCK_Q : 1;
                }
                const auto get_num_kv_blocks = [&](const uint32_t q_idx) {
                    const uint32_t kv_begin = kIsPaged ? 0 : cute::min(cu_seq_len_k_start[q_idx], num_kv_tokens);
                    const uint32_t kv_end = kIsPaged ? context_lens[q_idx] :
                        cute::max(kv_begin, cute::min(cu_seq_len_k_end[q_idx], num_kv_tokens));
                    if constexpr (not kIsPaged and not kUseUnalignedKs)
                        DG_DEVICE_ASSERT(kv_end == kv_begin or kv_begin % SPARSE_BLOCK_KV == 0);
                    return cute::min(NUM_MAX_SPARSE_BLOCKS, math::ceil_div(kv_end - kv_begin, SPARSE_BLOCK_KV));
                };
                const auto get_kv_len = [&](const uint32_t q_idx) {
                    const uint32_t kv_begin = kIsPaged ? 0 : cute::min(cu_seq_len_k_start[q_idx], num_kv_tokens);
                    const uint32_t kv_end = kIsPaged ? context_lens[q_idx] :
                        cute::max(kv_begin, cute::min(cu_seq_len_k_end[q_idx], num_kv_tokens));
                    return kv_end - kv_begin;
                };
                smem.q_block.q_token_base = q_token_idx;
                smem.q_block.num_q_tokens = num_q_block_tokens;
                smem.q_block.num_kv_blocks[0] = get_num_kv_blocks(q_token_idx);
                smem.q_block.num_kv_blocks[1] = num_q_block_tokens == 2 ? get_num_kv_blocks(q_token_idx + 1) : 0;
                smem.q_block.kv_lens[0] = get_kv_len(q_token_idx);
                smem.q_block.kv_lens[1] = num_q_block_tokens == 2 ? get_kv_len(q_token_idx + 1) : 0;
                smem.valid_sums[0] = 0;
                smem.valid_sums[1] = 0;
                break;
            }
        }
        __syncthreads();

        const uint32_t num_q_block_tokens = smem.q_block.num_q_tokens;
        if (num_q_block_tokens == 0)
            break;
        const uint32_t q_token_base = smem.q_block.q_token_base;
        const uint32_t num_kv_blocks_in_q0 = smem.q_block.num_kv_blocks[0];
        const uint32_t num_kv_blocks_in_q1 = smem.q_block.num_kv_blocks[1];

        // Load sparse indices
        uint32_t valid_sum[2] = {0, 0};
        #pragma unroll
        for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++ q_token_offset) {
            const auto src = sparse_kv_block_indices + static_cast<uint64_t>(q_token_base + q_token_offset) * NUM_MAX_SPARSE_BLOCKS;
            const auto dst = smem.logical_kv_block_indices[q_token_offset];
            const uint32_t num_kv_blocks = q_token_offset == 0 ? num_kv_blocks_in_q0 : num_kv_blocks_in_q1;
            if constexpr (kUseUnalignedKs) {
                if (num_kv_blocks != 0) {
                    // Relative candidate slots: logical KV token start = slot * SPARSE_BLOCK_KV + ks.
                    // This equals (slot + ks / SPARSE_BLOCK_KV) * SPARSE_BLOCK_KV + ks % SPARSE_BLOCK_KV,
                    // so the host no longer needs to materialize absolute block indices.
                    const uint32_t kv_base = cu_seq_len_k_start[q_token_base + q_token_offset];
                    for (uint32_t q_slot_idx = thread_idx * 4; q_slot_idx < num_kv_blocks; q_slot_idx += kNumThreads * 4) {
                        const auto blocks = ptx::ld_evict_first(reinterpret_cast<const uint4*>(src + q_slot_idx));
                        ptx::st_shared(dst + q_slot_idx,
                            blocks.x * SPARSE_BLOCK_KV + kv_base, blocks.y * SPARSE_BLOCK_KV + kv_base,
                            blocks.z * SPARSE_BLOCK_KV + kv_base, blocks.w * SPARSE_BLOCK_KV + kv_base);
                        if (candidate_valid != nullptr) {
                            const uint32_t kl = smem.q_block.kv_lens[q_token_offset];
                            valid_sum[q_token_offset] += block_contrib(blocks.x, kl) + block_contrib(blocks.y, kl)
                                                       + block_contrib(blocks.z, kl) + block_contrib(blocks.w, kl);
                        }
                    }
                    if (candidate_valid != nullptr) {
                        const uint32_t kl = smem.q_block.kv_lens[q_token_offset];
                        for (uint32_t j = num_kv_blocks + thread_idx; j < NUM_MAX_SPARSE_BLOCKS; j += kNumThreads)
                            valid_sum[q_token_offset] += block_contrib(src[j], kl);
                    }
                }
            } else {
                for (uint32_t q_slot_idx = thread_idx * 4; q_slot_idx < num_kv_blocks; q_slot_idx += kNumThreads * 4)
                    ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(src + q_slot_idx), reinterpret_cast<uint4*>(dst + q_slot_idx));
            }
        }
        if constexpr (not kUseUnalignedKs) {
            cutlass::arch::cp_async_fence();
            cutlass::arch::cp_async_wait<0>();
            if (candidate_valid != nullptr) {
                #pragma unroll
                for (uint32_t r = 0; r < BLOCK_Q; ++ r) {
                    const uint32_t kl = smem.q_block.kv_lens[r];
                    for (uint32_t j = thread_idx; j < NUM_MAX_SPARSE_BLOCKS; j += kNumThreads)
                        valid_sum[r] += block_contrib(smem.logical_kv_block_indices[r][j], kl);
                }
            }
        }
        __syncthreads();

        // Fused compact_lens: reduce per-row valid-token counts accumulated during the index load.
        if (candidate_valid != nullptr) {
            #pragma unroll
            for (uint32_t r = 0; r < BLOCK_Q; ++ r) {
                uint32_t s = __reduce_add_sync(0xFFFFFFFFu, valid_sum[r]);
                if (ptx::get_lane_idx() == 0)
                    atomicAdd(&smem.valid_sums[r], s);
            }
            __syncthreads();
            if (thread_idx == 0) {
                candidate_valid[q_token_base] = (int)smem.valid_sums[0];
                if (num_q_block_tokens == 2)
                    candidate_valid[q_token_base + 1] = (int)smem.valid_sums[1];
            }
        }

        // Merge-path partition
        const uint32_t num_input_kv_blocks = num_kv_blocks_in_q0 + num_kv_blocks_in_q1;
        const uint32_t merge_begin = thread_idx * num_input_kv_blocks / kNumThreads;
        const uint32_t merge_end = (thread_idx + 1) * num_input_kv_blocks / kNumThreads;
        uint32_t lo = cute::max(merge_begin, num_kv_blocks_in_q1) - num_kv_blocks_in_q1;
        uint32_t hi = cute::min(merge_begin, num_kv_blocks_in_q0);
        while (lo < hi) {
            const uint32_t q0_slot_idx = (lo + hi) / 2;
            const uint32_t q1_slot_idx = merge_begin - q0_slot_idx;
            if (q1_slot_idx > 0 and q0_slot_idx < num_kv_blocks_in_q0 and
                    ptx::ld_shared(smem.logical_kv_block_indices[1] + q1_slot_idx - 1) >= ptx::ld_shared(smem.logical_kv_block_indices[0] + q0_slot_idx))
                lo = q0_slot_idx + 1;
            else
                hi = q0_slot_idx;
        }
        uint32_t q0_slot_idx = lo;
        uint32_t q1_slot_idx = merge_begin - lo;
        uint32_t num_remaining_inputs = merge_end - merge_begin;
        uint32_t num_merged_kv_blocks_in_thread = 0;

        // Drop a duplicate carried across merge partitions
        if (num_remaining_inputs > 0 and q0_slot_idx > 0 and q1_slot_idx < num_kv_blocks_in_q1 and
            ptx::ld_shared(smem.logical_kv_block_indices[0] + q0_slot_idx - 1) == ptx::ld_shared(smem.logical_kv_block_indices[1] + q1_slot_idx)) {
            ++ q1_slot_idx;
            -- num_remaining_inputs;
        }

        // Merge and pack each Q's sparse slot and presence bit
        constexpr uint32_t kPresentBit = 1u << (kNumSparseSlotBits - 1);
        constexpr uint32_t kSlotIndexMask = kPresentBit - 1;
        const auto pack_slot = [](const uint32_t q_slot_idx, const bool is_present) {
            return q_slot_idx | (is_present ? kPresentBit : 0u);
        };
        constexpr uint32_t kNumKVBlocksPerThread =
            math::constexpr_ceil_div(smem_t::kNumMaxMergedKVBlocks, kNumThreads);
        uint32_t packed_slots_in_thread[kNumKVBlocksPerThread];
        #pragma unroll
        for (uint32_t merged_kv_block_offset_in_thread = 0; merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++ merged_kv_block_offset_in_thread) {
            if (num_remaining_inputs == 0)
                continue;
            const uint32_t q0_logical_kv_block_idx = q0_slot_idx < num_kv_blocks_in_q0 ? ptx::ld_shared(smem.logical_kv_block_indices[0] + q0_slot_idx) : ~0u;
            const uint32_t q1_logical_kv_block_idx = q1_slot_idx < num_kv_blocks_in_q1 ? ptx::ld_shared(smem.logical_kv_block_indices[1] + q1_slot_idx) : ~0u;
            const bool in_q0 = q0_logical_kv_block_idx <= q1_logical_kv_block_idx;
            const bool in_q1 = q1_logical_kv_block_idx <= q0_logical_kv_block_idx;
            // Carry a final duplicate into the next partition
            const bool consume_q1 = in_q1 and num_remaining_inputs > in_q0;
            packed_slots_in_thread[merged_kv_block_offset_in_thread] =
                pack_slot(q0_slot_idx, in_q0) | (pack_slot(q1_slot_idx, in_q1) << kNumSparseSlotBits);
            ++ num_merged_kv_blocks_in_thread;
            q0_slot_idx += in_q0;
            q1_slot_idx += consume_q1;
            num_remaining_inputs -= in_q0 + consume_q1;
        }
        DG_DEVICE_ASSERT(num_remaining_inputs == 0);

        // Compact merged blocks and reserve their KV-split range
        uint32_t num_merged_kv_blocks;
        const uint32_t merged_kv_block_base = math::cta_exclusive_sum<kNumThreads>(
            num_merged_kv_blocks_in_thread, smem.warp_sums, num_merged_kv_blocks);
        const uint32_t num_kv_splits_in_q_block = math::ceil_div(num_merged_kv_blocks, kNumKVBlocksPerSplit);
        if (thread_idx == 0) {
            smem.q_block.kv_split_base = static_cast<uint32_t>(num_kv_splits_in_q_block == 0 ? 0 :
                ptx::atomic_add(&workspace_state->num_kv_splits, num_kv_splits_in_q_block));
            q_block_infos[q_token_base] = QBlockInfo(smem.q_block.kv_split_base, num_kv_splits_in_q_block);
            if constexpr (kIsPaged) {
            if (num_q_block_tokens == 2)
                    q_block_infos[q_token_base + 1] = QBlockInfo(0, 0);
            }
        }
        #pragma unroll
        for (uint32_t merged_kv_block_offset_in_thread = 0; merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++ merged_kv_block_offset_in_thread) {
            if (merged_kv_block_offset_in_thread >= num_merged_kv_blocks_in_thread)
                continue;
            const uint32_t merged_kv_block_idx = merged_kv_block_base + merged_kv_block_offset_in_thread;
            if constexpr (kIsPaged) {
                if (merged_kv_block_idx % kNumKVBlocksPerSplit == 0) {
                    smem.packed_slot_bases_by_kv_split[merged_kv_block_idx / kNumKVBlocksPerSplit] = packed_slots_in_thread[merged_kv_block_offset_in_thread] &
                        (kSlotIndexMask | (kSlotIndexMask << kNumSparseSlotBits));
                }
            } else {
                smem.packed_slots_by_merged_kv_block[merged_kv_block_idx] = packed_slots_in_thread[merged_kv_block_offset_in_thread];
            }
        }
        __syncthreads();

        const uint32_t kv_split_base = smem.q_block.kv_split_base;
        const auto write_merged_kv_block = [&](const uint32_t merged_kv_block_idx, const uint32_t packed_slots) {
            const uint32_t kv_split_offset = merged_kv_block_idx / kNumKVBlocksPerSplit;
            const uint32_t kv_split_idx = kv_split_base + kv_split_offset;
            const uint32_t kv_block_idx_in_split = merged_kv_block_idx % kNumKVBlocksPerSplit;
            // Paged stores one packed Q-slot base per split; non-paged retains each merged KV block's Q slots for its second pass
            uint32_t packed_slot_bases;
            if constexpr (kIsPaged)
                packed_slot_bases = ptx::ld_shared(smem.packed_slot_bases_by_kv_split + kv_split_offset);
            else
                packed_slot_bases = ptx::ld_shared(smem.packed_slots_by_merged_kv_block + merged_kv_block_idx - kv_block_idx_in_split);
            const uint32_t q0_slot_idx = packed_slots & kSlotIndexMask;
            const uint32_t q1_slot_idx = (packed_slots >> kNumSparseSlotBits) & kSlotIndexMask;
            const uint32_t q0_slot_base = packed_slot_bases & kSlotIndexMask;
            const uint32_t q1_slot_base = (packed_slot_bases >> kNumSparseSlotBits) & kSlotIndexMask;
            const bool in_q0 = (packed_slots & kPresentBit) != 0;
            const bool in_q1 = ((packed_slots >> kNumSparseSlotBits) & kPresentBit) != 0;
            const uint32_t logical_kv_block_idx = in_q0 ? ptx::ld_shared(smem.logical_kv_block_indices[0] + q0_slot_idx)
                                                        : ptx::ld_shared(smem.logical_kv_block_indices[1] + q1_slot_idx);
            uint32_t physical_kv_block_idx;
            if constexpr (kIsPaged) {
                constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t logical_page_idx = logical_kv_block_idx / kNumKVBlocksPerPage;
                DG_DEVICE_ASSERT(logical_page_idx < block_table_stride);
                const uint64_t block_table_idx = static_cast<uint64_t>(q_token_base) * block_table_stride + logical_page_idx;
                physical_kv_block_idx = block_table[block_table_idx] * kNumKVBlocksPerPage + logical_kv_block_idx % kNumKVBlocksPerPage;
            } else {
                physical_kv_block_idx = logical_kv_block_idx * (kUseUnalignedKs ? 1 : SPARSE_BLOCK_KV);
            }
            DG_DEVICE_ASSERT(not in_q0 or q0_slot_idx - q0_slot_base < kInvalidSparseSlot);
            DG_DEVICE_ASSERT(not in_q1 or q1_slot_idx - q1_slot_base < kInvalidSparseSlot);
            const uint32_t q0_slot_offset = in_q0 ? q0_slot_idx - q0_slot_base : kInvalidSparseSlot;
            const uint32_t q1_slot_offset = in_q1 ? q1_slot_idx - q1_slot_base : kInvalidSparseSlot;
            if (kv_block_idx_in_split == 0) {
                const uint32_t num_kv_blocks_in_split = cute::min(kNumKVBlocksPerSplit, num_merged_kv_blocks - merged_kv_block_idx);
                bool is_contiguous = false, has_partial_tail = false;
                if constexpr (not kIsPaged) {
                    // Merged blocks are sorted, so only the last one can run past the KV tensor
                    const uint32_t last_packed_slots = ptx::ld_shared(smem.packed_slots_by_merged_kv_block + merged_kv_block_idx + num_kv_blocks_in_split - 1);
                    const bool last_in_q0 = (last_packed_slots & kPresentBit) != 0;
                    const uint32_t last_q_slot_idx = last_in_q0 ? last_packed_slots & kSlotIndexMask
                                                                : (last_packed_slots >> kNumSparseSlotBits) & kSlotIndexMask;
                    const auto last_logical_kv_block_indices = smem.logical_kv_block_indices[last_in_q0 ? 0 : 1];
                    const uint32_t last_logical_kv_block_idx = ptx::ld_shared(last_logical_kv_block_indices + last_q_slot_idx);
                    const uint32_t last_kv_token_start = last_logical_kv_block_idx * (kUseUnalignedKs ? 1 : SPARSE_BLOCK_KV);
                    // TMA copies full splits; partial splits stay on the generic path
                    if constexpr (not kUseUnalignedKs) {
                        is_contiguous = num_kv_blocks_in_split == kNumKVBlocksPerSplit and
                                        last_logical_kv_block_idx == logical_kv_block_idx + num_kv_blocks_in_split - 1;
                    }
                    // TMA zero-fills past the tensor itself, so only generic copies need the clipped path
                    has_partial_tail = not is_contiguous and last_kv_token_start + SPARSE_BLOCK_KV > num_kv_tokens;
                }
                kv_splits[kv_split_idx].header = KVSplitHeader(q_token_base, num_kv_blocks_in_split, is_contiguous, has_partial_tail,
                    q0_slot_base, num_q_block_tokens == 2 ? q1_slot_base : ~0u);
            }
            kv_splits[kv_split_idx].kv_block_infos[kv_block_idx_in_split] = KVBlockInfo(physical_kv_block_idx, q0_slot_offset, q1_slot_offset);
        };

        // Write KV split metadata
        if constexpr (kIsPaged) {
            #pragma unroll
            for (uint32_t merged_kv_block_offset_in_thread = 0; merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++ merged_kv_block_offset_in_thread) {
                if (merged_kv_block_offset_in_thread >= num_merged_kv_blocks_in_thread)
                    continue;
                const uint32_t merged_kv_block_idx = merged_kv_block_base + merged_kv_block_offset_in_thread;
                write_merged_kv_block(merged_kv_block_idx, packed_slots_in_thread[merged_kv_block_offset_in_thread]);
            }
        } else {
            for (uint32_t merged_kv_block_idx = thread_idx; merged_kv_block_idx < num_merged_kv_blocks; merged_kv_block_idx += kNumThreads)
                write_merged_kv_block(merged_kv_block_idx, ptx::ld_shared(smem.packed_slots_by_merged_kv_block + merged_kv_block_idx));
        }

        // Pad the last split to keep the main-kernel copy loop branch-free
        for (uint32_t padded_kv_block_slot_idx = num_merged_kv_blocks + thread_idx; padded_kv_block_slot_idx < num_kv_splits_in_q_block * kNumKVBlocksPerSplit;
             padded_kv_block_slot_idx += kNumThreads) {
            kv_splits[kv_split_base + padded_kv_block_slot_idx / kNumKVBlocksPerSplit].kv_block_infos[padded_kv_block_slot_idx % kNumKVBlocksPerSplit] =
                KVBlockInfo(0, kInvalidSparseSlot, kInvalidSparseSlot);
        }
        if (thread_idx == 0) {
            q_token_idx = kIsPaged ? gridDim.x + ptx::atomic_add(&workspace_state->next_q_offset, 1u) :
                                     q_token_idx + gridDim.x * BLOCK_Q;
        }
    }

    // The last CTA acquires all producer writes and builds the schedule
    if (thread_idx == 0) {
        smem.is_last_cta = ptx::atomic_add_rel(&workspace_state->num_finished_ctas, 1u) + 1 == gridDim.x;
        if (smem.is_last_cta != 0)
            ptx::ld_acq(&workspace_state->num_finished_ctas);
    }
    __syncthreads();
    if (smem.is_last_cta == 0)
        return;

    const uint32_t total_kv_splits = workspace_state->num_kv_splits;
    const auto schedule_entries = reinterpret_cast<ScheduleEntry*>(kv_splits + total_kv_splits);
    uint32_t num_waves;
    if constexpr (not kIsPaged) {
        num_waves = build_contiguous_schedule<BLOCK_Q, kNumKVBlocksPerSplit, kNumSMs>(
            schedule_entries, kv_splits, q_block_infos, num_q_tokens, total_kv_splits, smem);
    } else {
        num_waves = build_paged_schedule<BLOCK_Q, kNumKVSplitsPerEntry, kNumSMs, kNumThreads>(
            schedule_entries, q_block_infos, num_q_tokens, indices, smem);
    }
    if (thread_idx == 0) {
        const auto header = reinterpret_cast<MetadataHeader*>(metadata);
        header->num_kv_splits = total_kv_splits;
        header->num_waves = num_waves;
        header->use_unaligned_ks = kUseUnalignedKs;
        header->num_sms = kNumSMs;
        workspace_state->num_kv_splits = 0;
        workspace_state->next_q_offset = 0;
        workspace_state->num_finished_ctas = 0;
    }
}

template <bool kIsPaged, bool kUseUnalignedKs, uint32_t BLOCK_Q,
          uint32_t SPLIT_KV, uint32_t SPARSE_BLOCK_KV, uint32_t NUM_MAX_SPARSE_BLOCKS,
          uint32_t PAGE_KV, uint32_t kNumKVSplitsPerEntry, uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 4)
void sm100_sparse_mqa_logits_single_q_metadata(
    const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
    const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
    const uint32_t* context_lens, const uint32_t* block_table, const uint32_t block_table_stride,
    const uint32_t* indices,
    const uint32_t* sparse_kv_block_indices, uint8_t* metadata, uint8_t* workspace,
    int* candidate_valid
) {
    DG_STATIC_ASSERT(not kIsPaged and kUseUnalignedKs and BLOCK_Q == 1, "Invalid single-Q sparse route");
    DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS % 4 == 0, "Invalid sparse row alignment");
    constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    DG_STATIC_ASSERT(kNumKVBlocksPerSplit % 4 == 0, "Invalid sparse split alignment");
    using smem_t = SharedStorage<1, NUM_MAX_SPARSE_BLOCKS, kNumKVBlocksPerSplit,
                                 kNumKVSplitsPerEntry, kNumThreads>;
    extern __shared__ __align__(16) uint8_t storage[];
    auto& smem = *reinterpret_cast<smem_t*>(storage);
    const auto workspace_state = reinterpret_cast<WorkspaceState*>(workspace);
    const auto q_block_infos = reinterpret_cast<QBlockInfo*>(workspace + sizeof(WorkspaceState));
    const auto kv_splits = reinterpret_cast<KVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(MetadataHeader));
    const uint32_t thread_idx = threadIdx.x;
    cudaGridDependencySynchronize();

    for (uint32_t q_token_idx = blockIdx.x; q_token_idx < num_q_tokens; q_token_idx += gridDim.x) {
        if (thread_idx == 0) {
            const uint32_t kv_begin = cute::min(cu_seq_len_k_start[q_token_idx], num_kv_tokens);
            const uint32_t kv_end = cute::max(kv_begin, cute::min(cu_seq_len_k_end[q_token_idx], num_kv_tokens));
            const uint32_t num_kv_blocks = cute::min(NUM_MAX_SPARSE_BLOCKS,
                math::ceil_div(kv_end - kv_begin, SPARSE_BLOCK_KV));
            const uint32_t num_kv_splits = math::ceil_div(num_kv_blocks, kNumKVBlocksPerSplit);
            const uint32_t kv_split_base = num_kv_splits == 0 ? 0 :
                ptx::atomic_add(&workspace_state->num_kv_splits, num_kv_splits);
            smem.q_block.kv_lens[0] = kv_end - kv_begin;
            smem.q_block.num_kv_blocks[0] = num_kv_blocks;
            smem.q_block.kv_split_base = kv_split_base;
            q_block_infos[q_token_idx] = QBlockInfo(kv_split_base, num_kv_splits);
        }
        __syncthreads();
        const uint32_t num_kv_blocks = smem.q_block.num_kv_blocks[0];
        const uint32_t kv_split_base = smem.q_block.kv_split_base;
        const uint32_t kv_len = smem.q_block.kv_lens[0];
        const uint32_t kv_begin = cu_seq_len_k_start[q_token_idx];
        const auto src = sparse_kv_block_indices + static_cast<uint64_t>(q_token_idx) * NUM_MAX_SPARSE_BLOCKS;
        uint32_t valid_sum = 0;
        for (uint32_t slot_base = thread_idx * 4; slot_base < NUM_MAX_SPARSE_BLOCKS; slot_base += kNumThreads * 4) {
            const auto packed_blocks = ptx::ld_evict_first(reinterpret_cast<const uint4*>(src + slot_base));
            const uint32_t candidate_blocks[4] = {packed_blocks.x, packed_blocks.y, packed_blocks.z, packed_blocks.w};
            const bool full_group = slot_base + 4 <= num_kv_blocks;
            if (full_group) {
                const uint32_t split_offset = slot_base / kNumKVBlocksPerSplit;
                const uint32_t block_offset = slot_base % kNumKVBlocksPerSplit;
                const uint32_t invalid_slot = kInvalidSparseSlot << kNumSparseSlotBits;
                auto destination = reinterpret_cast<uint4*>(
                    kv_splits[kv_split_base + split_offset].kv_block_infos + block_offset);
                destination[0] = make_uint4(packed_blocks.x * SPARSE_BLOCK_KV + kv_begin,
                    block_offset | invalid_slot, packed_blocks.y * SPARSE_BLOCK_KV + kv_begin,
                    (block_offset + 1) | invalid_slot);
                destination[1] = make_uint4(packed_blocks.z * SPARSE_BLOCK_KV + kv_begin,
                    (block_offset + 2) | invalid_slot, packed_blocks.w * SPARSE_BLOCK_KV + kv_begin,
                    (block_offset + 3) | invalid_slot);
            }
            #pragma unroll
            for (uint32_t slot_offset = 0; slot_offset < 4; ++slot_offset) {
                const uint32_t block_idx = candidate_blocks[slot_offset];
                if (candidate_valid != nullptr) {
                    const int remaining = static_cast<int>(kv_len) - static_cast<int>(block_idx) * static_cast<int>(SPARSE_BLOCK_KV);
                    if (static_cast<int>(block_idx) >= 0 and remaining > 0)
                        valid_sum += cute::min(static_cast<uint32_t>(remaining), SPARSE_BLOCK_KV);
                }
                const uint32_t slot_idx = slot_base + slot_offset;
                if (slot_idx >= num_kv_blocks)
                    continue;
                const uint32_t split_offset = slot_idx / kNumKVBlocksPerSplit;
                const uint32_t block_offset = slot_idx % kNumKVBlocksPerSplit;
                if (not full_group)
                    kv_splits[kv_split_base + split_offset].kv_block_infos[block_offset] =
                        KVBlockInfo(block_idx * SPARSE_BLOCK_KV + kv_begin, block_offset, kInvalidSparseSlot);
                if (block_offset + 1 == kNumKVBlocksPerSplit or slot_idx + 1 == num_kv_blocks)
                    smem.packed_slot_bases_by_kv_split[split_offset] =
                        block_idx * SPARSE_BLOCK_KV + kv_begin + SPARSE_BLOCK_KV > num_kv_tokens;
            }
        }
        const uint32_t padded_slots = math::ceil_div(num_kv_blocks, kNumKVBlocksPerSplit) * kNumKVBlocksPerSplit;
        for (uint32_t slot_idx = num_kv_blocks + thread_idx; slot_idx < padded_slots; slot_idx += kNumThreads)
            kv_splits[kv_split_base + slot_idx / kNumKVBlocksPerSplit].kv_block_infos[slot_idx % kNumKVBlocksPerSplit] =
                KVBlockInfo(0, kInvalidSparseSlot, kInvalidSparseSlot);
        if (candidate_valid != nullptr) {
            uint32_t total_valid;
            math::cta_exclusive_sum<kNumThreads>(valid_sum, smem.warp_sums, total_valid);
            if (thread_idx == 0)
                candidate_valid[q_token_idx] = static_cast<int>(total_valid);
        } else {
            __syncthreads();
        }
        for (uint32_t split_offset = thread_idx; split_offset < padded_slots / kNumKVBlocksPerSplit; split_offset += kNumThreads) {
            const uint32_t slot_base = split_offset * kNumKVBlocksPerSplit;
            kv_splits[kv_split_base + split_offset].header = KVSplitHeader(
                q_token_idx, cute::min(kNumKVBlocksPerSplit, num_kv_blocks - slot_base), false,
                smem.packed_slot_bases_by_kv_split[split_offset] != 0, slot_base, ~0u);
        }
        __syncthreads();
    }
    if (thread_idx == 0) {
        smem.is_last_cta = ptx::atomic_add_rel(&workspace_state->num_finished_ctas, 1u) + 1 == gridDim.x;
        if (smem.is_last_cta != 0)
            ptx::ld_acq(&workspace_state->num_finished_ctas);
    }
    __syncthreads();
    if (smem.is_last_cta == 0)
        return;
    const uint32_t total_kv_splits = workspace_state->num_kv_splits;
    const auto schedule_entries = reinterpret_cast<ScheduleEntry*>(kv_splits + total_kv_splits);
    const uint32_t num_waves = build_contiguous_schedule<1, kNumKVBlocksPerSplit, kNumSMs>(
        schedule_entries, kv_splits, q_block_infos, num_q_tokens, total_kv_splits, smem);
    if (thread_idx == 0) {
        const auto header = reinterpret_cast<MetadataHeader*>(metadata);
        header->num_kv_splits = total_kv_splits;
        header->num_waves = num_waves;
        header->use_unaligned_ks = true;
        header->num_sms = kNumSMs;
        workspace_state->num_kv_splits = 0;
        workspace_state->next_q_offset = 0;
        workspace_state->num_finished_ctas = 0;
    }
}

template <bool kIsPaged, bool kUseUnalignedKs, uint32_t BLOCK_Q,
          uint32_t SPLIT_KV, uint32_t SPARSE_BLOCK_KV, uint32_t NUM_MAX_SPARSE_BLOCKS,
          uint32_t PAGE_KV, uint32_t kNumKVSplitsPerEntry, uint32_t kNumSMs, uint32_t kNumThreads,
          bool kHeadersOnly = false>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads)
void sm100_sparse_mqa_row_plan(const uint32_t rows, const uint32_t num_kv_tokens,
    const uint32_t* starts, const uint32_t* ends, const uint32_t*, const uint32_t*, const uint32_t,
    const uint32_t*, const uint32_t* candidates, uint8_t* metadata, uint8_t*, int* candidate_valid) {
    static_assert(!kIsPaged && kUseUnalignedKs && BLOCK_Q == 1);
    constexpr uint32_t capacity = SPLIT_KV / SPARSE_BLOCK_KV;
    constexpr uint32_t reserved = math::constexpr_ceil_div(NUM_MAX_SPARSE_BLOCKS, capacity);
    const uint32_t row = blockIdx.x;
    cudaGridDependencySynchronize();
    const uint32_t begin = cute::min(starts[row], num_kv_tokens);
    const uint32_t length = cute::max(begin, cute::min(ends[row], num_kv_tokens)) - begin;
    const uint32_t count = cute::min(NUM_MAX_SPARSE_BLOCKS, math::ceil_div(length, SPARSE_BLOCK_KV));
    const uint32_t splits = math::ceil_div(count, capacity);
    const uint32_t waves = math::ceil_div(rows, kNumSMs);
    const uint32_t tail = rows % kNumSMs;
    const bool split_tail = tail > 1 && tail * reserved > kNumSMs &&
        math::ceil_div(reserved, kNumSMs / tail) > math::ceil_div(tail * reserved, kNumSMs);
    const uint32_t lane = threadIdx.x;
    auto all_splits = reinterpret_cast<KVSplit<capacity>*>(metadata + sizeof(MetadataHeader));
    auto row_splits = all_splits + row * reserved;
    auto headers = reinterpret_cast<KVSplitHeader*>(metadata + sizeof(MetadataHeader));
    auto schedule = kHeadersOnly ? reinterpret_cast<ScheduleEntry*>(headers + rows * reserved) :
        reinterpret_cast<ScheduleEntry*>(all_splits + rows * reserved);
    const auto indices = candidates + static_cast<uint64_t>(row) * NUM_MAX_SPARSE_BLOCKS;

    if (row == 0 && lane == 0)
        *reinterpret_cast<MetadataHeader*>(metadata) = {rows * reserved, waves + split_tail, kHeadersOnly ? 3u : 1u, kNumSMs};
    if (lane == 0 && candidate_valid)
        candidate_valid[row] = count == 0 ? 0 : (count - 1) * SPARSE_BLOCK_KV +
            cute::min(SPARSE_BLOCK_KV, length - indices[count - 1] * SPARSE_BLOCK_KV);

    const uint32_t cohort = row / kNumSMs * kNumSMs;
    const uint32_t active = cute::min(kNumSMs, rows - cohort);
    const uint32_t query_desc = kHeadersOnly ? 1u | (count << 16) : 1u;
    if (active < kNumSMs && split_tail) {
        for (uint32_t sm = lane; sm < kNumSMs; sm += kNumThreads) {
            const uint32_t begin_slot = sm * (active * reserved) / kNumSMs;
            const uint32_t end_slot = (sm + 1) * (active * reserved) / kNumSMs;
            const uint32_t first_row = cohort + begin_slot / reserved;
            const uint32_t last_row = cohort + (end_slot - 1) / reserved;
            if (row == first_row) {
                const uint32_t first = cute::min(begin_slot % reserved, splits);
                const uint32_t last = first_row == last_row ? cute::min(end_slot - (row - cohort) * reserved, splits) : splits;
                schedule[cohort + sm] = ScheduleEntry(row * reserved + first, row * reserved + last, row, query_desc);
                if (first_row == last_row)
                    schedule[cohort + kNumSMs + sm] = ScheduleEntry(0, 0, 0, 0);
            } else if (row == last_row) {
                const uint32_t last = cute::min(end_slot - (row - cohort) * reserved, splits);
                schedule[cohort + kNumSMs + sm] = ScheduleEntry(row * reserved, row * reserved + last, row, query_desc);
            }
        }
    } else {
        const uint32_t first = (row % kNumSMs) * kNumSMs / active;
        const uint32_t owners = ((row % kNumSMs + 1) * kNumSMs / active) - first;
        for (uint32_t owner = lane; owner < owners; owner += kNumThreads)
            schedule[cohort + first + owner] = ScheduleEntry(row * reserved + owner * splits / owners,
                row * reserved + (owner + 1) * splits / owners, row, query_desc);
    }

    for (uint32_t split = lane; split < splits; split += kNumThreads) {
        const uint32_t base = split * capacity;
        const uint32_t end = cute::min(count, base + capacity);
        if constexpr (kHeadersOnly) {
            const bool may_clip = end == count && num_kv_tokens - begin - length < SPARSE_BLOCK_KV;
            headers[row * reserved + split] = KVSplitHeader(row, end - base, false, may_clip, base, begin);
        } else {
            const uint32_t last = begin + indices[end - 1] * SPARSE_BLOCK_KV;
            row_splits[split].header = KVSplitHeader(row, end - base, false,
                num_kv_tokens - last < SPARSE_BLOCK_KV, base, ~0u);
        }
    }
    if constexpr (kHeadersOnly) return;
    for (uint32_t slot = 2 * lane; slot < splits * capacity; slot += 2 * kNumThreads) {
        const uint32_t base = slot % capacity;
        const uint32_t physical0 = slot < count ? begin + indices[slot] * SPARSE_BLOCK_KV : 0;
        const uint32_t physical1 = slot + 1 < count ? begin + indices[slot + 1] * SPARSE_BLOCK_KV : 0;
        const uint32_t absent = kInvalidSparseSlot << kNumSparseSlotBits;
        *reinterpret_cast<uint4*>(row_splits[slot / capacity].kv_block_infos + base) =
            make_uint4(physical0, slot < count ? absent | base : ~0u,
                       physical1, slot + 1 < count ? absent | (base + 1) : ~0u);
    }
}

} // namespace deep_gemm::sched::sparse_mqa_logits
