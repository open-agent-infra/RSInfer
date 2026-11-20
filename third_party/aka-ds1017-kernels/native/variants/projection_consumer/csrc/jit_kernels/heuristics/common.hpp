#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

// L2 swizzle group size for the block scheduler. The default footprint heuristic balances the
// concurrent A/B working set, but for store-dominated normal GEMMs a larger group keeps the reused
// B operand resident in L2 across more m-tiles, cutting DRAM re-reads (measured ~2x B amplification
// at group=16 on M8192 N5120 K8192 and M4096 N36864 K1280). Bound the per-group A footprint to ~1/3
// of L2 so the streamed A plus the D flush do not evict B; cap at the number of m-tiles.
static int get_num_1d_blocks_per_group(const GemmDesc& desc, const Layout& layout) {
    const bool multicast_on_a = layout.cluster_n > 1;
    const auto default_group = [&]() {
        int best = 8, min_usage = std::numeric_limits<int>::max();
        for (const int candidate: {8, 16}) {
            const int usage = multicast_on_a ?
                candidate * layout.block_n + ((desc.num_sms + candidate - 1) / candidate) * layout.block_m :
                candidate * layout.block_m + ((desc.num_sms + candidate - 1) / candidate) * layout.block_n;
            if (usage < min_usage)
                min_usage = usage, best = candidate;
        }
        return best;
    };

    if (desc.gemm_type == GemmType::Normal and not layout.swap_ab) {
        const int l2_bytes = jit->device.get_num_l2_cache_bytes();
        const int a_bytes_per_mtile = layout.block_m * desc.k * static_cast<int>(c10::elementSize(desc.a_dtype));
        const int num_m_blocks = (desc.m + layout.block_m - 1) / layout.block_m;
        if (a_bytes_per_mtile > 0 and l2_bytes > 0) {
            int group = std::min(num_m_blocks, l2_bytes / (3 * a_bytes_per_mtile));
            int pow2 = 8;
            while (pow2 * 2 <= group)
                pow2 *= 2;
            group = std::max(pow2, 8);
            const int cluster = layout.get_cluster_size();
            group = (group / cluster) * cluster;
            if (group >= 8)
                return group;
        }
    }
    return default_group();
}

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }

    layout.num_1d_blocks_per_group = get_num_1d_blocks_per_group(desc, layout);
    if (use_compact_nt_raster(desc, layout))
        layout.num_1d_blocks_per_group = 4;

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        std::stringstream ss;
        ss << desc;
        const auto key = ss.str();

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm
