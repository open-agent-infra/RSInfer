#include <torch/python.h>
#include "csrc/apis/config.hpp"
#include "csrc/apis/gemm.hpp"

using Operand = std::pair<torch::Tensor, torch::Tensor>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    deep_gemm::config::register_apis(m);
    m.def("warm_jit", [] { (void)deep_gemm::jit.get(); });
    m.def("run", [](const Operand& a, const Operand& b, const torch::Tensor& out) {
        deep_gemm::gemm::fp8_fp4_gemm_nt(a, b, out, std::nullopt,
            std::make_tuple(1, 1, 32), std::nullopt, std::nullopt, "nk",
            false, std::nullopt, nullptr);
    });
}
