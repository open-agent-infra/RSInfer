#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>
#include "csrc/apis/config.hpp"
#include "csrc/apis/epilogue_class.hpp"
#include "csrc/apis/gemm.hpp"
#include "csrc/apis/einsum.hpp"
#include "csrc/apis/layout.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
deep_jit::register_python_api(m,deep_gemm::jit);
deep_gemm::config::register_apis(m);
deep_gemm::epilogue_class::register_apis(m);
deep_gemm::gemm::register_apis(m);
deep_gemm::einsum::register_apis(m);
deep_gemm::layout::register_apis(m);

using Operand=std::pair<torch::Tensor,torch::Tensor>;
m.def("agx_gemm", [](const Operand& a,const Operand& b,const torch::Tensor& out) {
    deep_gemm::gemm::fp8_fp4_gemm_nt(a,b,out,std::nullopt,std::make_tuple(1,1,32),std::nullopt,std::nullopt,"nk",false,std::nullopt,nullptr);
});
m.def("agx_wo", [](const Operand& a,const Operand& b,const Operand& out) {
    deep_gemm::einsum::fp8_einsum("bhr,hdr->bhd",a,b,out,std::nullopt,std::make_tuple(1,1,32),nullptr);
});
}
