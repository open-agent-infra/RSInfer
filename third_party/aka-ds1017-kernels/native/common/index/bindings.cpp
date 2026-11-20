#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>
#include "csrc/apis/config.hpp"
#include "csrc/apis/epilogue_class.hpp"
#include "csrc/apis/attention.hpp"
#include "csrc/apis/layout.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
deep_jit::register_python_api(m,deep_gemm::jit);
deep_gemm::config::register_apis(m);
deep_gemm::epilogue_class::register_apis(m);
deep_gemm::attention::register_apis(m);
deep_gemm::layout::register_apis(m);
}
