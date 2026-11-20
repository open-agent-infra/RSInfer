#include <torch/python.h>
#include "csrc/apis/config.hpp"
#include "csrc/apis/mega_mhc.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 deep_gemm::config::register_apis(m);
 m.def("warm_jit", [] { (void)deep_gemm::jit.get(); });
 deep_gemm::mega_mhc::register_apis(m);
}
