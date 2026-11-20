#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <optional>

extern "C" int paged(void**, const int*, double, void*);
using at::Tensor;
using OptionalTensor = std::optional<Tensor>;

static std::tuple<Tensor, Tensor, Tensor> run(
    const Tensor& q, const Tensor& k, const Tensor& idx,
    const OptionalTensor& count, const OptionalTensor& sink,
    const OptionalTensor& extra, const OptionalTensor& extra_idx, const OptionalTensor& extra_count,
    const Tensor& positions, const Tensor& rope_cache, double scale) {
  c10::cuda::CUDAGuard guard(q.device());
  const auto n = q.size(0);
  TORCH_CHECK(q.dim()==3 && q.size(1)==64 && q.size(2)==512 && k.dim()==4 && k.size(-1)==528);
  TORCH_CHECK(!extra || (extra->dim()==4 && (extra->size(-1)==288 || extra->size(-1)==528)));
  const auto aligned = (n + 3) / 4 * 4;
  auto out = at::empty({n,8,4096}, q.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty_strided({n,8,32},{1,32*aligned,aligned}, q.options().dtype(at::kInt));
  auto lse = at::empty({n,64}, q.options().dtype(at::kFloat));
  auto ptr = [](const OptionalTensor& t) -> void* { return t ? t->data_ptr() : nullptr; };
  void* pointers[] = {q.data_ptr(),k.data_ptr(),idx.data_ptr(),ptr(count),ptr(sink),lse.data_ptr(),
                      ptr(extra),ptr(extra_idx),ptr(extra_count),positions.data_ptr(),
                      rope_cache.data_ptr(),out.data_ptr(),scales.data_ptr()};
  const int64_t dims[] = {n,k.size(0),k.size(1),idx.size(1),extra ? extra->size(0):0,
                          extra ? extra->size(1):0,extra_idx ? extra_idx->size(1):0,
                          q.stride(0),k.stride(0),k.stride(1),idx.stride(0),
                          extra ? extra->stride(0):0,extra ? extra->stride(1):0,
                          extra ? extra->size(3):0,extra_idx ? extra_idx->stride(0):0,aligned};
  int values[16];
  for (int i=0;i<16;++i) {
    TORCH_CHECK(dims[i]>=0 && dims[i]<=std::numeric_limits<int>::max());
    values[i]=static_cast<int>(dims[i]);
  }
  const auto stream=c10::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(paged(pointers,values,scale,stream.stream())==0,"paged MLA launch failed");
  return {out,scales,lse};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
