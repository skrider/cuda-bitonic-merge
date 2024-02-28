#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_launch_template.h"

void
sort(at::Tensor &t, const int dim) {
    TORCH_CHECK(t.is_cuda(), "input tensor must be on CUDA");
    TORCH_CHECK(t.dtype() == at::ScalarType::Half, "input tensor must be half");
    TORCH_CHECK(t.dim() == 2, "input tensor must be 2D");
    TORCH_CHECK(t.stride(1) == 1, "input tensor must be contiguous");

    Bm_params params;
    params.in_ptr = reinterpret_cast<void*>(t.data_ptr());
    params.in_batch_stride = t.stride(0);
    params.n_seq = t.size(0);

    run_bm(params, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cbm";
    m.def("sort", &sort, "sort tensor along dim");
}
