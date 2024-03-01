#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_launch_template.h"
#include "utils.h"

void
sort(at::Tensor &t, const int dim) {
    TORCH_CHECK(t.is_cuda(), "input tensor must be on CUDA");
    TORCH_CHECK(t.dtype() == at::ScalarType::Short, "input tensor must be short");
    TORCH_CHECK(t.dim() == 2, "input tensor must be 2D");
    TORCH_CHECK(t.stride(1) == 1, "input tensor must be contiguous");
    TORCH_CHECK(dim == 1, "only support dim 1")

    int seqlen = t.size(dim);
    TORCH_CHECK((1 << ilog2(seqlen)) == seqlen, "seqlen must be power of 2");
    TORCH_CHECK(seqlen >= 4096, "seqlen must be at least 4096");

    Bm_params params;
    params.in_ptr = reinterpret_cast<void*>(t.data_ptr());
    params.in_batch_stride = t.stride(0);
    params.seqlen = seqlen;
    params.n_seq = t.size(0);
    params.k_start = 2;
    params.k_end = seqlen * 2;

    if (t.dtype() == at::ScalarType::Half) {
        params.dtype = BM_HALF;
    } else if (t.dtype() == at::ScalarType::BFloat16) {
        params.dtype = BM_BFLOAT16;
    } else if (t.dtype() == at::ScalarType::Short) {
        params.dtype = BM_INT16;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    run_bm(params, stream);
    stream.synchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cbm";
    m.def("sort", &sort, "sort tensor along dim");
}
