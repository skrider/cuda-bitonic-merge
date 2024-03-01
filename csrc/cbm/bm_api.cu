#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_launch_template.h"
#include "schedule.h"

void
sort(at::Tensor &t, const int dim) {
    TORCH_CHECK(t.is_cuda(), "input tensor must be on CUDA");
    TORCH_CHECK(t.dtype() == at::ScalarType::Half, "input tensor must be half");
    TORCH_CHECK(t.dim() == 2, "input tensor must be 2D");
    TORCH_CHECK(t.stride(1) == 1, "input tensor must be contiguous");
    TORCH_CHECK(dim == 1, "only dim 1 is supported for now")

    int seq_len = t.size(1);
    int block_size = 4096;
    int seq_len_log2 = std::log2(seq_len);
    int block_size_log2 = std::log2(block_size);

    auto schedule = get_schedule(seq_len_log2, block_size_log2);

    Bm_params params;
    params.in_ptr = reinterpret_cast<void*>(t.data_ptr());
    params.in_batch_stride = t.stride(0);
    params.n_seq = t.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();

    for (int i = 0; i < schedule.size(); i++) {
        auto launch = schedule[i];
        bool is_last = i == schedule.size() - 1;
        bool is_gmem = launch[4] == 1;

        params.k_start = launch[0];
        params.k_end = launch[1];
        params.j_start = launch[2];
        params.j_end = launch[3];
        params.end_inclusive = is_last;

        run_bm(params, is_gmem, stream);
    }
    
    stream.synchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cbm";
    m.def("sort", &sort, "sort tensor along dim");
}
