#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_launch_template.h"
#include "utils.h"
#include "schedule.h"
#include <unistd.h>

void
bm_sort(at::Tensor &t, const int dim) {
    TORCH_CHECK(t.is_cuda(), "input tensor must be on CUDA");
    TORCH_CHECK(t.dtype() == at::ScalarType::Short, "input tensor must be short");
    TORCH_CHECK(t.dim() == 2, "input tensor must be 2D");
    TORCH_CHECK(t.stride(1) == 1, "input tensor must be contiguous");
    TORCH_CHECK(dim == 1, "only support dim 1")

    int seqlen = t.size(dim);
    TORCH_CHECK((1 << ilog2(seqlen)) == seqlen, "seqlen must be power of 2");
    TORCH_CHECK(seqlen >= (1 << 9), "seqlen too low");
    TORCH_CHECK(dim == 1, "only dim 1 is supported for now")

    int seq_len = t.size(1);
    int block_size;
    SEQLEN_SWITCH(seq_len, [&]() { block_size = blockN; });
    
    int seq_len_log2 = std::log2(seq_len);
    int block_size_log2 = std::log2(block_size);

    auto schedule = get_schedule(seq_len_log2, block_size_log2);
    
    Bm_params params;
    params.in_ptr = reinterpret_cast<void*>(t.data_ptr());
    params.in_batch_stride = t.stride(0);
    params.seqlen = seqlen;
    params.n_seq = t.size(0);
    
    if (t.dtype() == at::ScalarType::Half) {
        params.dtype = BM_HALF;
    } else if (t.dtype() == at::ScalarType::BFloat16) {
        params.dtype = BM_BFLOAT16;
    } else if (t.dtype() == at::ScalarType::Short) {
        params.dtype = BM_INT16;
    }

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
        usleep(7000);

        stream.synchronize();
    }
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cbm";
    m.def("sort", &bm_sort, "sort tensor along dim");
}
