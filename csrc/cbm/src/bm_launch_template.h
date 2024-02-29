#pragma once
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_kernel.h"
#include "kernel_traits.h"

template<typename Kernel_traits>
__global__ void 
bm_kernel(__grid_constant__ const Bm_params params) {
    bm::do_sort<Kernel_traits>(params);
}

template<typename Kernel_traits>
void
run_bm_(const Bm_params &params, cudaStream_t stream) {
    int n_slices = DIV_ROUND_UP(params.in_batch_stride, Kernel_traits::blockN);
    dim3 grid(params.n_seq, n_slices);
    constexpr int smem_size = Kernel_traits::smemSize;

    auto kernel = &bm_kernel<Kernel_traits>;
    kernel<<<grid, Kernel_traits::nThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void
run_bm(const Bm_params& params, cudaStream_t stream) {
    run_bm_<Bm_kernel_traits<1024, 4>>(params, stream);
}
