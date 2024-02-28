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
    dim3 grid(params.n_seq);

    auto kernel = &bm_kernel<Kernel_traits>;
    kernel<<<grid, Kernel_traits::nThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void
run_bm(const Bm_params& params, cudaStream_t stream) {
    run_bm_<Bm_kernel_traits<128, 4>>(params, stream);
}
