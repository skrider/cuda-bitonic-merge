#pragma once
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_kernel.h"
#include "kernel_traits.h"

template<typename Kernel_traits, bool isSmem_>
__global__ void 
bm_kernel(__grid_constant__ const Bm_params params) {
    bm::do_sort<Kernel_traits, isSmem_>(params);
}

template<typename Kernel_traits, bool isSmem_>
void
run_bm_(const Bm_params &params, cudaStream_t stream) {
    int n_slices = DIV_ROUND_UP(params.in_batch_stride, Kernel_traits::blockN);
    dim3 grid(params.n_seq, n_slices);
    constexpr bool isSmem = isSmem_;
    constexpr int smem_size = isSmem ? Kernel_traits::smemSize : 0;

    auto kernel = &bm_kernel<Kernel_traits, isSmem_>;
    
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, Kernel_traits::nThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void
run_bm(const Bm_params& params, const bool is_gmem, cudaStream_t stream) {
    if (is_gmem) {
        run_bm_<Bm_kernel_traits<4096, 16>, false>(params, stream);
    } else {
        run_bm_<Bm_kernel_traits<4096, 16>, true>(params, stream);
    }
}
