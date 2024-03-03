#pragma once
#include <ATen/cuda/CUDAContext.h>
#include "bm.h"
#include "bm_kernel.h"
#include "kernel_traits.h"
#include "static_switch.h"

template<typename Kernel_traits, bool isGmem_>
__global__ void 
bm_kernel(__grid_constant__ const Bm_params params) {
    bm::do_sort<Kernel_traits, isGmem_>(params);
}

template<typename Kernel_traits, bool isGmem_>
void
run_bm_(const Bm_params &params, cudaStream_t stream) {
    int n_slices = DIV_ROUND_UP(params.seqlen, Kernel_traits::blockN);
    dim3 grid(params.n_seq, n_slices);
    constexpr bool isGmem = isGmem_;
    constexpr int smem_size = isGmem ? 0 : Kernel_traits::smemSize;

    auto kernel = &bm_kernel<Kernel_traits, isGmem>;
    
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, Kernel_traits::nThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void
run_bm(const Bm_params& params, const bool is_gmem, cudaStream_t stream) {
    DTYPE_SWITCH(params.dtype, [&]() {
        SEQLEN_SWITCH(params.seqlen, [&]() {
            BOOL_SWITCH(is_gmem, isGmem, [&]() {
                run_bm_<Bm_kernel_traits<blockN, nWarps, Element>, isGmem>(params, stream);
            });
        });
    });
}
