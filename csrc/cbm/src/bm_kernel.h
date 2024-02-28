#pragma once
#include <cute/algorithm/fill.hpp>
#include <cute/tensor.hpp>

#include "bm.h"

namespace bm {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
sort_row(const Bm_params &params, int row) {
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    constexpr int blockN = Kernel_traits::blockN;
    constexpr int nWarps = Kernel_traits::nWarps;

    int row_offset = row * params.in_batch_stride;
    Tensor gT = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.in_ptr) + row_offset),
        make_shape(params.in_batch_stride),
        make_stride(_1{})
    );

    fill(gT, 1.f * row);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
do_sort(const Bm_params &params) {
    const int row = blockIdx.x;
    bm::sort_row<Kernel_traits>(params, row);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
    
} // namespace bm

