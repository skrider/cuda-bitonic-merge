#pragma once
#include <cutlass/numeric_types.h>

template<int blockN_, int nWarps_, typename elem_type=cutlass::half_t>
struct Bm_kernel_traits {
    static constexpr int blockN = blockN_;
    static constexpr int nWarps = nWarps_;
    static constexpr int nThreads = nWarps_ * 32;
    using Element = elem_type;
    using index_t = int64_t;
};
