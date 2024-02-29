#pragma once
#include <cutlass/numeric_types.h>
#include <cute/layout.hpp>

using namespace cute;

template<int blockN_, int nWarps_, typename elem_type=cutlass::half_t>
struct Bm_kernel_traits {
    static constexpr int blockN = blockN_;
    static constexpr int nWarps = nWarps_;
    static constexpr int nThreads = nWarps_ * 32;

    using Element = elem_type;
    using index_t = int64_t;

    static constexpr int gmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);

    using SmemLayout = Layout<Shape<Int<blockN>>>;

    static constexpr int smemSize = sizeof(Element) * size(SmemLayout{});

    using GmemThreadLayout = Layout<Shape<Int<nThreads>>>;
    // TODO we may not want to disable caching
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemThreadLayout{},
                        Layout<Shape<Int<gmemElemsPerLoad>>>{}));
};
