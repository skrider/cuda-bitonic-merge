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

    static constexpr int sortElemsPerThread = blockN / nThreads;
    static constexpr int gmemElemsPerLoad = MIN(sizeof(cute::uint128_t) / sizeof(Element), sortElemsPerThread);

    using SmemLayout = Layout<Shape<Int<blockN>>>;

    static constexpr int smemSize = sizeof(Element) * size(SmemLayout{});

    using GmemThreadLayout = Layout<Shape<Int<nThreads>>>;
    // TODO we may not want to disable caching
    // using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using Gmem_copy_struct = DefaultCopy;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemThreadLayout{},
                        Layout<Shape<Int<gmemElemsPerLoad>>>{}));

    using SortTileShape = Shape<Int<sortElemsPerThread>>;
};

template<typename Kernel_traits>
void
__device__ __host__
print_kernel_traits() {
    printf("blockN: %d\n", Kernel_traits::blockN);
    printf("nWarps: %d\n", Kernel_traits::nWarps);
    printf("nThreads: %d\n", Kernel_traits::nThreads);
    printf("sortElemsPerThread: %d\n", Kernel_traits::sortElemsPerThread);
    printf("gmemElemsPerLoad: %d\n", Kernel_traits::gmemElemsPerLoad);
    printf("smemSize: %d\n", Kernel_traits::smemSize);
}
