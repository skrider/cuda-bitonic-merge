#pragma once
#include <cute/algorithm/fill.hpp>
#include <cute/tensor.hpp>

#include "bm.h"

namespace bm {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
for (k = 2; k <= n; k *= 2) // kernel iteration
    for (j = k/2; j > 0; j /= 2) // inside block
        for (i = 0; i < n; i++) // grid
            l = bitwiseXOR (i, j); // in C-like languages this is "i ^ j"
            if (l > i)
                if (  (bitwiseAND (i, k) == 0) AND (arr[i] > arr[l])
                   OR (bitwiseAND (i, k) != 0) AND (arr[i] < arr[l]) )
                      swap the elements arr[i] and arr[l]

for (k = 2; k <= n; k *= 2) // kernel iteration
    for (j = k/2; j > 0; j /= 2) // inside block
        for (i = 0; i < n; i+= blockN) // grid
            for (ii = 0; ii < blockN; ii++) // thread
                idx = i + ii;
                l = bitwiseXOR (idx, j); // in C-like languages this is "i ^ j"
                if (l > idx)
                    if (  (bitwiseAND (idx, k) == 0) AND (arr[idx] > arr[l])
                       OR (bitwiseAND (idx, k) != 0) AND (arr[idx] < arr[l]) )
                          swap the elements arr[idx] and arr[l]
*/

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void
bm_atom(Tensor<Engine0, Layout0> src, 
        const Tensor<Engine1, Layout1> idx, 
        const int j,
        const int k,
        const int offset) {
    Tensor src_offset = make_tensor(src.data() + offset, src.shape(), src.stride());

#pragma unroll 
    for (int ii = 0; ii < size(idx); ii++) {
        int i = idx(ii);
        int l = i ^ j;

        if (l > i) {
            if (  (((i & k) == 0) && (src_offset(i) > src_offset(l)))
               || (((i & k) != 0) && (src_offset(i) < src_offset(l)))) {
                auto tmp = src_offset(l);
                src_offset(l) = src_offset(i);
                src_offset(i) = tmp;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
sort_row_slice(const Bm_params &params, const int row, const int slice) {
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    constexpr int blockN = Kernel_traits::blockN;

    int slice_offset = slice * blockN;
    int row_offset = row * params.in_batch_stride + slice_offset;
    int tid = threadIdx.x;

    extern __shared__ char smem_[];

    Tensor gT = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.in_ptr) + row_offset),
        make_shape(Int<blockN>{}),
        make_stride(_1{})
    );
    Tensor sT = make_tensor(
        make_smem_ptr(reinterpret_cast<Element *>(smem_)),
        typename Kernel_traits::SmemLayout{}
    );
    Tensor cT = make_identity_tensor(Shape<Int<blockN>>{});

    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tid);
    Tensor tTgT = gmem_thr_copy.partition_S(gT);
    Tensor tTsT = gmem_thr_copy.partition_D(sT);

    Tensor tTcT = local_tile(cT, Kernel_traits::SortTileShape(), make_coord(tid));
    Tensor tI = make_tensor<int>(tTcT.shape());

    copy(tTgT, tTsT);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < size(tTcT); i++) {
        tI(i) = get<0>(tTcT(i)) + slice_offset;
    }

    for (int k = params.k_start; k < params.k_end; k *= 2) {
#pragma unroll
        for (int j = k/2; j > 0; j /= 2) {
            bm::bm_atom(sT, tI, j, k, -1 * slice_offset);
            __syncthreads();
        }
    }
    copy(tTsT, tTgT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
do_sort(const Bm_params &params) {
    const int row = blockIdx.x;
    const int slice = blockIdx.y;
    bm::sort_row_slice<Kernel_traits>(params, row, slice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
    
} // namespace bm

