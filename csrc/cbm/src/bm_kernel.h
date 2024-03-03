#pragma once
#include <cute/algorithm/fill.hpp>
#include <cute/tensor.hpp>

#include "utils.h"
#include "bm.h"
#include "kernel_traits.h"

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

template<bool inGmem_=false, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void
bm_atom(Tensor<Engine0, Layout0> src, 
        const Tensor<Engine1, Layout1> idx, 
        const int j,
        const int k,
        const int offset) {
    Tensor src_offset = make_tensor(src.data() + offset, src.shape(), src.stride());
    constexpr int n_elem = idx.size();
    Tensor rSrc = make_tensor<Engine0::value_type>(make_shape(_2{}, Int<n_elem>{}));
    Tensor rSwap = make_tensor<bool>(make_shape(Int<n_elem>{}));
    constexpr bool inGmem = false;

#pragma unroll
    for (int ii = 0; ii < n_elem; ii++) {
        int i = idx(ii);
        int l = i ^ j;
        rSrc(0, ii) = src_offset(i);
        rSrc(1, ii) = src_offset(l);
    }

#pragma unroll  
    for (int ii = 0; ii < n_elem; ii++) {
        int i = idx(ii);
        rSwap(ii) = ((((i & k) == 0) && (rSrc(0, ii) > rSrc(1, ii)))
           || (((i & k) != 0) && (rSrc(0, ii) < rSrc(1, ii))));
    }

    // atomic swap is required
    if constexpr (!inGmem) {
        __syncthreads();
    }

#pragma unroll  
    for (int ii = 0; ii < n_elem; ii++) {
        int i = idx(ii);
        int l = i ^ j;
        // TODO statically figure out how many are skipped to save on regs
        if (l > i) {
            src_offset(i) = rSrc(rSwap(ii) ? 1 : 0, ii);
            src_offset(l) = rSrc(rSwap(ii) ? 0 : 1, ii);
        }
    }
    
    if constexpr (!inGmem) {
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
sort_row_slice_smem(const Bm_params &params, const int row, const int slice) {
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
    Tensor tI = make_tensor<int>(Kernel_traits::SortTileShape());

    copy(gmem_tiled_copy, tTgT, tTsT);
    cute::cp_async_fence();

    for (int i = 0; i < size(tTcT); i++) {
        tI(i) = get<0>(tTcT(i)) + slice_offset;
    }

    cute::cp_async_wait<0>();

    int j = params.j_start;
    for (int k = params.k_start; k <= params.k_end; k *= 2, j = k / 2) {
#pragma unroll
        for (; j > 0; j /= 2) {
            if (k == params.k_end && !params.end_inclusive && j == params.j_end) {
                break;
            }
            bm::bm_atom(sT, tI, j, k, -1 * slice_offset);
        }
    }

    copy(tTsT, tTgT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
inline __device__ void 
sort_row_slice(const Bm_params &params, const int row, const int slice) {
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    constexpr int blockN = Kernel_traits::blockN;
    constexpr int nThreads = Kernel_traits::nThreads;

    // will always be zero
    extern __shared__ char smem_[];

    int slice_offset = slice * blockN;
    int thread_offset = slice * nThreads;
    int row_offset = row * params.in_batch_stride;
    int tid = threadIdx.x;

    Tensor gT = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.in_ptr) + row_offset),
        make_shape(params.seqlen)
    );
    Tensor cT = make_identity_tensor(make_shape(params.seqlen));
    Tensor tTcT = local_tile(cT, Kernel_traits::SortTileShape(), make_coord(thread_offset + tid));
    Tensor tI = make_tensor<int>(tTcT.shape());

#pragma unroll
    for (int i = 0; i < size(tTcT); i++) {
        tI(i) = get<0>(tTcT(i));
    }

    bm::bm_atom<true>(gT, tI, params.j_start, params.k_start, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool isGmem_>
inline __device__ void 
do_sort(const Bm_params &params) {
    constexpr bool isGmem = isGmem_;
    const int row = blockIdx.x;
    const int slice = blockIdx.y;

    if constexpr (isGmem) {
        bm::sort_row_slice<Kernel_traits>(params, row, slice);
    } else {
        bm::sort_row_slice_smem<Kernel_traits>(params, row, slice);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
    
} // namespace bm

