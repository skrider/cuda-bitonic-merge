#pragma once
#include <cuda.h>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct Bm_params {
    using index_t = int64_t;

    void *__restrict__ in_ptr;
    void *__restrict__ in_index_ptr;
    index_t in_batch_stride;
    index_t n_seq;

    index_t k_start;
    index_t k_end;
    
    index_t j_start;
    index_t j_end;

    bool end_inclusive;
};