#pragma once
#include <cuda.h>

struct Bm_params {
    using index_t = int64_t;

    void *__restrict__ in_ptr;
    index_t in_batch_stride;
    index_t n_seq;
};