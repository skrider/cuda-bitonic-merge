#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>

int ilog2(int x) {
    return sizeof(int)*8 - 1 - __builtin_clz(x);
}

#define PRINT_NUMPY(t)      \
    cute::print("\n<NUMPY>\n");   \
    cute::print(#t " = ");        \
    print_numpy(t);         \
    cute::print("\n</NUMPY>\n");
#define PRINT_NUMPY_VAL(t)  \
    cute::print("\n<NUMPY>\n");   \
    cute::print(#t " = ");        \
    cute::print(t);               \
    cute::print("\n</NUMPY>\n");
#define PRINT_NUMPY_STR(t)  \
    cute::print("\n<NUMPY>\n");   \
    cute::print(#t " = ");        \
    cute::print(t);               \
    cute::print("\n</NUMPY>\n");
#define PRINT_NUMPY_BARRIER() cute::print("\n<BARRIER>\n");

template<class Engine, class Layout>
__host__ __device__ void
print_numpy(cute::Tensor<Engine,Layout> const& t)
{
    using namespace cute;
    print("np.array([");
    for (int i = 0; i < t.size(); i++) {
            print(t(i));
            if (i < t.size() - 1) {
                    print(", ");
            }
            if (i % 10 == 9) {
                    print("\n");
            }
    }
    print("])");
    print(".reshape(");
    print(transform_apply(
            t.layout().shape(),
            // add 0 to force runtime eval and get rid of the underscore
            [](auto const& m) { return size(m) + 0; },
            [](auto const&... v) { return make_shape(v...); }
    ));
    print(", order='F')");
}

namespace bm {

template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

} // namespace bm
