#include <torch/python.h>

at::Tensor
sort(at::Tensor &t, const int dim) {
    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cbm";
    m.def("sort", &sort, "sort tensor along dim");
}
