#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::Half f16;

void cuda_forward(int B, int T, int C, int H, f16 *r, f16 *k, f16 *v, float *w, f16 *u, f16 *y);
void cuda_backward(int B, int T, int C, int H, f16 *r, f16 *k, f16 *v, float *w, f16 *u, f16 *gy, f16 *gr, f16 *gk, f16 *gv, f16 *gw, f16 *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y)
{
    cuda_forward(B, T, C, H, r.data_ptr<f16>(), k.data_ptr<f16>(), v.data_ptr<f16>(), w.data_ptr<float>(), u.data_ptr<f16>(), y.data_ptr<f16>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu)
{
    cuda_backward(B, T, C, H, r.data_ptr<f16>(), k.data_ptr<f16>(), v.data_ptr<f16>(), w.data_ptr<float>(), u.data_ptr<f16>(), gy.data_ptr<f16>(), gr.data_ptr<f16>(), gk.data_ptr<f16>(), gv.data_ptr<f16>(), gw.data_ptr<f16>(), gu.data_ptr<f16>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward, "wkv6 forward");
    m.def("backward", &backward, "wkv6 backward");
}

TORCH_LIBRARY(wkv6, m)
{
    m.def("forward", forward);
    m.def("backward", backward);
}