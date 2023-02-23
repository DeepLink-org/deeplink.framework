# pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace dipu::native {

struct DIPUNativeFunctions {
    static at::Tensor& add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);
    static at::Tensor relu(const at::Tensor& self);
    static at::Tensor& relu_(at::Tensor& self);

    static ::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(
        const at::Tensor & input, const c10::optional<at::Tensor> & weight,
        const c10::optional<at::Tensor> & bias,
        const c10::optional<at::Tensor> & running_mean,
        const c10::optional<at::Tensor> & running_var,
        bool training, double momentum, double eps);
    static ::std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
        const at::Tensor &grad_out, const at::Tensor &input,
        const c10::optional<at::Tensor> &weight,
        const c10::optional<at::Tensor> &running_mean,
        const c10::optional<at::Tensor> &running_var,
        const c10::optional<at::Tensor> &save_mean,
        const c10::optional<at::Tensor> &save_invstd,
        bool train, double eps, ::std::array<bool, 3> output_mask);
    static at::Tensor conv2d(
        const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias,
        at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
};

}  // namespace dipu::native