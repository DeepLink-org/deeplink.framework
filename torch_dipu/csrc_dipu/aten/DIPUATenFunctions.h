#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace dipu::native {

struct DIPUATenFunctions {

    // dipu native func
    static at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt);

    static at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt);

    static at::Tensor& copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking);

    static const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);

    // diopi func
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

    static at::Tensor& randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out);
    static at::Tensor& randperm_out(int64_t n, at::Tensor& result);

    static at::Tensor& random_(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator);
    static at::Tensor& random_(at::Tensor & self, c10::optional<at::Generator> generator);
    static at::Tensor& random_(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator);
    static at::Tensor& fillScalar_(at::Tensor & self, const at::Scalar & value);
    static at::Tensor& sum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out);

};

}  // namespace dipu::native
