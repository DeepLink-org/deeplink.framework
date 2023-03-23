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
    static ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
        const at::Tensor & input, const c10::optional<at::Tensor> & weight,
        const c10::optional<at::Tensor> & bias,
        const c10::optional<at::Tensor> & running_mean,
        const c10::optional<at::Tensor> & running_var,
        bool training, double momentum, double eps,
        at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd);
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
    static at::Tensor& mean_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out);
    static at::Tensor& addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out);
    static at::Tensor _adaptive_avg_pool2d(const at::Tensor & self, c10::SymIntArrayRef output_size);
    static at::Tensor& adaptive_avg_pool2d_out(const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out);
    static at::Tensor adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
    static at::Tensor linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias);
    static at::Tensor& _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);
    static at::Tensor& log_softmax_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out);
    static at::Tensor& _log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out);
    static at::Tensor cross_entropy_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing);
    static at::Tensor& nll_loss_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & out);
    static ::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight);
    static at::Tensor& nll_loss_backward_out_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input);
    static ::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices);
    static at::Tensor& max_pool2d_with_indices_backward_out_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input);
    static at::Tensor& mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
    static at::Tensor& div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);

};

}  // namespace dipu::native
