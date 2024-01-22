// Copyright (c) 2023, DeepLink.
#pragma once

#include <algorithm>
#include <iterator>

#include "csrc_dipu/aten/RegisterDIPU.hpp"

#include "OpUtils.hpp"

namespace dipu {
namespace native {

static c10::optional<at::Tensor> dipu_to_cpu(
    const c10::optional<at::Tensor>& device_tensor) {
  c10::optional<at::Tensor> cpu_tensor = c10::nullopt;
  if (device_tensor.has_value() && device_tensor.value().defined()) {
    cpu_tensor = device_tensor.value().cpu();
  }
  return cpu_tensor;
}

static at::Tensor to_cpu_with_half_to_float(const at::Tensor& devtensor) {
  auto cpu_tensor = devtensor.cpu();
  auto intype = devtensor.options().dtype_opt()->toScalarType();
  if (intype == at::ScalarType::Half) {
    return cpu_tensor.to(at::ScalarType::Float);
  }
  return cpu_tensor;
}

static at::Tensor& custom_fallback_dipu_silu_out(const at::Tensor& self,
                                                 at::Tensor& out) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=silu_out"
                           << std::endl);
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto out_cpu = to_cpu_with_half_to_float(out);

  out_cpu = at::silu_out(out_cpu, self_cpu);
  out.copy_(out_cpu);
  return out;
}

static c10::List<c10::optional<at::Tensor>> to_cpu(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  c10::List<c10::optional<at::Tensor>> indices_cpu;
  indices_cpu.reserve(indices.size());
  // input as x[1:2, [1, 2]], Slice by first dimension already executed before
  // this index(), in this case, indices[0] is an undefinedTensor.
  std::transform(
      indices.begin(), indices.end(), std::back_inserter(indices_cpu),
      [](const c10::optional<at::Tensor>& optional_tensor) {
        return optional_tensor.has_value() && optional_tensor.value().defined()
                   ? optional_tensor.value().to("cpu")
                   : at::Tensor();
      });
  return indices_cpu;
}
static at::Tensor& custom_fallback_dipu_index_tensor_out(
    const at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    at::Tensor& out) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=index.Tensor_out"
                           << std::endl);
  auto indices_cpu = to_cpu(indices);

  at::Tensor out_cpu = out.cpu();
  at::index_outf(self.cpu(), indices_cpu, out_cpu);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor& custom_fallback_dipu__index_put_impl_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool unsafe) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=_index_put_impl_"
                           << std::endl);

  auto indices_cpu = to_cpu(indices);
  at::Tensor self_cpu = self.cpu();
  at::native::_index_put_impl_(self_cpu, indices_cpu, values.cpu(), accumulate,
                               unsafe);
  self.copy_(self_cpu);

  return self;
}

static ::std::tuple<at::Tensor, at::Tensor, at::Tensor>
custom_fallback_dipu_native_batch_norm_out(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt, bool training,
    double momentum, double eps, at::Tensor& out, at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=native_batch_norm_out"
                           << std::endl);
  at::Tensor input_cpu = input.cpu();
  at::Tensor out_cpu = out.cpu();
  at::Tensor save_mean_cpu = save_mean.cpu();
  at::Tensor save_invstd_cpu = save_invstd.cpu();

  c10::optional<at::Tensor> weight_cpu = dipu_to_cpu(weight_opt);
  c10::optional<at::Tensor> bias_cpu = dipu_to_cpu(bias_opt);
  c10::optional<at::Tensor> running_mean_cpu = dipu_to_cpu(running_mean_opt);
  c10::optional<at::Tensor> running_var_cpu = dipu_to_cpu(running_var_opt);

  at::native_batch_norm_out(out_cpu, save_mean_cpu, save_invstd_cpu, input_cpu,
                            weight_cpu, bias_cpu, running_mean_cpu,
                            running_var_cpu, training, momentum, eps);

  out.copy_(out_cpu);
  save_mean.copy_(save_mean_cpu);
  save_invstd.copy_(save_invstd_cpu);

  if (running_mean_opt.has_value() && running_mean_opt.value().defined()) {
    running_mean_opt.value().copy_(running_mean_cpu.value());
  }
  if (running_var_opt.has_value() && running_var_opt.value().defined()) {
    running_var_opt.value().copy_(running_var_cpu.value());
  }

  return {out, save_mean, save_invstd};
}

static at::Tensor custom_fallback_dipu_convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  DIPU_OP_LOG_WARNING_ONCE(
      "custom fallback to cpu, name=convolution_overrideable" << std::endl);
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto bias_cpu = dipu_to_cpu(bias);
  auto result =
      at::convolution(input_cpu, weight_cpu, bias_cpu, stride, padding,
                      dilation, transposed, output_padding, groups);
  return result.to(input.device());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
custom_fallback_dipu_convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, ::std::array<bool, 3> output_mask) {
  DIPU_OP_LOG_WARNING_ONCE(
      "custom fallback to cpu, name=convolution_backward_overrideable"
      << std::endl);
  auto device = input.device();
  auto grad_output_cpu = grad_output.cpu();
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto output_mask_temp = output_mask;
  output_mask_temp[2] = false;
  auto result = at::convolution_backward(
      grad_output_cpu, input_cpu, weight_cpu, c10::nullopt, stride, padding,
      dilation, transposed, output_padding, groups, output_mask_temp);

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = std::get<0>(result).to(device);
  }
  if (output_mask[1]) {
    grad_weight = std::get<1>(result).to(device);
  }

  if (output_mask[2]) {
    std::vector<int64_t> bias_sizes{grad_output.size(1)};
    at::Tensor grad_bias_cpu = at::empty(
        bias_sizes, grad_output.options().device(c10::DeviceType::CPU));
    grad_bias = at::empty(bias_sizes, grad_output.options());
    at::Tensor at_tmp = grad_output_cpu;
    int64_t size = grad_output_cpu.dim() - 1;
    while (grad_bias_cpu.dim() != size) {
      at_tmp = at::sum(at_tmp, -1, false);
      size -= 1;
    }
    at_tmp = at::sum(at_tmp, 0, false);
    grad_bias_cpu = at_tmp;
    grad_bias = grad_bias_cpu.to(device);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
custom_fallback_dipu_native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt, bool training,
    double momentum, double eps) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=dipu_native_batch_norm"
                           << std::endl);
  int64_t dim_c = input.size(1);
  at::TensorOptions options = input.options().dtype(at::kFloat);

  at::Tensor save_mean;
  if (running_mean_opt.has_value() && running_mean_opt.value().defined()) {
    save_mean = at::empty(running_mean_opt.value().sizes(),
                          running_mean_opt.value().options().dtype(at::kFloat));
  } else {
    save_mean = at::empty({dim_c}, options);
  }

  at::Tensor save_invstd;
  if (running_var_opt.has_value() && running_var_opt.value().defined()) {
    save_invstd =
        at::empty(running_var_opt.value().sizes(),
                  running_var_opt.value().options().dtype(at::kFloat));
  } else {
    save_invstd = at::empty({dim_c}, options);
  }

  at::Tensor out = at::empty(input.sizes(), input.options());
  return custom_fallback_dipu_native_batch_norm_out(
      input, weight_opt, bias_opt, running_mean_opt, running_var_opt, training,
      momentum, eps, out, save_mean, save_invstd);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
custom_fallback_dipu_linear_backward(const at::Tensor& input,
                                     const at::Tensor& grad_output,
                                     const at::Tensor& weight,
                                     ::std::array<bool, 3> output_mask) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=linear_backward"
                           << std::endl);
  auto input_cpu = input.cpu();
  auto grad_output_cpu = grad_output.cpu();
  auto weight_cpu = weight.cpu();

  at::Tensor grad_input;
  at::Tensor grad_input_cpu;

  at::Tensor grad_weight;
  at::Tensor grad_weight_cpu;

  at::Tensor grad_bias;
  at::Tensor grad_bias_cpu;

  int64_t dims = input.dim();
  const auto device = input.device();

  if (output_mask[0]) {
    auto grad_input_cpu = at::matmul(grad_output_cpu, weight_cpu);
    grad_input = grad_input_cpu.to(device);
  }
  if (output_mask[1]) {
    auto grad_weight_cpu =
        at::matmul(input_cpu.transpose(dims - 2, dims - 1), grad_output_cpu);
    grad_weight_cpu = grad_weight_cpu.transpose(dims - 2, dims - 1);
    if (dims > 2) {
      std::vector<int64_t> sum_dim;
      sum_dim.reserve(dims - 2);
      for (int i = 0; i < dims - 2; ++i) {
        sum_dim.push_back(i);
      }
      at::IntArrayRef at_sum_dim(sum_dim.data(), sum_dim.size());
      grad_weight_cpu = at::sum(grad_weight_cpu, at_sum_dim);
      grad_weight = grad_weight_cpu.to(device);
    }
  }

  if (output_mask[2]) {
    std::vector<int64_t> sum_dim;
    sum_dim.reserve(dims - 1);
    for (int i = 0; i < dims - 1; ++i) {
      sum_dim.push_back(i);
    }
    at::IntArrayRef at_sum_dim(sum_dim.data(), sum_dim.size());
    grad_bias_cpu = at::sum(grad_output_cpu, at_sum_dim);
    grad_bias = grad_bias_cpu.to(device);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<at::Tensor, at::Tensor> custom_fallback_dipu_matmul_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& other, ::std::array<bool, 2> mask) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to cpu, name=matmul_backward\n");
  auto grad_out_cpu = to_cpu_with_half_to_float(grad_out);
  auto input_cpu = to_cpu_with_half_to_float(input);
  auto other_cpu = to_cpu_with_half_to_float(other);

  if (other.dim() == 1) {
    other_cpu.unsqueeze_(-1);
    grad_out_cpu.unsqueeze_(-1);
  }

  if (input.dim() == 1) {
    input_cpu.unsqueeze_(0);
    grad_out_cpu.unsqueeze_(-2);
  }

  const auto device = input.device();
  at::Tensor grad_input;
  at::Tensor grad_other;

  if (mask[0]) {
    grad_input =
        at::sum_to(at::matmul(grad_out_cpu, other_cpu.transpose(-1, -2)),
                   input.sizes())
            .to(device, input.dtype());
  }

  if (mask[1]) {
    at::Tensor grad_other_cpu =
        at::matmul(input_cpu.transpose(-1, -2), grad_out_cpu);
    if (other.dim() == 1) {
      grad_other_cpu.squeeze_(-1);
    }
    grad_other =
        at::sum_to(grad_other_cpu, other.sizes()).to(device, other.dtype());
  }

  return {grad_input, grad_other};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
custom_fallback_dipu_native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_invstd_opt, bool train, double eps,
    ::std::array<bool, 3> output_mask) {
  DIPU_OP_LOG_WARNING_ONCE(
      "custom fallback to cpu, name=native_batch_norm_backward" << std::endl);
  int64_t dim_c = input.size(1);
  at::TensorOptions options = input.options().dtype(at::ScalarType::Float);

  at::Tensor grad_input = at::empty(input.sizes(), input.options());
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight_opt.has_value() && weight_opt.value().defined()) {
    grad_weight =
        at::empty(weight_opt.value().sizes(),
                  weight_opt.value().options().dtype(at::ScalarType::Float));
    grad_bias =
        at::empty(weight_opt.value().sizes(),
                  weight_opt.value().options().dtype(at::ScalarType::Float));
  } else {
    grad_weight = at::empty({dim_c}, options);
    grad_bias = at::empty({dim_c}, options);
  }

  at::Tensor grad_out_cpu = grad_out.cpu();
  at::Tensor input_cpu = input.cpu();
  at::Tensor grad_input_cpu = grad_input.cpu();
  at::Tensor grad_weight_cpu = grad_weight.cpu();
  at::Tensor grad_bias_cpu = grad_bias.cpu();

  c10::optional<at::Tensor> weight_cpu = dipu_to_cpu(weight_opt);
  c10::optional<at::Tensor> running_mean_cpu = dipu_to_cpu(running_mean_opt);
  c10::optional<at::Tensor> running_var_cpu = dipu_to_cpu(running_var_opt);
  c10::optional<at::Tensor> save_mean_cpu = dipu_to_cpu(save_mean_opt);
  c10::optional<at::Tensor> save_invstd_cpu = dipu_to_cpu(save_invstd_opt);
  auto at_out = at::native_batch_norm_backward(
      grad_out_cpu, input_cpu, weight_cpu, running_mean_cpu, running_var_cpu,
      save_mean_cpu, save_invstd_cpu, train, eps, output_mask);

  grad_input.copy_(std::get<0>(at_out));
  grad_weight.copy_(std::get<1>(at_out));
  grad_bias.copy_(std::get<2>(at_out));

  return {grad_input, grad_weight, grad_bias};
}

at::Tensor& custom_fallback_dipu_copy_(at::Tensor& self, const at::Tensor& src,
                                       bool non_blocking);

void custom_fallback_dipu__amp_foreach_non_finite_check_and_unscale_(
    at::TensorList scaled_grads, at::Tensor& found_inf,
    const at::Tensor& inv_scale);

at::Tensor& custom_fallback_dipu__amp_update_scale_(at::Tensor& current_scale,
                                                    at::Tensor& growth_tracker,
                                                    const at::Tensor& found_inf,
                                                    double growth_factor,
                                                    double backoff_factor,
                                                    int64_t growth_interval);

static at::Tensor& custom_fallback_dipu_addmm_out(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto mat1_cpu = to_cpu_with_half_to_float(mat1);
  auto mat2_cpu = to_cpu_with_half_to_float(mat2);
  auto out_cpu = to_cpu_with_half_to_float(out);
  out_cpu = at::addmm_out(out_cpu, self_cpu, mat1_cpu, mat2_cpu, beta, alpha);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor& custom_fallback_dipu_bmm_out(const at::Tensor& self,
                                                const at::Tensor& mat2,
                                                at::Tensor& out) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto mat2_cpu = to_cpu_with_half_to_float(mat2);
  auto out_cpu = to_cpu_with_half_to_float(out);
  out_cpu = at::bmm_out(out_cpu, self_cpu, mat2_cpu);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor custom_fallback_dipu_mm(const at::Tensor& self,
                                          const at::Tensor& mat2) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto mat2_cpu = to_cpu_with_half_to_float(mat2);
  auto out_cpu = at::mm(self_cpu, mat2_cpu);
  auto out =
      out_cpu.to(self.device()).to(self.options().dtype_opt()->toScalarType());
  return out;
}

static at::Tensor& custom_fallback_dipu_mm_out(const at::Tensor& self,
                                               const at::Tensor& mat2,
                                               at::Tensor& out) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto mat2_cpu = to_cpu_with_half_to_float(mat2);
  auto out_cpu = to_cpu_with_half_to_float(out);
  out_cpu = at::mm_out(out_cpu, self_cpu, mat2_cpu);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor custom_fallback_dipu_linear(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  auto input_cpu = to_cpu_with_half_to_float(input);
  auto weight_cpu = to_cpu_with_half_to_float(weight);
  c10::optional<at::Tensor> bias_cpu = c10::nullopt;

  at::Tensor out;
  at::Tensor out_cpu;

  if (bias.has_value() && bias.value().defined()) {
    if (bias.value().options().dtype_opt()->toScalarType() ==
        at::ScalarType::Half) {
      bias_cpu = bias.value().to(at::ScalarType::Float).cpu();
    } else {
      bias_cpu = bias.value().cpu();
    }
  }

  out_cpu = at::linear(input_cpu, weight_cpu, bias_cpu);
  out = out_cpu.to(input.device())
            .to(input.options().dtype_opt()->toScalarType());
  return out;
}

static at::Tensor& custom_fallback_dipu_rsqrt_out(const at::Tensor& self,
                                                  at::Tensor& out) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto out_cpu = at::rsqrt(self_cpu);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor& custom_fallback_dipu__softmax_out(const at::Tensor& self,
                                                     int64_t dim,
                                                     bool half_to_float,
                                                     at::Tensor& out) {
  auto self_cpu = to_cpu_with_half_to_float(self);
  auto out_cpu = at::softmax(self_cpu, dim);
  out.copy_(out_cpu);
  return out;
}

}  // namespace native
}  // namespace dipu
