#pragma once

namespace dipu::native {

c10::optional<at::Tensor> dipu_to_cpu(const c10::optional<at::Tensor> & device_tensor) {
    c10::optional<at::Tensor> cpu_tensor = c10::nullopt;
    if (device_tensor.has_value() && device_tensor.value().defined()) {
        cpu_tensor = device_tensor.value().cpu();
    }
    return cpu_tensor;
}

at::Tensor to_cpu_no_half(const at::Tensor& devtensor) {
    auto cpu_tensor = devtensor.cpu();
    auto intype = devtensor.options().dtype_opt()->toScalarType();
    if (intype == at::ScalarType::Half) {
      return cpu_tensor.to(at::ScalarType::Float);
    } else {
      return cpu_tensor;
    }  
}

at::Tensor& custom_fallback_dipu_silu_out(const at::Tensor& self, at::Tensor& out) {
  std::cout << "custom fallback to cpu, name=" << "silu_out" << std::endl;

  auto self_cpu = to_cpu_no_half(self);
  auto out_cpu = to_cpu_no_half(self);
  out_cpu = at::silu_out(self_cpu, out_cpu);
  out.copy_(out_cpu.to(at::ScalarType::Half));
  return out;
}

::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> custom_fallback_dipu_native_batch_norm_out(
    const at::Tensor & input, const c10::optional<at::Tensor> & weight_opt,
    const c10::optional<at::Tensor> & bias_opt,
    const c10::optional<at::Tensor> & running_mean_opt,
    const c10::optional<at::Tensor> & running_var_opt,
    bool training, double momentum, double eps,
    at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {

  at::Tensor input_cpu = input.cpu();
  at::Tensor out_cpu = out.cpu();
  at::Tensor save_mean_cpu = save_mean.cpu();
  at::Tensor save_invstd_cpu = save_invstd.cpu();

  c10::optional<at::Tensor> weight_cpu = dipu_to_cpu(weight_opt);
  c10::optional<at::Tensor> bias_cpu = dipu_to_cpu(bias_opt);
  c10::optional<at::Tensor> running_mean_cpu = dipu_to_cpu(running_mean_opt);
  c10::optional<at::Tensor> running_var_cpu = dipu_to_cpu(running_var_opt);

  at::native_batch_norm_out(out_cpu, save_mean_cpu, save_invstd_cpu, input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, training, momentum, eps);

  out.copy_(out_cpu);
  save_mean.copy_(save_mean_cpu);
  save_invstd.copy_(save_invstd_cpu);

  if (running_mean_opt.has_value() && running_mean_opt.value().defined()) {
    running_mean_opt.value().copy_(running_mean_cpu.value());
  }
  if (running_var_opt.has_value() && running_var_opt.value().defined()) {
    running_var_opt.value().copy_(running_var_cpu.value());
  }

  return std::tie(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_native_batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt, const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt, bool training, double momentum, double eps) {
  std::cout << "enter into fallback native_batch_norm" << std::endl;
  int64_t dim_c = input.size(1);
  at::TensorOptions options = input.options().dtype(at::kFloat);

  at::Tensor save_mean;
  if (running_mean_opt.has_value() && running_mean_opt.value().defined()) {
    save_mean = at::empty(running_mean_opt.value().sizes(), running_mean_opt.value().options().dtype(at::kFloat));
  } else {
    save_mean = at::empty({dim_c}, options);
  }

  at::Tensor save_invstd;
  if (running_var_opt.has_value() && running_var_opt.value().defined()) {
    save_invstd = at::empty(running_var_opt.value().sizes(), running_var_opt.value().options().dtype(at::kFloat));
  } else {
    save_invstd = at::empty({dim_c}, options);
  }

  at::Tensor out = at::empty(input.sizes(), input.options());
  return custom_fallback_dipu_native_batch_norm_out(
      input, weight_opt, bias_opt, running_mean_opt, running_var_opt,
      training, momentum, eps, out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_native_batch_norm_backward(
        const at::Tensor& grad_out, const at::Tensor& input, const c10::optional<at::Tensor>& weight_opt,
        const c10::optional<at::Tensor>& running_mean_opt, const c10::optional<at::Tensor>& running_var_opt,
        const c10::optional<at::Tensor>& save_mean_opt, const c10::optional<at::Tensor>& save_invstd_opt,
        bool train, double eps, ::std::array<bool, 3> output_mask) {
    std::cout << "enter into fallback native_batch_norm backward" << std::endl;
    int64_t dim_c = input.size(1);
    at::TensorOptions options = input.options().dtype(at::ScalarType::Float);

    at::Tensor grad_input = at::empty(input.sizes(), input.options());
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (weight_opt.has_value() && weight_opt.value().defined()) {
        grad_weight = at::empty(weight_opt.value().sizes(), weight_opt.value().options().dtype(at::ScalarType::Float));
        grad_bias = at::empty(weight_opt.value().sizes(), weight_opt.value().options().dtype(at::ScalarType::Float));
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
    auto atOut = at::native_batch_norm_backward(grad_out_cpu, input_cpu, weight_cpu, running_mean_cpu,
        running_var_cpu, save_mean_cpu, save_invstd_cpu, train, eps, output_mask);

    grad_input.copy_(std::get<0>(atOut));
    grad_weight.copy_(std::get<1>(atOut));
    grad_bias.copy_(std::get<2>(atOut));

    return std::tie(grad_input, grad_weight, grad_bias);
}

};