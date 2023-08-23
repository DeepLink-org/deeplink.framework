#pragma once

namespace dipu::native {

static c10::optional<at::Tensor> dipu_to_cpu(const c10::optional<at::Tensor> & device_tensor) {
    c10::optional<at::Tensor> cpu_tensor = c10::nullopt;
    if (device_tensor.has_value() && device_tensor.value().defined()) {
        cpu_tensor = device_tensor.value().cpu();
    }
    return cpu_tensor;
}

static at::Tensor to_cpu_no_half(const at::Tensor& devtensor) {
    auto cpu_tensor = devtensor.cpu();
    auto intype = devtensor.options().dtype_opt()->toScalarType();
    if (intype == at::ScalarType::Half) {
      return cpu_tensor.to(at::ScalarType::Float);
    } else {
      return cpu_tensor;
    }
}

static at::Tensor& custom_fallback_dipu_silu_out(const at::Tensor& self, at::Tensor& out) {
  DIPU_REGISTER_LOG("custom fallback to cpu, name=silu_out" << std::endl);
  auto self_cpu = to_cpu_no_half(self);
  auto out_cpu = to_cpu_no_half(self);
  out_cpu = at::silu_out(self_cpu, out_cpu);
  out.copy_(out_cpu);
  return out;
}

static c10::List<c10::optional<at::Tensor>> to_cpu( const c10::List<c10::optional<at::Tensor>>& indices) {
  c10::List<c10::optional<at::Tensor>> indices_cpu;
  indices_cpu.reserve(indices.size());
  // input as x[1:2, [1, 2]], Slice by first dimension already executed before this index(),
  // in this case, indices[0] is an undefinedTensor.
  for (int i = 0; i < indices.size(); ++i) {
    indices_cpu.push_back((indices[i].has_value() && indices[i].value().defined()) ?
                          indices[i].value().to("cpu") : at::Tensor());
  }
  return indices_cpu;
}
static at::Tensor& custom_fallback_dipu_index_tensor_out(const at::Tensor& self,
            const c10::List<c10::optional<at::Tensor>>& indices, at::Tensor& out) {
  DIPU_REGISTER_LOG("custom fallback to cpu, name=index.Tensor_out" << std::endl);
  auto indices_cpu = to_cpu(indices);

  at::Tensor out_cpu = out.cpu();
  at::index_outf(self.cpu(), indices_cpu, out_cpu);
  out.copy_(out_cpu);
  return out;
}

static at::Tensor& custom_fallback_dipu__index_put_impl_(at::Tensor& self,
              const c10::List<c10::optional<at::Tensor>>& indices,
              const at::Tensor& values, bool accumulate, bool unsafe) {
  DIPU_REGISTER_LOG("custom fallback to cpu, name=_index_put_impl_" << std::endl);

  auto indices_cpu = to_cpu(indices);
  at::Tensor self_cpu = self.cpu();
  at::native::_index_put_impl_(self_cpu, indices_cpu, values.cpu(), accumulate, unsafe);
  self.copy_(self_cpu);

  return self;
}


static ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> custom_fallback_dipu_native_batch_norm_out(
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

static at::Tensor custom_fallback_dipu_convolution_overrideable(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto bias_cpu = dipu_to_cpu(bias);
  auto result = at::convolution(input_cpu, weight_cpu, bias_cpu, stride, padding, dilation, transposed, output_padding, groups);
  return result.to(input.device());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_convolution_backward_overrideable(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool, 3> output_mask) {
  auto device = input.device();
  auto grad_output_cpu = grad_output.cpu();
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto output_mask_temp = output_mask;
  output_mask_temp[2] = false;
  auto result = at::convolution_backward(grad_output_cpu, input_cpu, weight_cpu, c10::nullopt, stride, padding, dilation, transposed, output_padding, groups, output_mask_temp);

  at::Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    grad_input = std::get<0>(result).to(device);
  }
  if (output_mask[1]) {
    grad_weight = std::get<1>(result).to(device);
  }

  if (output_mask[2]) {
    std::vector<int64_t> bias_sizes{grad_output.size(1)};
    at::Tensor grad_bias_cpu = at::empty(bias_sizes, grad_output.options().device(c10::DeviceType::CPU));
    grad_bias = at::empty(bias_sizes, grad_output.options());
    at::Tensor atTmp = grad_output_cpu;
    int64_t size = grad_output_cpu.dim() - 1;
    while (grad_bias_cpu.dim() != size) {
      atTmp = at::sum(atTmp, -1, false);
      size -= 1;
    }
    atTmp = at::sum(atTmp, 0, false);
    grad_bias_cpu = atTmp;
    grad_bias = grad_bias_cpu.to(device);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_native_batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight_opt,
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

static std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_linear_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight, ::std::array<bool, 3> output_mask) {
    auto input_cpu = input.cpu();
    auto grad_output_cpu = grad_output.cpu();
    auto weight_cpu = weight.cpu();

    at::Tensor grad_input_cpu, grad_weight_cpu, grad_bias_cpu;
    at::Tensor grad_input, grad_weight, grad_bias;
    int64_t dims = input.dim();
    const auto device = input.device();

    if (output_mask[0]) {
      auto grad_input_cpu = at::matmul(grad_output_cpu, weight_cpu);
      grad_input = grad_input_cpu.to(device);
    }
    if (output_mask[1]) {
        auto grad_weight_cpu = at::matmul(input_cpu.transpose(dims - 2, dims - 1), grad_output_cpu);
        grad_weight_cpu = grad_weight_cpu.transpose(dims - 2, dims - 1);
        if (dims > 2) {
            std::vector<int64_t> sumDim;
            for (int i = 0; i < dims - 2; ++i) {
                sumDim.push_back(i);
            }
            at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
            grad_weight_cpu = at::sum(grad_weight_cpu, atSumDim);
            grad_weight = grad_weight_cpu.to(device);
        }
    }

    if (output_mask[2]) {
        std::vector<int64_t> sumDim;
        for (int i = 0; i < dims - 1; ++i) {
            sumDim.push_back(i);
        }
        at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
        grad_bias_cpu = at::sum(grad_output_cpu, atSumDim);
        grad_bias = grad_bias_cpu.to(device);
    }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_fallback_dipu_native_batch_norm_backward(
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