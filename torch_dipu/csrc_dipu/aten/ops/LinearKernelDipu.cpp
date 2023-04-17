#include <memory>

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

std::vector<int64_t> getOutputSize(const c10::IntArrayRef& input_size, const c10::IntArrayRef& weight_size) {
  std::vector<int64_t> output_size;
  for (int i = 0; i < input_size.size() - 1; ++i) {
    output_size.push_back(input_size[i]);
  }
  if (weight_size.size() > 1) {
    output_size.push_back(weight_size[0]);
  }
  return output_size;
}

at::Tensor linearKernelDipu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt) {
  ::diopiConstTensorHandle_t input_diopi = toDiopiTensorHandle(input);
  ::diopiConstTensorHandle_t weight_diopi = toDiopiTensorHandle(weight);
  ::diopiConstTensorHandle_t bias_diopi = toDiopiTensorHandle(bias_opt);

  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  std::vector<int64_t> out_size = getOutputSize(input.sizes(), weight.sizes());
  at::Tensor out = at::empty(out_size, input.options());
  ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

  ::diopiError_t ret = ::diopiLinear(&context, out_diopi, input_diopi, weight_diopi, bias_diopi);
  TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
      " linear error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
  return out;
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> linearBackwardKernelDipu(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, bool bias_has_value) {
  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  // construct the output tensor on device
  grad_input = at::empty(input.sizes(), input.options());
  grad_weight = at::empty(weight.sizes(), weight.options().dtype(at::kFloat));
  if (bias_has_value) {
      c10::IntArrayRef bias_size = { grad_output.size(1) };
      grad_bias = at::empty(bias_size, grad_output.options());
  }

  // generate diopi input parameter
  ::diopiConstTensorHandle_t input_diopi = toDiopiTensorHandle(input);
  ::diopiConstTensorHandle_t weight_diopi = toDiopiTensorHandle(weight);
  ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);

  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);
  ::diopiTensorHandle_t grad_weight_diopi = toDiopiTensorHandle(grad_weight);
  ::diopiTensorHandle_t grad_bias_diopi = nullptr;
  if (bias_has_value) {
    grad_bias_diopi = toDiopiTensorHandle(grad_bias);
  }

  ::diopiError_t ret = ::diopiLinearBackward(
    &context, grad_input_diopi, grad_weight_diopi, grad_bias_diopi,
    grad_output_diopi, input_diopi, weight_diopi);
  TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
    " linear backward error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());

  return std::tie(grad_input, grad_weight, grad_bias);
}

class DipuLinearFunction : public torch::autograd::Function<DipuLinearFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx, const at::Tensor &input,
      const at::Tensor &weight, const c10::optional<at::Tensor> &bias) {
    ctx->saved_data["bias_has_value"] = (bias.has_value() == true) ? bias.value().requires_grad() : false;

    at::AutoDispatchBelowADInplaceOrView g;
    ctx->save_for_backward({input, weight});
    return linearKernelDipu(input, weight, bias);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];

    ::std::tuple<at::Tensor, at::Tensor, at::Tensor> result = linearBackwardKernelDipu(
      grad_outputs[0], input, weight, bias_has_value);
    tensor_list output = {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
    return output;
  }
};

at::Tensor DIPUATenFunctions::linear(
    const at::Tensor & input, const at::Tensor & weight,
    const c10::optional<at::Tensor> & bias_opt) {
  c10::optional<at::Tensor> bias = c10::nullopt;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
      bias = bias_opt;
  }

  return DipuLinearFunction::apply(input, weight, bias);
}

}  // namespace dipu::native