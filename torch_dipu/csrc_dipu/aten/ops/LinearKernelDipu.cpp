#include <memory>

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"


namespace dipu::native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor dipu_linear_impl(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias);
std::tuple<at::Tensor, at::Tensor, at::Tensor> dipu_linear_backward_impl(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight, ::std::array<bool, 3> output_mask);


class DipuLinearFunction : public torch::autograd::Function<DipuLinearFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx, const at::Tensor &input,
      const at::Tensor &weight, const c10::optional<at::Tensor> &bias) {
    bool bias_has_value = (bias.has_value() == true) ? bias.value().requires_grad() : false;
    std::array<bool, 3> output_mask{input.requires_grad(), weight.requires_grad(), bias_has_value};
    ctx->saved_data["output_mask"] = output_mask;

    at::AutoDispatchBelowADInplaceOrView g;
    ctx->save_for_backward({input, weight});
    return dipu_linear_impl(input, weight, bias);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto output_mask = ctx->saved_data["output_mask"].to<std::array<bool, 3>>();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto grad_output = grad_outputs[0];

    auto grads = dipu_linear_backward_impl(input, grad_output, weight, output_mask);
    return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
  }
};


at::Tensor linear(
    const at::Tensor & input, const at::Tensor & weight,
    const c10::optional<at::Tensor> & bias_opt) {
  c10::optional<at::Tensor> bias = c10::nullopt;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
      bias = bias_opt;
  }

  return DipuLinearFunction::apply(input, weight, bias);
}

}  // namespace dipu::native


namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
  DIOPI_ATEN_FUNC("linear", diopiLinear, dipu::native::linear);
}

}  // namespace at