#include <memory>

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor convolutionKernelDipu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups) {
    ::diopiConstTensorHandle_t input_diopi = toDiopiTensorHandle(input);
    ::diopiConstTensorHandle_t weight_diopi = toDiopiTensorHandle(weight);
    ::diopiConstTensorHandle_t bias_diopi = toDiopiTensorHandle(bias_opt);
    ::diopiSize_t stride_diopi(stride.data(), stride.size());
    ::diopiSize_t padding_diopi(padding.data(), padding.size());
    ::diopiSize_t dilation_diopi(dilation.data(), dilation.size());

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());

    // calculate the output size
    int64_t batch_size = input.size(0);
    int64_t height = input.size(2);
    int64_t width = input.size(3);
    int64_t out_channel = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);
    int64_t out_height = (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t out_width = (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    c10::SmallVector<int64_t, 8> output_size = {batch_size, out_channel, out_height, out_width};
    at::Tensor out = at::empty(output_size, input.options());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiConvolution2d(&context, out_diopi, input_diopi, weight_diopi, bias_diopi,
        stride_diopi, padding_diopi, dilation_diopi, groups);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " conv2d error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> convolutionBackwardKernelDipu(
    const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, std::array<bool, 3> grad_input_mask) {

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    // construct the output tensor on device
    grad_input = at::empty(input.sizes(), input.options());
    grad_weight = at::empty(weight.sizes(), weight.options().dtype(at::kFloat));
    if (grad_input_mask[2]) {
        c10::IntArrayRef bias_size = { grad.size(1) };
        grad_bias = at::empty(bias_size, grad.options());
    }

    ::diopiConstTensorHandle_t input_diopi = toDiopiTensorHandle(input);
    ::diopiConstTensorHandle_t weight_diopi = toDiopiTensorHandle(weight);
    ::diopiConstTensorHandle_t grad_diopi = toDiopiTensorHandle(grad);
    ::diopiSize_t stride_diopi(stride.data(), stride.size());
    ::diopiSize_t padding_diopi(padding.data(), padding.size());
    ::diopiSize_t dilation_diopi(dilation.data(), dilation.size());
    // no output padding for conv2d backward
    ::diopiSize_t output_padding_diopi;
    std::unique_ptr<::diopiSize_t> bias_sizes;
    ::diopiTensorHandle_t grad_bias_diopi = nullptr;
    if (grad_input_mask[2]) {
        grad_bias_diopi = toDiopiTensorHandle(grad_bias);
        bias_sizes.reset(new ::diopiSize_t(grad_bias.sizes().data(), grad_bias.dim()));
    }

    ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);
    ::diopiTensorHandle_t grad_weight_diopi = toDiopiTensorHandle(grad_weight);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());

    ::diopiError_t ret = ::diopiConvolution2dBackward(
        &context, grad_input_diopi, grad_weight_diopi, grad_bias_diopi,
        grad_diopi, input_diopi, weight_diopi, bias_sizes.get(), stride_diopi,
        padding_diopi, dilation_diopi, false, output_padding_diopi, groups);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " conv2d backward error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());

    return std::tie(grad_input, grad_weight, grad_bias);
}

class DipuConvlutionFunction : public torch::autograd::Function<DipuConvlutionFunction> {
public:
    static at::Tensor forward(
        AutogradContext *ctx, const at::Tensor &input, const at::Tensor &weight,
        const c10::optional<at::Tensor> &bias, at::IntArrayRef stride, at::IntArrayRef padding,
        at::IntArrayRef dilation, int64_t groups) {
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["bias_has_value"] = (bias.has_value() == true) ? bias.value().requires_grad() : false;

        at::AutoNonVariableTypeMode g;
        ctx->save_for_backward({input, weight});
        return convolutionKernelDipu(input, weight, bias, stride, padding, dilation, groups);
    }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      auto padding = ctx->saved_data["padding"].toIntVector();
      auto stride = ctx->saved_data["stride"].toIntVector();
      auto dilation = ctx->saved_data["dilation"].toIntVector();
      auto groups = ctx->saved_data["groups"].toInt();
      auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto weight = saved[1];

      std::array<bool, 3> grad_input_mask;
      grad_input_mask[0] = input.requires_grad();
      grad_input_mask[1] = weight.requires_grad();
      grad_input_mask[2] = bias_has_value;

      ::std::tuple<at::Tensor, at::Tensor, at::Tensor> result = convolutionBackwardKernelDipu(
          input, grad_outputs[0], weight, stride, padding,
          dilation, groups, grad_input_mask);
      tensor_list output = {
          std::get<0>(result), std::get<1>(result), std::get<2>(result),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor DIPUATenFunctions::conv2d(
    const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt;
    }

    return DipuConvlutionFunction::apply(input, weight, bias, stride, padding, dilation, groups);
}

}  // namespace dipu::native
