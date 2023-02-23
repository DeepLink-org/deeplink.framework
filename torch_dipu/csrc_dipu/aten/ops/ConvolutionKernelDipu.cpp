#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

c10::SmallVector<int64_t, 8> convertIntArrayRefToSmallVector(c10::IntArrayRef array) {
    c10::SmallVector<int64_t, 8> vec;
    for (int i = 0; i < array.size(); i++) {
        vec.emplace_back(array[i]);
    }
    return vec;
}

inline c10::SmallVector<int64_t, 8> expandDimIfNeed(
    at::IntArrayRef list_param, int64_t expected_dim) {
    if (list_param.size() != 1) {
        return convertIntArrayRefToSmallVector(list_param);
    }

    c10::SmallVector<int64_t, 8> expand_dim_param_vec;
    for (int64_t i = 0; i < expected_dim; i++) {
        expand_dim_param_vec.emplace_back(list_param[0]);
    }
    return expand_dim_param_vec;
}

at::Tensor convolutionKernelDipu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups) {
    ::diopiConstTensorHandle_t input_diopi = dipu::diopi::toDiopiTensorHandle(input);
    ::diopiConstTensorHandle_t weight_diopi = dipu::diopi::toDiopiTensorHandle(weight);
    ::diopiConstTensorHandle_t bias_diopi = dipu::diopi::toDiopiTensorHandle(bias_opt);
    ::diopiSize_t stride_diopi(stride.data(), stride.size());
    ::diopiSize_t padding_diopi(padding.data(), padding.size());
    ::diopiSize_t dilation_diopi(dilation.data(), dilation.size());

    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());

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
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiConvolution2d(&context, out_diopi, input_diopi, weight_diopi, bias_diopi,
        stride_diopi, padding_diopi, dilation_diopi, groups);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " conv2d error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
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

    // tuple<at::Tensor, at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_convolution_backward(input,
    //     grad_outputs[0],
    //     weight,
    //     stride,
    //     padding,
    //     dilation,
    //     groups,
    //     grad_input_mask);
    // tensor_list output = {std::get<0>(result),
    //     std::get<1>(result),
    //     std::get<2>(result),
    //     at::Tensor(),
    //     at::Tensor(),
    //     at::Tensor(),
    //     at::Tensor()};


    tensor_list output = {at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    abort();
    return output;
  }
};

at::Tensor DIPUNativeFunctions::conv2d(
    const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt;
    }
    at::Tensor tensor;
    int64_t k = input.ndimension();
    int64_t dim = k - 2;

    auto expand_stride = expandDimIfNeed(stride, dim);
    auto expand_padding = expandDimIfNeed(padding, dim);
    auto expand_dilation = expandDimIfNeed(dilation, dim);

    return DipuConvlutionFunction::apply(input, weight, bias, stride, padding, dilation, groups);
}

}  // namespace dipu::native
