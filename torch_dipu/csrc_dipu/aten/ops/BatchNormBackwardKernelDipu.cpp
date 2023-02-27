#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> DIPUATenFunctions::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_invstd_opt,
    bool train, double eps, std::array<bool, 3> grad_input_mask) {
    ::diopiConstTensorHandle_t input_diopi = toDiopiTensorHandle(input);
    ::diopiConstTensorHandle_t grad_out_diopi = toDiopiTensorHandle(grad_out);

    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});
    const at::Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return at::Tensor();});
    const at::Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return at::Tensor();});

    int64_t dim_c = input.size(1);
    at::TensorOptions options = input.options().dtype(at::ScalarType::Float);

    at::Tensor weight_tensor = weight.defined() ? weight : at::ones({dim_c}, options);
    at::Tensor running_mean_tensor = running_mean.defined() ? running_mean : at::zeros({dim_c}, options);
    at::Tensor running_var_tensor = running_var.defined() ? running_var : at::ones({dim_c}, options);
    at::Tensor save_mean_tensor = save_mean.defined() ? save_mean : at::zeros({dim_c}, options);
    at::Tensor save_invsted_tensor = save_invstd.defined() ? save_invstd : at::ones({dim_c}, options);

    ::diopiConstTensorHandle_t weight_diopi = toDiopiTensorHandle(weight_tensor);
    ::diopiConstTensorHandle_t running_mean_diopi = toDiopiTensorHandle(running_mean_tensor);
    ::diopiConstTensorHandle_t running_var_diopi = toDiopiTensorHandle(running_var_tensor);
    ::diopiConstTensorHandle_t save_mean_diopi = toDiopiTensorHandle(save_mean_tensor);
    ::diopiConstTensorHandle_t save_invsted_diopi = toDiopiTensorHandle(save_invsted_tensor);
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());

    // construct the output tensor on dipu
    at::Tensor grad_input = at::empty(input.sizes(), input.options());
    at::Tensor grad_weight = at::empty(weight_tensor.sizes(), weight_tensor.options().dtype(at::ScalarType::Float));
    at::Tensor grad_bias = at::empty(weight_tensor.sizes(), weight_tensor.options().dtype(at::ScalarType::Float));
    ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);
    ::diopiTensorHandle_t grad_weight_diopi = toDiopiTensorHandle(grad_weight);
    ::diopiTensorHandle_t grad_bias_diopi = toDiopiTensorHandle(grad_bias);

    ::diopiError_t ret = ::diopiBatchNormBackward(
        &context, grad_input_diopi, grad_weight_diopi, grad_bias_diopi,
        grad_out_diopi, input_diopi, weight_diopi, running_mean_diopi,
        running_var_diopi, save_mean_diopi, save_invsted_diopi, train, eps);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiBatchNorm error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());

    return std::tie(grad_input, grad_weight, grad_bias);
}

}  // namespace dipu::native