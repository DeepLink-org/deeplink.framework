#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> DIPUNativeFunctions::native_batch_norm(
    const at::Tensor & input, const c10::optional<at::Tensor> & weight_opt,
    const c10::optional<at::Tensor> & bias_opt,
    const c10::optional<at::Tensor> & running_mean_opt,
    const c10::optional<at::Tensor> & running_var_opt,
    bool training, double momentum, double eps) {
    ::diopiConstTensorHandle_t input_diopi = dipu::diopi::toDiopiTensorHandle(input);

    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});

    int64_t dim_c = input.size(1);
    at::TensorOptions options = input.options().dtype(at::ScalarType::Float);

    at::Tensor weight_tensor = weight.defined() ? weight : at::ones({dim_c}, options);
    at::Tensor bias_tensor = bias.defined() ? bias : at::zeros({dim_c}, options);
    at::Tensor running_mean_tensor = running_mean.defined() ? running_mean : at::zeros({dim_c}, options);
    at::Tensor running_var_tensor = running_var.defined() ? running_var : at::ones({dim_c}, options);

    ::diopiConstTensorHandle_t weight_diopi = dipu::diopi::toDiopiTensorHandle(weight_tensor);
    ::diopiConstTensorHandle_t bias_diopi = dipu::diopi::toDiopiTensorHandle(bias_tensor);
    ::diopiConstTensorHandle_t running_mean_diopi = dipu::diopi::toDiopiTensorHandle(running_mean_tensor);
    ::diopiConstTensorHandle_t running_var_diopi = dipu::diopi::toDiopiTensorHandle(running_var_tensor);
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());

    at::Tensor out = at::empty(input.sizes(), input.options());
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(out);

    ::diopiTensorHandle_t save_mean_diopi = nullptr;
    ::diopiTensorHandle_t save_invstd_diopi = nullptr;
    at::Tensor save_mean = {};
    at::Tensor save_invstd = {};
    if (training) {
        save_mean = at::empty(running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat));
        save_invstd = at::empty(running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat));
    }
    save_mean_diopi = dipu::diopi::toDiopiTensorHandle(save_mean);
    save_invstd_diopi = dipu::diopi::toDiopiTensorHandle(save_invstd);

    ::diopiError_t ret = ::diopiBatchNorm(
        &context, out_diopi, save_mean_diopi, save_invstd_diopi,
        input_diopi, weight_diopi, bias_diopi, running_mean_diopi,
        running_var_diopi, training, momentum, eps);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiBatchNorm error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return std::tie(out, save_mean, save_invstd);
}

}  // namespace dipu::native