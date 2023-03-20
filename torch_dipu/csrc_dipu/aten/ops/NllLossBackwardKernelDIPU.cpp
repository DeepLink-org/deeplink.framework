#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::nll_loss_backward_out_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
    ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t target_diopi = toDiopiTensorHandle(target);
    ::diopiConstTensorHandle_t weight_diopi = nullptr;
    if (weight.has_value() && weight.value().defined()) {
        weight_diopi = toDiopiTensorHandle(weight.value());
    }

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);

    ::diopiError_t ret = ::diopiNLLLossBackward(&context, grad_input_diopi, grad_output_diopi, self_diopi,
        target_diopi, weight_diopi, static_cast<diopiReduction_t>(reduction), ignore_index);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiNLLLossBackward error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return grad_input;
}

}  // namespace dipu::native