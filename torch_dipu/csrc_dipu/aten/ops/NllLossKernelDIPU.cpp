#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::nll_loss_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t target_diopi = toDiopiTensorHandle(target);
    int64_t ignore_index_diopi = ignore_index.expect_int();
    ::diopiConstTensorHandle_t weight_diopi = nullptr;
    if (weight.has_value() && weight.value().defined()) {
        weight_diopi = toDiopiTensorHandle(weight.value());
    }

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiNLLLoss(&context, out_diopi, self_diopi, target_diopi,
        weight_diopi, static_cast<diopiReduction_t>(reduction), ignore_index_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiNLLLoss error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return out;
}

::std::tuple<at::Tensor &,at::Tensor &> DIPUATenFunctions::nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
    at::Tensor &result = DIPUATenFunctions::nll_loss_out(self, target, weight, reduction, ignore_index, output);
    return std::tie(result, total_weight);
}

}  // namespace dipu::native