#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor DIPUATenFunctions::cross_entropy_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t target_diopi = toDiopiTensorHandle(target);
    int64_t ignore_index_diopi = ignore_index.expect_int();
    ::diopiConstTensorHandle_t weight_diopi = nullptr;
    if (weight.has_value() && weight.value().defined()) {
        weight_diopi = toDiopiTensorHandle(weight.value());
    }

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    at::Tensor scalar_tensor = at::empty({}, at::kFloat);
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(scalar_tensor);

    ::diopiError_t ret = ::diopiCrossEntropyLoss(&context, out_diopi, self_diopi,
        target_diopi, weight_diopi, static_cast<diopiReduction_t>(reduction),
        ignore_index_diopi, label_smoothing);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiCrossEntropyLoss error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return scalar_tensor;
}

}  // namespace dipu::native