#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor DIPUATenFunctions::adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self) {
    ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    at::Tensor out = at::empty(self.sizes(), self.options());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAdaptiveAvgPool2dBackward(&context, out_diopi, grad_output_diopi, self_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAdaptiveAvgPool2dBackward error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native