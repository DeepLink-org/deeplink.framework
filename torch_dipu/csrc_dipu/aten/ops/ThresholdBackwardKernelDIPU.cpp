#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::threshold_backward_out_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
    ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiScalar_t threshold_diopi = toDiopiScalar(threshold);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);

    ::diopiError_t ret = ::diopiThresholdBackward(&context, grad_input_diopi, grad_output_diopi, self_diopi, &threshold_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiThresholdBackward error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return grad_input;
}

}  // namespace dipu::native