#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t other_diopi = toDiopiTensorHandle(other);
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiMul(&context, out_diopi, self_diopi, other_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiMul error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native