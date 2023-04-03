#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::log_softmax_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiLogSoftmax(&context, out_diopi, self_diopi, dim);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiLogSoftmax error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

at::Tensor& DIPUATenFunctions::_log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    return DIPUATenFunctions::log_softmax_out(self, dim, c10::nullopt, out);
}

}  // namespace dipu::native
