#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiDtype;

namespace dipu::native {

extern ::diopiDtype_t getDiopiDtype(const c10::optional<at::ScalarType>& dtype, const at::Tensor & out);

at::Tensor& DIPUATenFunctions::log_softmax_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiDtype_t diopi_dtype = getDiopiDtype(dtype, out);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiLogSoftmax(&context, out_diopi, self_diopi, dim, diopi_dtype);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiLogSoftmax error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

at::Tensor& DIPUATenFunctions::_log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    return DIPUATenFunctions::log_softmax_out(result, dim, c10::nullopt, out);
}

}  // namespace dipu::native