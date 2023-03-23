#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiSize;
using dipu::diopi_helper::toDiopiDtype;

namespace dipu::native {

extern ::diopiDtype_t getDiopiDtype(const c10::optional<at::ScalarType>& dtype, const at::Tensor & out);

at::Tensor& DIPUATenFunctions::mean_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiSize_t diopi_size = toDiopiSize(dim);
    ::diopiDtype_t diopi_dtype = getDiopiDtype(dtype, out);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiMean(&context, out_diopi, self_diopi, diopi_size, diopi_dtype);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiMean error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native