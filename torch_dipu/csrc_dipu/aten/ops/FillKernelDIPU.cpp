#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

namespace dipu::native {
at::Tensor& DIPUATenFunctions::fillScalar_(at::Tensor& self, const at::Scalar& value) {
    ::diopiTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiScalar_t value_diopi = toDiopiScalar(value);
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());

    ::diopiError_t ret = ::diopiFill(&context, self_diopi, &value_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiFill error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return self;
}
}