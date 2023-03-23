#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiDtype;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::_log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out) {
    ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);
    ::diopiConstTensorHandle_t output_diopi = toDiopiTensorHandle(output);
    ::diopiDtype_t input_dtype_diopi = toDiopiDtype(input_dtype);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiLogSoftmaxBackward(&context, out_diopi, grad_output_diopi, output_diopi, dim, input_dtype_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiLogSoftmaxBackward error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native