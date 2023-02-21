#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

at::Tensor& DIPUNativeFunctions::add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = dipu::diopi::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t other_diopi = dipu::diopi::toDiopiTensorHandle(other);
    ::diopiScalar_t alpha_diopi = dipu::diopi::toDiopiScalar(alpha);
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAdd(&context, out_diopi, self_diopi, other_diopi, &alpha_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAdd error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native