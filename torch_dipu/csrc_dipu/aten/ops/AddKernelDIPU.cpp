#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t other_diopi = toDiopiTensorHandle(other);
    ::diopiScalar_t alpha_diopi = toDiopiScalar(alpha);
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAdd(&context, out_diopi, self_diopi, other_diopi, &alpha_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAdd error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

at::Tensor DIPUATenFunctions::add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiScalar_t other_diopi = toDiopiScalar(other);
    ::diopiScalar_t alpha_diopi = toDiopiScalar(alpha);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    at::Tensor out = at::empty(self.sizes(), self.options());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAddScalar(&context, out_diopi, self_diopi, &other_diopi, &alpha_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAddScalar error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return out;
}

at::Tensor& add_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    ::diopiScalar_t other_diopi = toDiopiScalar(other);
    ::diopiScalar_t alpha_diopi = toDiopiScalar(alpha);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t self_diopi = toDiopiTensorHandle(self);

    ::diopiError_t ret = ::diopiAddInpScalar(&context, self_diopi, &other_diopi, &alpha_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAddInpScalar error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return self;
}

}  // namespace dipu::native