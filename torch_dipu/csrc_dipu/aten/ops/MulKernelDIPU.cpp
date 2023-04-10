#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

namespace dipu::native {

at::Tensor& mul_scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor &out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiScalar_t other_diopi = toDiopiScalar(other);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiMulScalar(&context, out_diopi, self_diopi, &other_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiMulScalar error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return out;
}

at::Tensor& DIPUATenFunctions::mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    if (other.dim() == 0 && other.is_cpu()) {
        return mul_scalar_out(self, other.item(), out);
    } else if (self.dim() == 0 && self.is_cpu()) {
        return mul_scalar_out(other, self.item(), out);
    }

    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t other_diopi = toDiopiTensorHandle(other);
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiMul(&context, out_diopi, self_diopi, other_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiMul error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return out;
}

at::Tensor DIPUATenFunctions::mul(const at::Tensor & self, const at::Scalar & other) {
    at::Tensor out = at::empty(self.sizes(), self.options());
    return mul_scalar_out(self, other, out);
}

at::Tensor& DIPUATenFunctions::mul_(at::Tensor & self, const at::Scalar & other) {
    ::diopiScalar_t other_diopi = toDiopiScalar(other);

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t self_diopi = toDiopiTensorHandle(self);

    ::diopiError_t ret = ::diopiMulInpScalar(&context, self_diopi, &other_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiMulInpScalar error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
    return self;
}

}  // namespace dipu::native