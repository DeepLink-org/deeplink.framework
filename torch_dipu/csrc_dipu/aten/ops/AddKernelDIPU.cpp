#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

std::vector<int64_t> broadcastOpsOutputSize(
    c10::IntArrayRef shape1, c10::IntArrayRef shape2) {
    return at::infer_size(shape1, shape2);
}

std::vector<int64_t> broadcastOpsOutputSize(
    const at::Tensor &self, const at::Tensor &other) {
    return broadcastOpsOutputSize(self.sizes(), other.sizes());
}

at::Tensor DIPUNativeFunctions::add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    ::diopiConstTensorHandle_t self_diopi = dipu::diopi::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t other_diopi = dipu::diopi::toDiopiTensorHandle(other);
    ::diopiScalar_t alpha_diopi = dipu::diopi::toDiopiScalar(alpha);
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());

    std::vector<int64_t> out_sizes = broadcastOpsOutputSize(self, other);
    at::Tensor out = at::empty(out_sizes, self.options());
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAdd(&context, out_diopi, self_diopi, other_diopi, &alpha_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAdd error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

}  // namespace dipu::native