#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

at::Tensor DIPUNativeFunctions::relu(const at::Tensor& self) {
    ::diopiConstTensorHandle_t self_diopi = dipu::diopi::toDiopiTensorHandle(self);
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());

    c10::IntArrayRef out_sizes = self.sizes();
    at::Tensor out = at::empty(out_sizes, self.options());
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiRelu(&context, out_diopi, self_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiRelu error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

at::Tensor& DIPUNativeFunctions::relu_(at::Tensor& self) {
    ::diopiTensorHandle_t self_diopi = dipu::diopi::toDiopiTensorHandle(self);
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());

    ::diopiError_t ret = ::diopiReluInp(&context, self_diopi);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiReluInp error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return self;
}

}  // namespace dipu::native