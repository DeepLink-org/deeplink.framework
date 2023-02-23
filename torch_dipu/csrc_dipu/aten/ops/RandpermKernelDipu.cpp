#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"

namespace dipu::native {

at::Tensor& DIPUNativeFunctions::randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& result) {
    ::diopiContext context(c10::cuda::getCurrentCUDAStream().stream());
    ::diopiTensorHandle_t out_diopi = dipu::diopi::toDiopiTensorHandle(result);
    int64_t idx = 0;

    ::diopiError_t ret = ::diopiRandperm(&context, out_diopi, n, idx);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiRandperm error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return result;
}

at::Tensor& DIPUNativeFunctions::randperm_out(int64_t n, at::Tensor& result) {
    return DIPUNativeFunctions::randperm_out(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), result);
}

}  // namespace dipu::native