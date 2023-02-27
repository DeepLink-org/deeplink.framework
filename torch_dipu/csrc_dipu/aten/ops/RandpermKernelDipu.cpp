#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& result) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(result);
    int64_t idx = 0;

    ::diopiError_t ret = ::diopiRandperm(&context, out_diopi, n, idx);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiRandperm error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return result;
}

at::Tensor& DIPUATenFunctions::randperm_out(int64_t n, at::Tensor& result) {
    return DIPUATenFunctions::randperm_out(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), result);
}

}  // namespace dipu::native