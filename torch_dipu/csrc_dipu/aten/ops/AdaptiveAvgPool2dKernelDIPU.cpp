#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

std::vector<int64_t> getOutputSize(const c10::SymIntArrayRef &output_size) {
    std::vector<int64_t> data;
    data.reserve(output_size.size());
    for (auto iter = output_size.cbegin(); iter != output_size.cend(); ++iter) {
        data.emplace_back(iter->expect_int());
    }
    return data;
}

at::Tensor& DIPUATenFunctions::adaptive_avg_pool2d_out(const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
    ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
    std::vector<int64_t> data = getOutputSize(output_size);
    ::diopiSize_t diopi_size(data.data(), data.size());

    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

    ::diopiError_t ret = ::diopiAdaptiveAvgPool2d(&context, out_diopi, self_diopi, diopi_size);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiAdaptiveAvgPool2d error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return out;
}

at::Tensor DIPUATenFunctions::_adaptive_avg_pool2d(const at::Tensor & self, c10::SymIntArrayRef output_size) {
    auto self_size = self.sizes();
    std::vector<int64_t> out_tensor_size = self_size.vec();
    std::vector<int64_t> data = getOutputSize(output_size);
    TORCH_CHECK(data.size() == 2, __func__, ":", __FILE__, ":", __LINE__,
        " output_size should equal 2, size is ", data.size());
    out_tensor_size[self.dim() - 1] = data[1];
    out_tensor_size[self.dim() - 2] = data[0];
    at::Tensor out = at::empty(out_tensor_size, self.options());

    return DIPUATenFunctions::adaptive_avg_pool2d_out(self, output_size, out);
}

}  // namespace dipu::native