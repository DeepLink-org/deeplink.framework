#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

static at::Tensor & randomInp(at::Tensor & self, int64_t from, c10::optional<int64_t> to_opt, c10::optional<at::Generator> generator) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(self);
    int64_t idx = 0;
    int64_t *to = to_opt.has_value() ? &to_opt.value() : nullptr;

    ::diopiError_t ret = ::diopiRandomInp(&context, out_diopi, from, to, idx);
    TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
        " diopiRandomInp error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
    return self;
}

at::Tensor & DIPUATenFunctions::random_(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
    int64_t from = 0;
    c10::optional<int64_t> to_opt(to);
    return randomInp(self, from, to_opt, generator);
}

at::Tensor & DIPUATenFunctions::random_(at::Tensor & self, c10::optional<at::Generator> generator) {
    int64_t from = 0;
    c10::optional<int64_t> to = c10::nullopt;
    return randomInp(self, from, to, generator);
}

at::Tensor & DIPUATenFunctions::random_(at::Tensor & self, int64_t from, c10::optional<int64_t> to_opt, c10::optional<at::Generator> generator) {
    return randomInp(self, from, to_opt, generator);
}

}  // namespace dipu::native