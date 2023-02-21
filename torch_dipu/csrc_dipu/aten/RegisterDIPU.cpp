#include <stdio.h>
#include <tuple>

#include <torch/library.h>

#include "csrc_dipu/aten/DIPUNativeFunctions.h"
#include "csrc_dipu/diopirt/diopi.h"
#include "csrc_dipu/aten/util/Log.h"

namespace at {

namespace {

namespace {

at::Tensor& wrapperTensorAddOut(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    return dipu::native::DIPUNativeFunctions::add_out(self, other, alpha, out);
}

at::Tensor wrapperRelu(const at::Tensor & self) {
    return dipu::native::DIPUNativeFunctions::relu(self);
}

at::Tensor& wrapperReluInp(at::Tensor & self) {
    return dipu::native::DIPUNativeFunctions::relu_(self);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapperNativeBatchNorm(
    const at::Tensor & input, const c10::optional<at::Tensor> & weight,
    const c10::optional<at::Tensor> & bias,
    const c10::optional<at::Tensor> & running_mean,
    const c10::optional<at::Tensor> & running_var,
    bool training, double momentum, double eps) {
    return dipu::native::DIPUNativeFunctions::native_batch_norm(input, weight,
        bias, running_mean, running_var, training, momentum, eps);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapperNativeBatchNormBackward(
    const at::Tensor & grad_out, const at::Tensor & input,
    const c10::optional<at::Tensor> & weight,
    const c10::optional<at::Tensor> & running_mean,
    const c10::optional<at::Tensor> & running_var,
    const c10::optional<at::Tensor> & save_mean,
    const c10::optional<at::Tensor> & save_invstd,
    bool train, double eps, ::std::array<bool,3> output_mask) {
    return dipu::native::DIPUNativeFunctions::native_batch_norm_backward(
        grad_out, input, weight, running_mean, running_var, save_mean,
        save_invstd, train, eps, output_mask);
}

}  // inner anonymous namespace

#define DIPU_LIBRARY_IMPL(opname, diopiFunc, wapperFunc) do {           \
    if (reinterpret_cast<void*>(diopiFunc) != nullptr) {                \
        m.impl(opname, TORCH_FN(wapperFunc));                           \
    }  else {                                                           \
        DIPU_LOG << #diopiFunc << " not implemented, do not register\n"; \
    }                                                                   \
} while (false);

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    DIPU_LIBRARY_IMPL("add.out", diopiAdd222, wrapperTensorAddOut);
    DIPU_LIBRARY_IMPL("add.out", diopiAdd, wrapperTensorAddOut);
    DIPU_LIBRARY_IMPL("relu", diopiRelu, wrapperRelu);
    DIPU_LIBRARY_IMPL("relu_", diopiReluInp, wrapperReluInp);
    DIPU_LIBRARY_IMPL("native_batch_norm", diopiBatchNorm, wrapperNativeBatchNorm);
    DIPU_LIBRARY_IMPL("native_batch_norm_backward", diopiBatchNormBackward, wrapperNativeBatchNormBackward);
}

}  // outer anonymous namespace

}  // namespace at