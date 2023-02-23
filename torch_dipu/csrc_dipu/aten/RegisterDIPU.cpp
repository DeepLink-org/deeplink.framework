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

at::Tensor wrapperConvolution2d(
    const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    return dipu::native::DIPUNativeFunctions::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor & wrapperGeneratorOutRandpermOut(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
    return dipu::native::DIPUNativeFunctions::randperm_out(n, generator, out);
}

at::Tensor & wrapperOutRandpermOut(int64_t n, at::Tensor & out) {
  return dipu::native::DIPUNativeFunctions::randperm_out(n, out);
}

at::Tensor & wrapperFromRandomInp(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
  return dipu::native::DIPUNativeFunctions::random_(self, from, to, generator);
}

at::Tensor & wrapperToRandomInp(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
  return dipu::native::DIPUNativeFunctions::random_(self, to, generator);
}

at::Tensor & wrapperRandomInp(at::Tensor & self, c10::optional<at::Generator> generator) {
  return dipu::native::DIPUNativeFunctions::random_(self, generator);
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
    DIPU_LIBRARY_IMPL("conv2d", diopiConvolution2d, wrapperConvolution2d);
    DIPU_LIBRARY_IMPL("randperm.generator_out", diopiRandperm, wrapperGeneratorOutRandpermOut);
    DIPU_LIBRARY_IMPL("randperm.out", diopiRandperm, wrapperOutRandpermOut);
    DIPU_LIBRARY_IMPL("random_.from", diopiRandomInp, wrapperFromRandomInp);
    DIPU_LIBRARY_IMPL("random_.to", diopiRandomInp, wrapperToRandomInp);
    DIPU_LIBRARY_IMPL("random_", diopiRandomInp, wrapperRandomInp);
}

}  // outer anonymous namespace

}  // namespace at