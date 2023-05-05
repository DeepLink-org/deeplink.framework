
// autogened file
#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/diopirt/diopirt_impl.h"

namespace dipu::native {

using at::Tensor;
using at::Scalar;

using namespace dipu::diopi_helper;

//  exampleop.overloadname(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_exampleop_overloadname(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    /* Here can be a piece of c++ code at the begining*/

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);
    ::diopiScalar_t alphaDiopiScalar = dipu::diopi_helper::toDiopiScalar(alpha);

    std::cout << "self:" << self << std::endl;
    std::cout << "other:" << other << std::endl;

    ::diopiError_t ret = diopiAddScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAddScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    dipu::getCurrentDIPUStream().synchronize();
    std::cout << "out:" << out << std::endl;

    return out;
}

//  aten::add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_add_scalar_out(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);
    ::diopiScalar_t alphaDiopiScalar = dipu::diopi_helper::toDiopiScalar(alpha);

    ::diopiError_t ret = diopiAddScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAddScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_add_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        return dipu_add_scalar_out(self, other.item(), alpha, out);
    } else if (self.numel() == 1 && self.is_cpu()) {
        return dipu_add_scalar_out(other, self.item(), alpha, out);
    }

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t alphaDiopiScalar = dipu::diopi_helper::toDiopiScalar(alpha);

    ::diopiError_t ret = diopiAdd(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, &alphaDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAdd(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, &alphaDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_sub_scalar_out(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);
    ::diopiScalar_t alphaDiopiScalar = dipu::diopi_helper::toDiopiScalar(alpha);

    ::diopiError_t ret = diopiSubScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiSubScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, &alphaDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_sub_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        return dipu_sub_scalar_out(self, other.item(), alpha, out);
    } else if (self.numel() == 1 && self.is_cpu()) {
        return dipu_sub_scalar_out(other, self.item(), alpha, out);
    }

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t alphaDiopiScalar = dipu::diopi_helper::toDiopiScalar(alpha);

    ::diopiError_t ret = diopiSub(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, &alphaDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiSub(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, &alphaDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  div.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor dipu_div_scalar(const at::Tensor& self, const at::Scalar& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    auto out = at::empty_like(self);

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);

    ::diopiError_t ret = diopiDivScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, RoundModeNone);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDivScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, RoundModeNone);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor& dipu_div__scalar(at::Tensor& self, const at::Scalar& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);

    ::diopiError_t ret = diopiDivInpScalar(ctx, selfDiopiTensorHandle, &otherDiopiScalar, RoundModeNone);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDivInpScalar(ctx, selfDiopiTensorHandle, &otherDiopiScalar, RoundModeNone);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor& dipu_div__tensor(at::Tensor& self, const at::Tensor& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        return dipu_div__scalar(self, other.item());
    }

    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiError_t ret = diopiDivInp(ctx, selfDiopiTensorHandle, otherDiopiTensorHandle, RoundModeNone);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDivInp(ctx, selfDiopiTensorHandle, otherDiopiTensorHandle, RoundModeNone);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        out = dipu_div_scalar(self, other.item());
        return out;
    } else if (self.numel() == 1 && self.is_cpu()) {
        out = dipu_div_scalar(other, self.item());
        return out;
    }

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiDiv(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, RoundModeNone);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDiv(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, RoundModeNone);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_div_scalar_mode_out(const at::Tensor& self, const at::Scalar& other, c10::optional<c10::string_view> rounding_mode, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    const auto mode = toDiopiRoundMode(rounding_mode.has_value() ? rounding_mode.value().data():"none");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);

    ::diopiError_t ret = diopiDivScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, mode);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDivScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar, mode);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_div_out_mode(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        return dipu_div_scalar_mode_out(self, other.item(), rounding_mode, out);
    } else if (self.numel() == 1 && self.is_cpu()) {
        return dipu_div_scalar_mode_out(other, self.item(), rounding_mode, out);
    } const auto mode = toDiopiRoundMode(rounding_mode.has_value() ? rounding_mode.value().data():"none");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiDiv(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, mode);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiDiv(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle, mode);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor& dipu_fill__scalar(at::Tensor& self, const at::Scalar& value) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiScalar_t valueDiopiScalar = dipu::diopi_helper::toDiopiScalar(value);

    ::diopiError_t ret = diopiFill(ctx, selfDiopiTensorHandle, &valueDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiFill(ctx, selfDiopiTensorHandle, &valueDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  mul.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor dipu_mul_scalar(const at::Tensor& self, const at::Scalar& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    auto out = at::empty_like(self);

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);

    ::diopiError_t ret = diopiMulScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiMulScalar(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, &otherDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor& dipu_mul__scalar(at::Tensor& self, const at::Scalar& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiScalar_t otherDiopiScalar = dipu::diopi_helper::toDiopiScalar(other);

    ::diopiError_t ret = diopiMulInpScalar(ctx, selfDiopiTensorHandle, &otherDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiMulInpScalar(ctx, selfDiopiTensorHandle, &otherDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor& dipu_mul__tensor(at::Tensor& self, const at::Tensor& other) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        return dipu_mul__scalar(self, other.item());
    }

    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiError_t ret = diopiMulInp(ctx, selfDiopiTensorHandle, otherDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiMulInp(ctx, selfDiopiTensorHandle, otherDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    if (other.numel() == 1 && other.is_cpu()) {
        out = dipu_mul_scalar(self, other.item());
        return out;
    } else if (self.numel() == 1 && self.is_cpu()) {
        out = dipu_mul_scalar(other, self.item());
        return out;
    }

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t otherDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(other);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiMul(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiMul(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, otherDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> dipu_native_batch_norm_out(const at::Tensor& input, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, at::Tensor& out, at::Tensor& save_mean, at::Tensor& save_invstd) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiConstTensorHandle_t inputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(input);
    
    ::diopiConstTensorHandle_t running_varDiopiTensorHandle = nullptr;
    if (running_var.has_value() && running_var.value().defined()) running_varDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_var.value());
    
    
    ::diopiConstTensorHandle_t running_meanDiopiTensorHandle = nullptr;
    if (running_mean.has_value() && running_mean.value().defined()) running_meanDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_mean.value());
    
    
    ::diopiConstTensorHandle_t weightDiopiTensorHandle = nullptr;
    if (weight.has_value() && weight.value().defined()) weightDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(weight.value());
    
    
    ::diopiConstTensorHandle_t biasDiopiTensorHandle = nullptr;
    if (bias.has_value() && bias.value().defined()) biasDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(bias.value());

    ::diopiTensorHandle_t save_meanDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(save_mean);
    ::diopiTensorHandle_t save_invstdDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(save_invstd);
    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiBatchNorm(ctx, outDiopiTensorHandle, save_meanDiopiTensorHandle, save_invstdDiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, biasDiopiTensorHandle, const_cast<diopiTensorHandle_t>(running_meanDiopiTensorHandle), const_cast<diopiTensorHandle_t>(running_varDiopiTensorHandle), training, momentum, eps);;
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiBatchNorm(ctx, outDiopiTensorHandle, save_meanDiopiTensorHandle, save_invstdDiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, biasDiopiTensorHandle, const_cast<diopiTensorHandle_t>(running_meanDiopiTensorHandle), const_cast<diopiTensorHandle_t>(running_varDiopiTensorHandle), training, momentum, eps);;' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return std::tie(out, save_mean, save_invstd);
}

//  aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor> dipu_native_batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    const int64_t dim_c = input.size(1);
    auto out0 = at::empty_like(input);
    auto options = input.options().dtype(at::kFloat);
    auto out1 = at::empty({dim_c}, options);
    auto out2 = at::empty({dim_c}, options);

    ::diopiConstTensorHandle_t inputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(input);
    
    ::diopiConstTensorHandle_t running_varDiopiTensorHandle = nullptr;
    if (running_var.has_value() && running_var.value().defined()) running_varDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_var.value());
    
    
    ::diopiConstTensorHandle_t running_meanDiopiTensorHandle = nullptr;
    if (running_mean.has_value() && running_mean.value().defined()) running_meanDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_mean.value());
    
    
    ::diopiConstTensorHandle_t weightDiopiTensorHandle = nullptr;
    if (weight.has_value() && weight.value().defined()) weightDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(weight.value());
    
    
    ::diopiConstTensorHandle_t biasDiopiTensorHandle = nullptr;
    if (bias.has_value() && bias.value().defined()) biasDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(bias.value());

    ::diopiTensorHandle_t out2DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out2);
    ::diopiTensorHandle_t out1DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out1);
    ::diopiTensorHandle_t out0DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out0);

    ::diopiError_t ret = diopiBatchNorm(ctx, out0DiopiTensorHandle, out1DiopiTensorHandle, out2DiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, biasDiopiTensorHandle, const_cast<diopiTensorHandle_t>(running_meanDiopiTensorHandle), const_cast<diopiTensorHandle_t>(running_varDiopiTensorHandle), training, momentum, eps);;
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiBatchNorm(ctx, out0DiopiTensorHandle, out1DiopiTensorHandle, out2DiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, biasDiopiTensorHandle, const_cast<diopiTensorHandle_t>(running_meanDiopiTensorHandle), const_cast<diopiTensorHandle_t>(running_varDiopiTensorHandle), training, momentum, eps);;' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return std::tie(out0, out1, out2);
}

//  native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor> dipu_native_batch_norm_backward(const at::Tensor& grad_out, const at::Tensor& input, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var, const c10::optional<at::Tensor>& save_mean, const c10::optional<at::Tensor>& save_invstd, bool train, double eps, ::std::array<bool, 3> output_mask) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    int64_t dim_c = input.size(1);
    auto options = input.options().dtype(at::kFloat);
    at::Tensor out0 = at::empty(input.sizes(), input.options());
    at::Tensor out1 = at::empty({dim_c}, options);
    at::Tensor out2 = at::empty({dim_c}, options);

    ::diopiConstTensorHandle_t inputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(input);
    ::diopiConstTensorHandle_t grad_outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(grad_out);
    
    ::diopiConstTensorHandle_t running_varDiopiTensorHandle = nullptr;
    if (running_var.has_value() && running_var.value().defined()) running_varDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_var.value());
    
    
    ::diopiConstTensorHandle_t save_invstdDiopiTensorHandle = nullptr;
    if (save_invstd.has_value() && save_invstd.value().defined()) save_invstdDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(save_invstd.value());
    
    
    ::diopiConstTensorHandle_t running_meanDiopiTensorHandle = nullptr;
    if (running_mean.has_value() && running_mean.value().defined()) running_meanDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(running_mean.value());
    
    
    ::diopiConstTensorHandle_t save_meanDiopiTensorHandle = nullptr;
    if (save_mean.has_value() && save_mean.value().defined()) save_meanDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(save_mean.value());
    
    
    ::diopiConstTensorHandle_t weightDiopiTensorHandle = nullptr;
    if (weight.has_value() && weight.value().defined()) weightDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(weight.value());

    ::diopiTensorHandle_t out2DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out2);
    ::diopiTensorHandle_t out1DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out1);
    ::diopiTensorHandle_t out0DiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out0);

    ::diopiError_t ret = diopiBatchNormBackward(ctx, out0DiopiTensorHandle, out1DiopiTensorHandle, out2DiopiTensorHandle, grad_outDiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, running_meanDiopiTensorHandle, running_varDiopiTensorHandle, save_meanDiopiTensorHandle, save_invstdDiopiTensorHandle, train, eps);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiBatchNormBackward(ctx, out0DiopiTensorHandle, out1DiopiTensorHandle, out2DiopiTensorHandle, grad_outDiopiTensorHandle, inputDiopiTensorHandle, weightDiopiTensorHandle, running_meanDiopiTensorHandle, running_varDiopiTensorHandle, save_meanDiopiTensorHandle, save_invstdDiopiTensorHandle, train, eps);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return std::tie(out0, out1, out2);
}

//  adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_adaptive_avg_pool2d_out(const at::Tensor& self, c10::SymIntArrayRef output_size, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    auto symIntToInt = [](const c10::SymInt& t)-> int64_t {return t.expect_int();};
    std::vector<int64_t> output_sizeVector(output_size.size());
    std::transform(output_size.cbegin(), output_size.cend(), output_sizeVector.begin(), symIntToInt);
    ::diopiSize_t output_sizeDiopiSize(output_sizeVector.data(), output_sizeVector.size());

    ::diopiError_t ret = diopiAdaptiveAvgPool2d(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, output_sizeDiopiSize);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAdaptiveAvgPool2d(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, output_sizeDiopiSize);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  _adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
at::Tensor dipu__adaptive_avg_pool2d(const at::Tensor& self, c10::SymIntArrayRef output_size) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    TORCH_CHECK(output_size.size() == 2, __func__, ":", __FILE__, ":", __LINE__, 
        " output_size should equal 2, size is ", output_size.size());
    auto out_tensor_size = self.sizes().vec();
    out_tensor_size[self.dim() - 2] = output_size[0].expect_int();
    out_tensor_size[self.dim() - 1] = output_size[1].expect_int();
    at::Tensor out = at::empty(out_tensor_size, self.options());

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    auto symIntToInt = [](const c10::SymInt& t)-> int64_t {return t.expect_int();};
    std::vector<int64_t> output_sizeVector(output_size.size());
    std::transform(output_size.cbegin(), output_size.cend(), output_sizeVector.begin(), symIntToInt);
    ::diopiSize_t output_sizeDiopiSize(output_sizeVector.data(), output_sizeVector.size());

    ::diopiError_t ret = diopiAdaptiveAvgPool2d(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, output_sizeDiopiSize);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAdaptiveAvgPool2d(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, output_sizeDiopiSize);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  _adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor dipu__adaptive_avg_pool2d_backward(const at::Tensor& grad_output, const at::Tensor& self) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    auto out = at::empty_like(self);

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t grad_outputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(grad_output);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiAdaptiveAvgPool2dBackward(ctx, outDiopiTensorHandle, grad_outputDiopiTensorHandle, selfDiopiTensorHandle);;
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiAdaptiveAvgPool2dBackward(ctx, outDiopiTensorHandle, grad_outputDiopiTensorHandle, selfDiopiTensorHandle);;' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  relu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor& dipu_relu_(at::Tensor& self) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiError_t ret = diopiReluInp(ctx, selfDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiReluInp(ctx, selfDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  relu(Tensor self) -> Tensor
at::Tensor dipu_relu(const at::Tensor& self) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    auto out = at::empty_like(self);

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiRelu(ctx, outDiopiTensorHandle, selfDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiRelu(ctx, outDiopiTensorHandle, selfDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_randperm_out(int64_t n, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiRandperm(ctx, outDiopiTensorHandle, n, 0);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiRandperm(ctx, outDiopiTensorHandle, n, 0);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_randperm_generator_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    const int64_t seed = (generator.has_value() && generator.value().defined()) ? generator.value().seed() : 0;

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiRandperm(ctx, outDiopiTensorHandle, n, seed);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiRandperm(ctx, outDiopiTensorHandle, n, seed);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_sum_intlist_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    ::diopiSize_t diopi_size = toDiopiSize(dim);

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiSum(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, diopi_size);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiSum(ctx, outDiopiTensorHandle, selfDiopiTensorHandle, diopi_size);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_floor_out(const at::Tensor& self, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "floor.out");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiError_t ret = diopiFloor(ctx, outDiopiTensorHandle, selfDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiFloor(ctx, outDiopiTensorHandle, selfDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::floor_(Tensor(a!) self) -> Tensor(a!)
at::Tensor& dipu_floor_(at::Tensor& self) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "floor_");

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiError_t ret = diopiFloorInp(ctx, selfDiopiTensorHandle);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiFloorInp(ctx, selfDiopiTensorHandle);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
at::Tensor dipu_hardtanh(const at::Tensor& self, const at::Scalar& min_val=-1, const at::Scalar& max_val) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "hardtanh");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t min_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(min_val);
    ::diopiScalar_t max_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(max_val);

    ::diopiError_t ret = diopiHardtanh(ctx, outDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiHardtanh(ctx, outDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& dipu_hardtanh_out(const at::Tensor& self, const at::Scalar& min_val=-1, const at::Scalar& max_val, at::Tensor& out) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "hardtanh.out");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t min_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(min_val);
    ::diopiScalar_t max_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(max_val);

    ::diopiError_t ret = diopiHardtanh(ctx, outDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiHardtanh(ctx, outDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

//  aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
at::Tensor& dipu_hardtanh_(at::Tensor& self, const at::Scalar& min_val=-1, const at::Scalar& max_val) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "hardtanh_");

    ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);

    ::diopiScalar_t min_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(min_val);
    ::diopiScalar_t max_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(max_val);

    ::diopiError_t ret = diopiHardtanhInp(ctx, input, &min_valDiopiScalar, &max_valDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiHardtanhInp(ctx, input, &min_valDiopiScalar, &max_valDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return self;
}

//  aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor& dipu_hardtanh_backward_grad_input(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val, at::Tensor& grad_input) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "hardtanh_backward.grad_input");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t grad_outputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(grad_output);

    ::diopiTensorHandle_t grad_inputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(grad_input);

    ::diopiScalar_t min_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(min_val);
    ::diopiScalar_t max_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(max_val);

    ::diopiError_t ret = diopiHardtanhBackward(ctx, grad_inputDiopiTensorHandle, grad_outputDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiHardtanhBackward(ctx, grad_inputDiopiTensorHandle, grad_outputDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return grad_input;
}

//  aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
at::Tensor dipu_hardtanh_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val) {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    printf("[%s:%s:%d]:%s\n", __FILE__, __FUNCTION__, __LINE__, "hardtanh_backward");

    ::diopiConstTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
    ::diopiConstTensorHandle_t grad_outputDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(grad_output);

    ::diopiTensorHandle_t outDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(out);

    ::diopiScalar_t min_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(min_val);
    ::diopiScalar_t max_valDiopiScalar = dipu::diopi_helper::toDiopiScalar(max_val);

    ::diopiError_t ret = diopiHardtanhBackward(ctx, grad_input, grad_outputDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, "'diopiHardtanhBackward(ctx, grad_input, grad_outputDiopiTensorHandle, input, &min_valDiopiScalar, &max_valDiopiScalar);' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    return out;
}

}  // namespace dipu::native

namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
    
    DIOPI_ATEN_FUNC("add.Scalar_out", ::diopiAddScalar, dipu::native::dipu_add_scalar_out);
    
    DIOPI_ATEN_FUNC("add.out", ::diopiAdd, dipu::native::dipu_add_out);
    
    DIOPI_ATEN_FUNC("sub.Scalar_out", ::diopiSubScalar, dipu::native::dipu_sub_scalar_out);
    
    DIOPI_ATEN_FUNC("sub.out", ::diopiSub, dipu::native::dipu_sub_out);
    
    DIOPI_ATEN_FUNC("div.Scalar", ::diopiDivScalar, dipu::native::dipu_div_scalar);
    
    DIOPI_ATEN_FUNC("div_.Scalar", ::diopiDivInpScalar, dipu::native::dipu_div__scalar);
    
    DIOPI_ATEN_FUNC("div_.Tensor", ::diopiDivInp, dipu::native::dipu_div__tensor);
    
    DIOPI_ATEN_FUNC("div.out", ::diopiDiv, dipu::native::dipu_div_out);
    
    DIOPI_ATEN_FUNC("div.Scalar_mode_out", ::diopiDivScalar, dipu::native::dipu_div_scalar_mode_out);
    
    DIOPI_ATEN_FUNC("div.out_mode", ::diopiDiv, dipu::native::dipu_div_out_mode);
    
    DIOPI_ATEN_FUNC("fill_.Scalar", ::diopiFill, dipu::native::dipu_fill__scalar);
    
    DIOPI_ATEN_FUNC("mul.Scalar", ::diopiMulScalar, dipu::native::dipu_mul_scalar);
    
    DIOPI_ATEN_FUNC("mul_.Scalar", ::diopiMulInpScalar, dipu::native::dipu_mul__scalar);
    
    DIOPI_ATEN_FUNC("mul_.Tensor", ::diopiMulInp, dipu::native::dipu_mul__tensor);
    
    DIOPI_ATEN_FUNC("mul.out", ::diopiMul, dipu::native::dipu_mul_out);
    
    DIOPI_ATEN_FUNC("native_batch_norm.out", ::diopiBatchNorm, dipu::native::dipu_native_batch_norm_out);
    
    DIOPI_ATEN_FUNC("native_batch_norm", ::diopiBatchNorm, dipu::native::dipu_native_batch_norm);
    
    DIOPI_ATEN_FUNC("native_batch_norm_backward", ::diopiBatchNormBackward, dipu::native::dipu_native_batch_norm_backward);
    
    DIOPI_ATEN_FUNC("adaptive_avg_pool2d.out", ::diopiAdaptiveAvgPool2d, dipu::native::dipu_adaptive_avg_pool2d_out);
    
    DIOPI_ATEN_FUNC("_adaptive_avg_pool2d", ::diopiAdaptiveAvgPool2d, dipu::native::dipu__adaptive_avg_pool2d);
    
    DIOPI_ATEN_FUNC("_adaptive_avg_pool2d_backward", ::diopiAdaptiveAvgPool2dBackward, dipu::native::dipu__adaptive_avg_pool2d_backward);
    
    DIOPI_ATEN_FUNC("relu_", ::diopiReluInp, dipu::native::dipu_relu_);
    
    DIOPI_ATEN_FUNC("relu", ::diopiRelu, dipu::native::dipu_relu);
    
    DIOPI_ATEN_FUNC("randperm.out", ::diopiRandperm, dipu::native::dipu_randperm_out);
    
    DIOPI_ATEN_FUNC("randperm.generator_out", ::diopiRandperm, dipu::native::dipu_randperm_generator_out);
    
    DIOPI_ATEN_FUNC("sum.IntList_out", ::diopiSum, dipu::native::dipu_sum_intlist_out);
    
    DIOPI_ATEN_FUNC("floor.out", ::diopiFloor, dipu::native::dipu_floor_out);
    
    DIOPI_ATEN_FUNC("floor_", ::diopiFloorInp, dipu::native::dipu_floor_);
    
    DIOPI_ATEN_FUNC("hardtanh", ::diopiHardtanh, dipu::native::dipu_hardtanh);
    
    DIOPI_ATEN_FUNC("hardtanh.out", ::diopiHardtanh, dipu::native::dipu_hardtanh_out);
    
    DIOPI_ATEN_FUNC("hardtanh_", ::diopiHardtanhInp, dipu::native::dipu_hardtanh_);
    
    DIOPI_ATEN_FUNC("hardtanh_backward.grad_input", ::diopiHardtanhBackward, dipu::native::dipu_hardtanh_backward_grad_input);
    
    DIOPI_ATEN_FUNC("hardtanh_backward", ::diopiHardtanhBackward, dipu::native::dipu_hardtanh_backward);
}

}  // namespace at

