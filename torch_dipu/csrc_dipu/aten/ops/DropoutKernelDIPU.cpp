#include <vector>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/diopirt/diopirt_impl.h"
#include <torch/csrc/autograd/custom_function.h>

namespace dipu::native {

std::tuple<at::Tensor&, at::Tensor&> dipu_dropout_impl(const at::Tensor& input,
    double p, bool train, at::Tensor& out, at::Tensor& mask);
std::tuple<at::Tensor&, at::Tensor&> dipu_dropout__impl(
    at::Tensor& input, at::Tensor& mask, double p, bool train);

class DipuDropoutFunction
    : public torch::autograd::Function<DipuDropoutFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        const at::Tensor& input, at::Tensor& out, double p, bool train, bool inplace) {
        at::AutoDispatchBelowADInplaceOrView g;
        auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
        if (inplace) {
            dipu_dropout__impl(out, mask, p, train);
        } else {
            dipu_dropout_impl(input, p, train, out, mask);
        }
        if (train) {
            ctx->saved_data["p"] = p;
            ctx->saved_data["out"] = out;
            ctx->saved_data["mask"] = mask;
        }
        return out;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs) {
        auto p = ctx->saved_data["p"].toDouble();
        double p1m = 1. - p;
        // Check for probability of zero to avoid divide by zero and NaN results
        double scale = p1m == 0 ? 0. : 1. / p1m;

        auto mask = ctx->saved_data["mask"].toTensor();

        at::Tensor out = grad_outputs[0] * mask * scale;

        std::vector<at::Tensor> outputs(6);
        outputs[0] = out;
        return outputs;
    }
};

at::Tensor dipu_dropout(const at::Tensor& input, double p, bool train) {
    at::Tensor out = at::empty_like(input);
    DipuDropoutFunction::apply(input, out, p, train, false);
    return out;
}

at::Tensor& dipu__dropout(at::Tensor& input, double p, bool train) {
    DipuDropoutFunction::apply(input, input, p, train, true);
    return input;
}

}  // namespace dipu::native

namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
    DIOPI_ATEN_FUNC("dropout", ::diopiDropout, dipu::native::dipu_dropout);
    DIOPI_ATEN_FUNC("dropout_", ::diopiDropoutInp, dipu::native::dipu__dropout);
}

}  // namespace at
