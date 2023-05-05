#include <vector>
#include <memory>

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"
#include <torch/csrc/autograd/custom_function.h>
#include "csrc_dipu/aten/RegisterDIPU.hpp"



using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor dipu_cross_entropy_loss_impl(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing);
at::Tensor dipu_cross_entropy_loss_backward_impl(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing);

class DipuCrossEntropyLossFunction : public torch::autograd::Function<DipuCrossEntropyLossFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx, const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing) {
        ctx->saved_data["ignore_index"] = ignore_index.expect_int();
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["label_smoothing"] = label_smoothing;
        ctx->saved_data["weight"] = weight;

        at::AutoDispatchBelowADInplaceOrView g;
        ctx->save_for_backward({self, target});
        return dipu_cross_entropy_loss_impl(self, target, weight, reduction, ignore_index, label_smoothing);
    }

  static std::vector<at::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<at::Tensor> grad_outputs) {
      auto reduction = ctx->saved_data["reduction"].toInt();
      auto ignore_index = ctx->saved_data["ignore_index"].toInt();
      auto label_smoothing = ctx->saved_data["label_smoothing"].toDouble();
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto target = saved[1];
      c10::optional<at::Tensor>  weight = ctx->saved_data["weight"].toOptional<at::Tensor>();
      auto grad_output = grad_outputs.at(0);

      auto out = dipu_cross_entropy_loss_backward_impl(grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
      std::vector<at::Tensor> outputs(6);
      outputs[0] = out;
      return outputs;
  }
};

at::Tensor cross_entropy_loss(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction,
        c10::SymInt ignore_index, double label_smoothing) {
    return DipuCrossEntropyLossFunction::apply(self, target, weight, reduction, ignore_index, label_smoothing);
}

}  // namespace dipu::native

namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
    DIOPI_ATEN_FUNC("cross_entropy_loss", ::diopiCrossEntropyLoss, dipu::native::cross_entropy_loss);
}

}  // namespace at
