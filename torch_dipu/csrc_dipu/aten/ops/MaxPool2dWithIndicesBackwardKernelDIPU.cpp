#include <memory>

#include <ATen/Tensor.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::max_pool2d_with_indices_backward_out_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  ::diopiConstTensorHandle_t grad_output_diopi = toDiopiTensorHandle(grad_output);
  ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
  ::diopiSize_t kernel_size_diopi(kernel_size.data(), kernel_size.size());
  ::diopiSize_t stride_diopi(stride.data(), stride.size());
  ::diopiSize_t padding_diopi(padding.data(), padding.size());
  ::diopiSize_t dilation_diopi(dilation.data(), dilation.size());
  ::diopiConstTensorHandle_t indices_diopi = toDiopiTensorHandle(indices);

  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t grad_input_diopi = toDiopiTensorHandle(grad_input);

  ::diopiError_t ret = ::diopiMaxPool2dBackward(&context, grad_input_diopi, grad_output_diopi, self_diopi,
      kernel_size_diopi, stride_diopi, padding_diopi, dilation_diopi, ceil_mode, indices_diopi);
  TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
      " diopiMaxPool2dBackward error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
  return grad_input;
}


}  // namespace dipu::native