#include <memory>

#include <ATen/Tensor.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;

namespace dipu::native {

::std::tuple<at::Tensor &,at::Tensor &> DIPUATenFunctions::max_pool2d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
  ::diopiSize_t kernel_size_diopi(kernel_size.data(), kernel_size.size());
  ::diopiSize_t stride_diopi(stride.data(), stride.size());
  ::diopiSize_t padding_diopi(padding.data(), padding.size());
  ::diopiSize_t dilation_diopi(dilation.data(), dilation.size());

  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);
  ::diopiTensorHandle_t indices_diopi = toDiopiTensorHandle(indices);

  ::diopiError_t ret = ::diopiMaxPool2dWithIndices(&context, out_diopi, indices_diopi, self_diopi,
      kernel_size_diopi, stride_diopi, padding_diopi, dilation_diopi, ceil_mode);
  TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
      " conv2d error, error code is ", ret, "\nerror message is ", diopiGetLastErrorString());
  return std::tie(out, indices);
}


}  // namespace dipu::native