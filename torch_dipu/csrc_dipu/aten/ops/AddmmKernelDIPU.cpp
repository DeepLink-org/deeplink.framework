#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

namespace dipu::native {

at::Tensor& DIPUATenFunctions::addmm_out(
    const at::Tensor & self, const at::Tensor & mat1,
    const at::Tensor & mat2, const at::Scalar & beta,
    const at::Scalar & alpha, at::Tensor & out) {
  ::diopiConstTensorHandle_t self_diopi = toDiopiTensorHandle(self);
  ::diopiConstTensorHandle_t mat1_diopi = toDiopiTensorHandle(mat1);
  ::diopiConstTensorHandle_t mat2_diopi = toDiopiTensorHandle(mat2);
  ::diopiScalar_t beta_diopi = toDiopiScalar(beta);
  ::diopiScalar_t alpha_diopi = toDiopiScalar(alpha);
  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t out_diopi = toDiopiTensorHandle(out);

  ::diopiError_t ret = ::diopiAddmm(&context, out_diopi, self_diopi,
    mat1_diopi, mat2_diopi, &beta_diopi, &alpha_diopi);
  TORCH_CHECK(ret == ::diopiSuccess, __func__, ":", __FILE__, ":", __LINE__,
    " diopiAddmm error, error code is ", ret, "\nerror message is", diopiGetLastErrorString());
  return out;
}

}  // namespace dipu::native