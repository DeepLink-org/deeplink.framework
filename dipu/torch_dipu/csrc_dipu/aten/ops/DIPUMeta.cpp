#include "DIPUMeta.hpp"

#include <ATen/ExpandUtils.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/BinaryOps.h>
namespace dipu {
namespace native {

Infer& Infer::add_input(const at::Tensor* p_tensor) {
  p_tensors_.push_back(p_tensor);
  return *this;
}
Infer& Infer::set_config(const InferConfig& config) {
  config_ = config;
  return *this;
}
void Infer::build() {
  compute_common_dtype();
  compute_shape();
}
void Infer::compute_common_dtype() {
  at::native::ResultTypeState state = {};
  for (const auto p_tensor : p_tensors_) {
    state = at::native::update_result_type_state(*p_tensor, state);
  }

  common_dtype_ = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype_ != at::ScalarType::Undefined);
}

void Infer::compute_shape() {
  bool all_ops_same_shape = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (const auto p_tensor : p_tensors_) {
    if (!p_tensor->defined()) {
      continue;
    }

    auto shape = p_tensor->sizes();
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape = false;
    }
    if (target_shape_.empty()) {
      target_shape_ = shape;
    } else if (!shape.equals(target_shape_)) {
      all_ops_same_shape = false;
      target_shape_ = at::infer_size_dimvector(target_shape_, shape);
    }
  }
}


// meta impl of binary op
/*************************** start binary op's meta ******************************/
DIPU_TORCH_META_FUNC(add, Tensor)
(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  set_config(DIPU_BINARY_OP_CONFIG())
      .add_input(&self)
      .add_input(&other)
      .build();
   at::native::alpha_check(common_dtype(), alpha);
}

DIPU_TORCH_META_FUNC(sub, Tensor)
(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  at::native::sub_check(self, other);
  set_config(DIPU_BINARY_OP_CONFIG())
      .add_input(&self)
      .add_input(&other)
      .build();
 at::native::alpha_check(common_dtype(), alpha);
}



/**************************** end binary op's meta ****************************************/



}  // namespace native
}  // namespace dipu
