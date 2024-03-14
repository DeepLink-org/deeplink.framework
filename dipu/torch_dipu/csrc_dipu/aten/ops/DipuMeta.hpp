#include <cstdint>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ops/add_meta.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/BinaryOps.h>


#include <iostream>
#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/ops/DIPUAmp.hpp"

class InferConfig {};
#define DIPU_BINARY_OP_CONFIG()             \
  InferConfig()

#define DIPU_TORCH_META_FUNC2(name, overload) \
  void Infer_##name##_##overload::meta


class Infer {
  public:
    Infer& add_input(const at::Tensor* p_tensor) {
      p_tensors_.push_back(p_tensor);
      return *this;
    }

    Infer& set_config(const InferConfig& config){
      config_ = config;
      return *this;
    }

    void build(){
      compute_common_dtype();
      compute_shape();
    }
    c10::IntArrayRef target_shape() { return target_shape_; }
    at::ScalarType common_dtype() { return common_dtype_; }

   private:
    void compute_common_dtype();
    void compute_shape();

    c10::SmallVector<const at::Tensor*, 4> p_tensors_={};
    c10::IntArrayRef target_shape_={};
    at::ScalarType common_dtype_ = at::ScalarType::Undefined;
    InferConfig config_{};
};

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


struct Infer_add_Tensor final : public Infer {
    void meta(const at::Tensor& self, const at::Tensor& other,
              const at::Scalar& alpha);
};

DIPU_TORCH_META_FUNC2(add, Tensor)
(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  set_config(DIPU_BINARY_OP_CONFIG()).add_input(&self).add_input(&other).build();
}
