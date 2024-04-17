// Copyright (c) 2024, DeepLink.
#pragma once

#include <ATen/ATen.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

using DimVector = c10::SmallVector<int64_t, 4>;

// Designed for inferring the shape and dtype of an output Tensor
// based on its inputs, then malloc the output tensor.
// [Usage of TensorInferer]
// for binary op:
//    auto out = TensorInferer().add_input(t1).add_input(t2).infer_binary_op();
// for unary op:
//    auto out = TensorInferer().add_input(t).infer_unary_op();
// for reduce op:
//    auto out = TensorInferer().add_input(self).infer_reduce_dim_op(dim,
//    keepdim, dtype);
// for compare op:
//    auto out = TensorInferer().add_input(t).infer_comparison_op();
// for martix op:
//    auto out =
//    TensorInferer().add_input(self).add_input(mat2).infer_matrix_op(); auto
//    out =
//    TensorInferer().add_input(self).add_input(mat1).add_input(mat2).infer_matrix_op();
class TensorInferer {
 public:
  TensorInferer() = default;
  ~TensorInferer() = default;

  TensorInferer& add_input(const at::Tensor& tensor) {
    // inputs_.push_back(c10::MaybeOwned<at::Tensor>::borrowed(tensor));
    inputs_.push_back(tensor);
    return *this;
  }
  at::Tensor infer_binary_op();
  at::Tensor infer_binary_float_op();
  at::Tensor infer_unary_op();
  at::Tensor infer_unary_float_op();
  at::Tensor infer_reduce_op(
      c10::OptionalIntArrayRef dim, bool keep_dim = false,
      c10::optional<at::ScalarType> dtype = c10::nullopt);
  at::Tensor infer_comparison_op();
  at::Tensor infer_matrix_op();
  at::Tensor infer_cat(int64_t dim);

 private:
  // Computes the shape of the output, supporting broadcasting rules.
  void compute_shape();

  // Computes the dtype of the output based on all input tensors,
  // supporting dtype promotion
  void compute_dtype();

  // Allocates the output Tensor based on the inferred shape and dtype.
  at::Tensor malloc_output();

  // Member Variables
  c10::SmallVector<at::Tensor, 4> inputs_;
  DimVector shape_;
  at::ScalarType dtype_ = at::ScalarType::Undefined;
  at::Device device_ = dipu::DIPU_DEVICE_TYPE;
};

}  // namespace dipu
