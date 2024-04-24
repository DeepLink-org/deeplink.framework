// Copyright (c) 2024, DeepLink.
#pragma once

#include <ATen/ATen.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

using DimVector = c10::SmallVector<int64_t, 4>;
using StrideVector = c10::SmallVector<int64_t, 6>;

// Base Class for inferring the shape, dtype, and memory format of output Tensor
// based on its inputs, then malloc the output tensor.
class OpInferrer {
 public:
  OpInferrer() = default;
  virtual ~OpInferrer() = default;

  at::ScalarType common_dtype() const { return dtype_; }

  DimVector target_shape() const { return shape_; }

  at::MemoryFormat memory_format() const { return memory_format_; }

 protected:
  void add_inputs(const std::vector<at::Tensor>& inputs) {
    TORCH_CHECK(!inputs.empty(), "Input tensors must not be empty");
    for (const auto& tensor : inputs) {
      TORCH_CHECK(tensor.defined(), "Input tensor is undefined");
    }
    inputs_ = inputs;
  }
  // Computes the shape of the output, supporting broadcasting rules.
  void compute_shape();

  int ndim() const { return shape_.size(); }
  int ntensors() const { return inputs_.size(); }

  // Computes the dtype of the output based on all input tensors,
  // supporting dtype promotion
  void compute_dtype();

  bool fast_compute_memory_format();
  void calculate_perm();

  // Determine the best memory format for the output tensor based on input
  // tensors.
  void compute_memory_format();

  // Allocates the output Tensor based on the inferred attributes, if strides_
  // inferred, use it.
  at::Tensor malloc_output() {
    at::TensorOptions options =
        at::TensorOptions().dtype(dtype_).device(device_);
    auto out = native::nodispatch::empty(shape_, options, memory_format_);

    if (!strides_.empty()) {
      out.as_strided_(shape_, strides_);
    }
    return out;
  }

  bool all_same_shape_ = true;
  c10::SmallVector<at::Tensor, 4> inputs_;
  DimVector shape_;
  at::ScalarType dtype_ = at::ScalarType::Undefined;
  at::Device device_ = dipu::DIPU_DEVICE_TYPE;
  at::MemoryFormat memory_format_ = at::MemoryFormat::Contiguous;
  StrideVector strides_;
  DimVector perm_;
};

class BinaryOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other);
};

class BinaryFloatOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other);
};

class UnaryOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self);
};

class ComparisonOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other);
};

class ReduceOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, c10::OptionalIntArrayRef dim,
                       bool keep_dim, c10::optional<at::ScalarType> dtype);
};

// class MatrixOpInferrer final : public OpInferrer {
//   at::Tensor infer_out(const at::Tensor& self, c10::OptionalIntArrayRef dim,
//                        bool keep_dim, c10::optional<at::ScalarType> dtype);
// };

}  // namespace dipu
