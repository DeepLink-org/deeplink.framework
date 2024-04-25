// Copyright (c) 2024, DeepLink.
#pragma once

#include <ATen/ATen.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

constexpr size_t kMaxTensorDimensions = 6;

using DimVector = c10::SmallVector<int64_t, kMaxTensorDimensions>;
using StrideVector = c10::SmallVector<int64_t, kMaxTensorDimensions>;

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
  void add_input(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.defined(), "Input tensor is undefined");
    inputs_.push_back(tensor);
  }

  size_t ndim() const { return shape_.size(); }
  size_t ntensors() const { return inputs_.size(); }

  virtual void compute_shape();
  virtual void compute_dtype();
  virtual void compute_memory_format();

  // Allocates the output Tensor based on the inferred attributes, if strides_
  // inferred, use it.
  at::Tensor malloc_output() {
    at::TensorOptions options =
        at::TensorOptions().dtype(dtype_).device(dipu::DIPU_DEVICE_TYPE);
    auto out = native::nodispatch::empty(shape_, options, memory_format_);

    if (!strides_.empty()) {
      out.as_strided_(shape_, strides_);
    }
    return out;
  }

  // TODO(ywt): we may use c10::MaybeOwned to improve efficiency.
  c10::SmallVector<at::Tensor, 4> inputs_;
  DimVector shape_;
  at::ScalarType dtype_ = at::ScalarType::Undefined;
  at::MemoryFormat memory_format_ = at::MemoryFormat::Contiguous;

 private:
  // common logic for calculation, not inherited by children.
  bool fast_compute_memory_format();
  void compute_perm();
  void compute_broadcast_shape(c10::IntArrayRef shape);
  std::vector<StrideVector> compute_effective_strides();

  bool all_same_shape_ = true;
  DimVector perm_;
  StrideVector strides_;
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

class LogicOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other);
};

class ReduceOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::Tensor& self, c10::OptionalIntArrayRef dim,
                       bool keep_dim, c10::optional<at::ScalarType> dtype);

 protected:
  void compute_shape(c10::OptionalIntArrayRef dim, bool keep_dim);
  void compute_dtype() override;
};

class CatOpInferrer final : public OpInferrer {
 public:
  at::Tensor infer_out(const at::ITensorListRef& tensors, int64_t dim);

 protected:
  void compute_memory_format() override;
  void compute_shape(int64_t dim);

  static inline bool cat_should_skip_tensor(const at::Tensor& t) {
    return t.numel() == 0 && t.dim() == 1;
  }

  void check_cat_shape_except_dim(size_t index, size_t index_2,
                                  int64_t dimension);
};

}  // namespace dipu
