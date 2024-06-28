// Copyright (c) 2024, DeepLink.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

namespace native {
// coumpute broadcast shape based on two inputs
// for example: a = [2, 1, 3], b = [2, 1], the result shape would be [2, 2, 3]
at::DimVector compute_broadcast_shape(c10::IntArrayRef a, c10::IntArrayRef b);

}  // namespace native

// This class is intended as a base class only and should not be instantiated
// directly.
class OpInferrerMeta {
 public:
  virtual ~OpInferrerMeta() = default;
  at::ScalarType common_dtype() const { return dtype_; }
  at::Device common_device() const { return device_; }
  c10::DimVector target_shape() const { return shape_; }
  at::MemoryFormat memory_format() const { return memory_format_; }
  void compute_device();

 protected:
  // Protected constructor to prevent direct instantiation
  OpInferrerMeta() = default;

  void add_input(const at::Tensor& tensor);

  const at::Tensor& tensor(size_t idx) { return *inputs_[idx]; }

  size_t ndim() const { return shape_.size(); }
  size_t ntensors() const { return inputs_.size(); }

  // Allocates the output based on the inferred attributes, use strides_ if set
  inline at::Tensor malloc_output();

  c10::SmallVector<c10::MaybeOwned<at::Tensor>, 4> inputs_;
  c10::DimVector shape_;
  at::ScalarType dtype_ = at::ScalarType::Undefined;
  at::MemoryFormat memory_format_ = at::MemoryFormat::Contiguous;
  c10::DimVector strides_;
  c10::Device device_ = dipu::DIPU_DEVICE_TYPE;
};

// This class is intended as a base class only and should not be instantiated
// directly.
class OpInferrer : public OpInferrerMeta {
 public:
  ~OpInferrer() override = default;

 protected:
  OpInferrer() = default;

  void compute_shape();
  void compute_dtype();
  void compute_memory_format();

 private:
  // Common logic for calculation, not inherited by children.
  bool fast_compute_memory_format();
  void compute_perm();
  std::vector<c10::DimVector> compute_effective_strides();

  bool all_same_shape_ = true;
  c10::DimVector perm_;
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

class ReduceOpInferrer final : public OpInferrerMeta {
 public:
  at::Tensor infer_out(const at::Tensor& self, c10::OptionalIntArrayRef dim,
                       bool keep_dim, c10::optional<at::ScalarType> dtype);

 private:
  void compute_shape(c10::OptionalIntArrayRef dim, bool keep_dim);
  void compute_dtype();
};

class CatOpInferrer final : public OpInferrerMeta {
 public:
  at::Tensor infer_out(const at::ITensorListRef& tensors, int64_t dim);

 private:
  void compute_memory_format();
  void compute_shape(int64_t dim);

  void check_cat_shape_except_dim(size_t index, size_t index_2,
                                  int64_t dimension);
};

}  // namespace dipu
