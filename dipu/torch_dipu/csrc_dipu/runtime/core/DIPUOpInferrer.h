// Copyright (c) 2024, DeepLink.
#pragma once

#include <ATen/ATen.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

using DimVector = c10::SmallVector<int64_t, 4>;

// Base Class for inferring the shape and dtype of an output Tensor
// based on its inputs, then malloc the output tensor.
class OpInferrer {
 public:
  OpInferrer() = default;
  virtual ~OpInferrer() = default;

  at::ScalarType common_dtype() const { return dtype_; }

  DimVector target_shape() const { return shape_; }

  void add_inputs(const std::vector<at::Tensor>& inputs) {
    TORCH_CHECK(!inputs.empty(), "Input tensors must not be empty");
    for (const auto& tensor : inputs) {
      TORCH_CHECK(tensor.defined(), "Input tensor is undefined");
    }
    inputs_ = inputs;
  }

  // core logic to infer
  virtual void infer() = 0;

 protected:
  // Computes the shape of the output, supporting broadcasting rules.
  void compute_shape();

  // Computes the dtype of the output based on all input tensors,
  // supporting dtype promotion
  void compute_dtype();

  // Allocates the output Tensor based on the inferred shape and dtype.
  at::Tensor malloc_output() {
    at::TensorOptions options =
        at::TensorOptions().dtype(dtype_).device(device_);
    return native::nodispatch::empty(shape_, options);
  }

  // Member Variables
  c10::SmallVector<at::Tensor, 4> inputs_;
  DimVector shape_;
  at::ScalarType dtype_ = at::ScalarType::Undefined;
  at::Device device_ = dipu::DIPU_DEVICE_TYPE;
};

class BinaryOpInferrer final: public OpInferrer {
 public:
  void meta(const at::Tensor& self, const at::Tensor& other) {
    add_inputs({self, other});
    infer();
  }
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other) {
    meta(self, other);
    return malloc_output();
  }

 private:
  void infer() override;
};

class BinaryFloatOpInferrer final: public OpInferrer {
 public:
  void meta(const at::Tensor& self, const at::Tensor& other) {
    add_inputs({self, other});
    infer();
  }
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other) {
    meta(self, other);
    return malloc_output();
  }

 private:
  void infer() override;
};

class UnaryOpInferrer final: public OpInferrer  {
 public:
  void meta(const at::Tensor& self) {
    add_inputs({self});
    infer();
  }
  at::Tensor infer_out(const at::Tensor& self) {
    meta(self);
    return malloc_output();
  }

 private:
  void infer() override;
};

class ComparisonOpInferrer final: public OpInferrer {
 public:
  void meta(const at::Tensor& self, const at::Tensor& other) {
    add_inputs({self, other});
    infer();
  }
  at::Tensor infer_out(const at::Tensor& self, const at::Tensor& other) {
    meta(self, other);
    return malloc_output();
  }

 private:
  void infer() override;
};

}  // namespace dipu
