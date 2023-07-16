// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>

namespace dipu {

class DIPUCopyInplace {
public:
  DIPUCopyInplace() = default;
  ~DIPUCopyInplace() = default;

  // TODO(caikun): optimize function names and parameters!!!!
  virtual at::Tensor& run(at::Tensor& self, const at::Tensor& src, bool non_blocking);
  virtual void copy_between_devices(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking);
  virtual void copy_same_dtype(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking);
  virtual void copy_between_host_device(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking);

protected:
  // proxy device tensor to cpu to handle different dtype/view problem
  at::Tensor& slow_copy(at::Tensor& self, const at::Tensor& src, bool non_blocking);
};

DIPUCopyInplace* getDipuCopyInplace();
void setDipuCopyInplace(DIPUCopyInplace *op);

}  // namespace dipu