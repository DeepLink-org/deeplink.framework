// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIterator.h>

namespace dipu {

class DIPUCopyInplace {
 public:
  DIPUCopyInplace() = default;
  virtual ~DIPUCopyInplace() = default;

  virtual at::Tensor &run(at::Tensor &self, const at::Tensor &src,
                          bool non_blocking);

  // copy between devices
  // 1. dtype & shape & stride all equal, use memCopyD2DAsync
  // 2. use DIPUATenFunctions::copy_, proxy device tensor to cpu to handle
  // different dtype/view problem
  virtual at::Tensor &copy_between_devices(at::TensorIterator &iter,
                                           at::Tensor &self,
                                           const at::Tensor &src,
                                           bool non_blocking);

  // copy between cpu and device, dtype & shape & stride all equal
  // 1. host to device, use memCopyH2DAsync
  // 2. device to host, use memCopyD2HAsync
  virtual at::Tensor &copy_contiguous(at::TensorIterator &iter,
                                      at::Tensor &self, const at::Tensor &src,
                                      bool non_blocking);

  // copy between cpu and device, different dtype or view
  virtual at::Tensor &copy_uncontiguous(at::TensorIterator &iter,
                                        at::Tensor &self, const at::Tensor &src,
                                        bool non_blocking);
};

DIPUCopyInplace *getDipuCopyInplace();
void setDipuCopyInplace(DIPUCopyInplace *op);

}  // namespace dipu