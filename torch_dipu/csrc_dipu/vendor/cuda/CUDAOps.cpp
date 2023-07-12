// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/runtime/core/DIPUOps.h>

#include <iostream>
#include <cuda_runtime_api.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/common.h>
#include <c10/util/Exception.h>

namespace dipu {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (self.numel() == 0) {
    return self;
  }

  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  ::diopiContext context(stream.rawstream());
  auto ctx = &context;

  ::diopiConstTensorHandle_t srcDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(src);
  ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
  ::diopiError_t ret = ::diopiCopyInp(ctx, srcDiopiTensorHandle, selfDiopiTensorHandle);
  TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, R"(::diopiCopyInp(ctx, src, dst);)", " error, error code is ", ret, "error message is ", diopiGetLastErrorString());

  // TODO(caikun): sync stream when non_blocking=false if cache allocator is ready
  dipu::devapis::syncStream(stream.rawstream());
  return self;
}

}  // namespace dipu