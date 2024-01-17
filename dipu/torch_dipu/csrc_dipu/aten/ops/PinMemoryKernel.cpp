// Copyright (c) 2023, DeepLink.
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/runtime/rthelper.h"

namespace dipu {
namespace native {
namespace dipu_aten {

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }

  // prefer using dipu::isPinnedPtr instead of dipu::devapis::isPinnedPtr
  // because device may not support it
  return dipu::isPinnedPtr(self.storage().data());
}

at::Tensor _pin_memory(const at::Tensor& self,
                       c10::optional<at::Device> device) {
  auto allocator = dipu::getAllocator(at::DeviceType::CPU);
  auto storage =
      c10::Storage(c10::Storage::use_byte_size_t(),
                   static_cast<int64_t>(at::detail::computeStorageNbytes(
                       self.sizes(), self.strides(), self.dtype().itemsize())),
                   allocator, false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

}  // namespace dipu_aten
}  // namespace native
}  // namespace dipu
