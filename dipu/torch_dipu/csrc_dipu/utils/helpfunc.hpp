// Copyright (c) 2023, DeepLink.
#pragma once
#include <bits/stdint-intn.h>
#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/vendor/vendorapi.h>

namespace dipu {

using dipu::devapis::VendorDeviceType;
constexpr const char* VendorTypeToStr(VendorDeviceType t) noexcept {
  switch (t) {
    case VendorDeviceType::MLU:
      return "MLU";
    case VendorDeviceType::CUDA:
      return "CUDA";
    case VendorDeviceType::NPU:
      return "NPU";
    case VendorDeviceType::GCU:
      return "GCU";
    case VendorDeviceType::SUPA:
      return "SUPA";
    case VendorDeviceType::DROPLET:
      return "DROPLET";
    case VendorDeviceType::KLX:
      return "KLX";
  }
  return "null";
}

enum class NativeMemoryFormat_t : int64_t;

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

at::Tensor native_memory_format_cast(at::Tensor tensor, NativeMemoryFormat_t format);

NativeMemoryFormat_t get_native_memory_format(const at::Tensor& tensor);

DIPU_API bool is_in_bad_fork();
void poison_fork();

}  // namespace dipu
