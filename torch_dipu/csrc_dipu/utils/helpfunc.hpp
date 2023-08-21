// Copyright (c) 2023, DeepLink.
#pragma once
#include <csrc_dipu/base/basedef.h>

using dipu::devapis::VendorDeviceType;

namespace dipu {

constexpr const char* VendorTypeToStr(VendorDeviceType t) noexcept {
  switch (t) {
    case VendorDeviceType::MLU: return "MLU";
    case VendorDeviceType::CUDA: return "CUDA";
    case VendorDeviceType::NPU: return "NPU";
    case VendorDeviceType::GCU: return "GCU";
    case VendorDeviceType::SUPA: return "SUPA";
    case VendorDeviceType::STPU: return "STPU";
  }
  return "null";
}

DIPU_API bool isDeviceTensor(const at::Tensor &tensor);

DIPU_API bool is_in_bad_fork();
void poison_fork();

} // end dipu