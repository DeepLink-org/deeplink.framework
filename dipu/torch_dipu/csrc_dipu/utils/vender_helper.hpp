// Copyright (c) 2023, DeepLink.
#pragma once

#include <stdexcept>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu {

constexpr const char* VendorDeviceTypeToStr(devapis::VendorDeviceType t) {
  switch (t) {
    case devapis::VendorDeviceType::MLU:
      return "MLU";
    case devapis::VendorDeviceType::CUDA:
      return "CUDA";
    case devapis::VendorDeviceType::NPU:
      return "NPU";
    case devapis::VendorDeviceType::GCU:
      return "GCU";
    case devapis::VendorDeviceType::SUPA:
      return "SUPA";
    case devapis::VendorDeviceType::DROPLET:
      return "DROPLET";
    case devapis::VendorDeviceType::KLX:
      return "KLX";
    default:
      throw std::invalid_argument("Unknown device type");
  }
}

// constexpr version of C-style string comparison
constexpr bool c_string_equal(const char* a, const char* b) noexcept {
  return *a == *b && (*a == '\0' || c_string_equal(a + 1, b + 1));
}

constexpr devapis::VendorDeviceType VendorNameToDeviceType(const char* str) {
#define DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(name, type) \
  if (c_string_equal(str, #name)) {                          \
    return devapis::VendorDeviceType::type;                  \
  }
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(camb, MLU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(cuda, CUDA);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(ascend, NPU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(tops, GCU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(supa, SUPA);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(droplet, DROPLET);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(kunlunxin, KLX);
#undef DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE
  throw std::invalid_argument("Unknown device name");
}

}  // namespace dipu
