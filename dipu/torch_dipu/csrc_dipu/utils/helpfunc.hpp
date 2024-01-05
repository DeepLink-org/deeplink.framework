// Copyright (c) 2023, DeepLink.
#pragma once
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

enum class CustomFormat_t {
  UNDEFINED = -1,
  NCHW = 0,
  NHWC = 1,
  ND = 2,
  NC1HWC0 = 3,
  FRACTAL_Z = 4,
  NC1HWC0_C04 = 12,
  HWCN = 16,
  NDHWC = 27,
  FRACTAL_NZ = 29,
  NCDHW = 30,
  NDC1HWC0 = 32,
  FRACTAL_Z_3D = 33
};

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

at::Tensor format_cast(at::Tensor tensor, CustomFormat_t format);

CustomFormat_t get_format(const at::Tensor& tensor);

DIPU_API bool is_in_bad_fork();
void poison_fork();

}  // namespace dipu
