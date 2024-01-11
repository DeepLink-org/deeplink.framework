// Copyright (c) 2023, DeepLink.
#pragma once
#include <c10/util/Exception.h>

#include <csrc_dipu/base/basedef.h>

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

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

DIPU_API bool is_in_bad_fork();
void poison_fork();

class IgnoreOpRegWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    // do nothing
  }
};

inline c10::WarningHandler* getIgnoreHandler() {
  static IgnoreOpRegWarningHandler handler_ = IgnoreOpRegWarningHandler();
  return &handler_;
}

}  // namespace dipu
