// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/runtime/device/basedef.h"

#include "vender_helper.hpp"  // provide VendorDeviceTypeToStr for backward compatibility

namespace dipu {

// TODO(fandaoyi): remove this function after complete refactoring
// TODO(lilingjie,fandaoyi): use constexpr after 910b CI is ready
// throw in constexpr funtions (c++14) is not supported in gcc-7.5, which is a
// known bug already fixed later (at least in gcc-10.x).
[[deprecated("Use VendorDeviceTypeToStr instead")]]
inline const char* VendorTypeToStr(devapis::VendorDeviceType t) {
  return VendorDeviceTypeToStr(t);
}

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

enum class NativeMemoryFormat_t : int64_t;

at::Tensor native_memory_format_cast(at::Tensor tensor,
                                     NativeMemoryFormat_t format);

NativeMemoryFormat_t get_native_memory_format(const at::Tensor& tensor);

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
