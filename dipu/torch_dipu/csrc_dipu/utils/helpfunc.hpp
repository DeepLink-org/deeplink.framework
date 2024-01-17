// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu {

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
