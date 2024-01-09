// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu {

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

enum class NativeMemoryFormat_t : int64_t;

at::Tensor native_memory_format_cast(at::Tensor tensor,
                                     NativeMemoryFormat_t format);

NativeMemoryFormat_t get_native_memory_format(const at::Tensor& tensor);

DIPU_API bool is_in_bad_fork();
void poison_fork();

}  // namespace dipu
