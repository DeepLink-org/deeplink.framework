// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <csrc_dipu/base/basedef.h>

namespace dipu {

DIPU_API c10::Allocator* getHostAllocator();
DIPU_API bool isPinnedPtr(const void* ptr);

}  // namespace dipu