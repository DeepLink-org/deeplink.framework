// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <csrc_dipu/common.h>

namespace dipu {

class DIPUHostAllocator : public c10::Allocator {
public:
  c10::DataPtr allocate(size_t size) const;
};

DIPU_API bool isPinnedPtr(const void* ptr);

}  // namespace dipu