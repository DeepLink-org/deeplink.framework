// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "DIPUStream.h"

namespace dipu {

class DIPU_API CacheAllocator: public c10::Allocator {
  c10::Allocator* raw_allocator_ = nullptr;
  public:
    virtual c10::DataPtr allocate(size_t n) const = 0;
    explicit CacheAllocator(c10::Allocator* raw_allocator):raw_allocator_(raw_allocator) {
      TORCH_CHECK(raw_allocator_);
    }

    virtual ~CacheAllocator() = 0;
};

}  // namespace dipu

