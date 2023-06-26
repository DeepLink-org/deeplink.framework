// Copyright (c) 2023, DeepLink.
#include "allocator.h"
#include "DIPUCachingAllocator.h"

namespace dipu {

static DIPUAllocator allocator;

// using in at::empty
// static DIPUCachingAllocator cache_allocator;


REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, &allocator);

}  // namespace dipu