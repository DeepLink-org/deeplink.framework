// Copyright (c) 2023, DeepLink.
#include "allocator.h"

namespace dipu {

static DIPUAllocator allocator;

REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, &allocator);

}  // namespace dipu