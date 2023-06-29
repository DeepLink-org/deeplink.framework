// Copyright (c) 2023, DeepLink.
#include "allocator.h"

namespace dipu {

static DIPUAllocator allocator;

c10::Allocator* getDIPUAllocator(void) {
  return &allocator;
}

REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, &allocator);

}  // namespace dipu