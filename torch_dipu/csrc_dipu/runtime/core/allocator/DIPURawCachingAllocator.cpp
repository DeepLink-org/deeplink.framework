// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

namespace dipu {

class RawCachingAllocator: public CacheAllocator {
public:
  RawCachingAllocator() {

  }

  ~RawCachingAllocator() {

  }

  c10::DataPtr allocate(size_t size) const override {
    DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: malloc " << size << " nbytes");
    return raw_allocator()->allocate(size);
  }

  void empty_cache() const override {

  }

  void release_all_memory() const override {

  }

};

DIPU_REGISTER_ALLOCATOR(RAW, dipu::DIPU_DEVICE_TYPE, RawCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(RAW, at::DeviceType::CPU, RawCachingAllocator, 0);

}  // namespace dipu
