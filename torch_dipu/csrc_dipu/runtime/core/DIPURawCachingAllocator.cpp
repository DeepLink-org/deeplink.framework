// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include "allocator.h"

namespace dipu {

class RawCachingAllocator: public CacheAllocator {
public:
  RawCachingAllocator(c10::Allocator* raw_allocator): CacheAllocator(raw_allocator) {

  }

  ~RawCachingAllocator() {

  }

  inline virtual c10::DataPtr allocate(size_t size) const {
    auto rawData = raw_allocator_->allocate(size);
    DIPU_DEBUG_ALLOCATOR("RawCachingAllocator: malloc " << size << " nbytes, ptr:" << rawData.get());
    return rawData;
  }

};

DIPU_REGISTER_ALLOCATOR("RAW", dipu::DIPU_DEVICE_TYPE, DIPUAllocator, RawCachingAllocator, 0);

}  // namespace dipu