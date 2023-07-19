// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

namespace dipu {

static void deleteRawCachingAllocatorContext(void*);

class RawCachingAllocator: public CacheAllocator {
public:
  RawCachingAllocator() {

  }

  ~RawCachingAllocator() {

  }

  c10::DataPtr allocate(size_t size) const override {
    DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: malloc " << size << " nbytes");
    auto ptr = raw_allocator()->raw_allocate(size);
    return c10::DataPtr(ptr, new DataPtrContextBase(this, ptr, size), deleteRawCachingAllocatorContext, device());
  }

  void empty_cache() const override {

  }

  void release_all_memory() const override {

  }
};

static void deleteRawCachingAllocatorContext(void* ptr) {
  auto ctx = static_cast<CacheAllocator::DataPtrContextBase*>(ptr);
  delete ctx;
}

DIPU_REGISTER_ALLOCATOR(RAW, dipu::DIPU_DEVICE_TYPE, RawCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(RAW, at::DeviceType::CPU, RawCachingAllocator, 0);

}  // namespace dipu
