// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

namespace dipu {

static void RawCachingAllocatorDeleter(void *ptr) {
  DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: free:" << ptr);
  dipu::getCurrentDIPUStream().synchronize();
  DIPUDeviceAllocatorDeleter(ptr);
}

class RawCachingAllocator: public CacheAllocator {
public:
  RawCachingAllocator() {

  }

  ~RawCachingAllocator() {

  }

  c10::DataPtr allocate(size_t size) const override{
    DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: malloc " << size << " nbytes");
    auto data_ptr = raw_allocator()->allocate(size);
    auto ptr = data_ptr.get();
    data_ptr.release_context();
    dipu::getCurrentDIPUStream().synchronize();
    return c10::DataPtr(ptr, ptr, &RawCachingAllocatorDeleter, data_ptr.device());
  }

};

DIPU_REGISTER_ALLOCATOR("RAW", dipu::DIPU_DEVICE_TYPE, RawCachingAllocator, 0);

}  // namespace dipu
