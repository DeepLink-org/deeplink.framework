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

  class Context: public DataPtrContextBase {
    public:
      Context(const CacheAllocator* allocator, void* ptr, size_t size):DataPtrContextBase(allocator, ptr, size){}
      ~Context() {
        std::deque<DIPUEvent> events;
        for (auto iter = streams().begin(); iter != streams().end(); iter++) {
          events.emplace_back();
          events.back().record(*iter);
        }
        auto allocator_ = static_cast<const RawCachingAllocator*>(allocator());
        allocator_->async_mem_pool()->add(std::make_tuple(ptr(), size()), events);
        allocator_->empty_cache();
      }
  };

  c10::DataPtr allocate(size_t size) const override {
    empty_cache();
    DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: malloc " << size << " nbytes");
    auto ptr = raw_allocator()->raw_allocate(size);
    return c10::DataPtr(ptr, new DataPtrContextBase(this, ptr, size), deleteRawCachingAllocatorContext, device());
  }

  void empty_cache() const override {
    DIPU_DEBUG_ALLOCATOR(8, "RawCachingAllocator: empty_cache");
    while(async_mem_pool()->size() > 0) {
      if(async_mem_pool()->ready()) {
        auto mem = async_mem_pool()->get();
        void* ptr = std::get<0>(mem);
        raw_allocator()->raw_deallocate(ptr);
      } else {
        std::this_thread::yield();
      }
    }
  }

  void release_all_memory() const override {
    DIPU_DEBUG_ALLOCATOR(8, "RawCachingAllocator: release_all_memory");
    empty_cache();
  }
};

static void deleteRawCachingAllocatorContext(void* ptr) {
  auto ctx = static_cast<RawCachingAllocator::Context*>(ptr);
  delete ctx;
}

DIPU_REGISTER_ALLOCATOR(RAW, dipu::DIPU_DEVICE_TYPE, RawCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(RAW, at::DeviceType::CPU, RawCachingAllocator, 0);

}  // namespace dipu
