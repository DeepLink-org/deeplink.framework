// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

namespace dipu {

static void deleteRawCachingAllocatorContext(void* ptr);

class RawCachingAllocator : public CacheAllocator {
 public:
  RawCachingAllocator() = default;

  ~RawCachingAllocator() override = default;

  class Context : public DataPtrContextBase {
   public:
    Context(const CacheAllocator* allocator, void* ptr, size_t size,
            size_t real_size)
        : DataPtrContextBase(allocator, ptr, size), real_size_(real_size) {}

    ~Context() {
      auto alloc = static_cast<const RawCachingAllocator*>(allocator());
      alloc->async_mem_pool()->put(std::make_tuple(ptr(), size()),
                                   streams_to_events());
      alloc->set_memory_allocated(alloc->memory_allocated() - real_size_);
      alloc->empty_cache();
    }
    size_t real_size_ = 0;
  };

  static size_t getAllocateSize(size_t nbytes) {
    return getMemoryAlignmentStrategy()->roundBytes(nbytes);
  }

  c10::DataPtr allocate(size_t size) const override {
    size_t nbytes = getAllocateSize(size);
    empty_cache();
    DIPU_DEBUG_ALLOCATOR(4, "RawCachingAllocator: malloc "
                                << nbytes << " nbytes"
                                << ", requires:" << size << " bytes");
    auto ptr = raw_allocator()->raw_allocate(nbytes);
    set_memory_reserved(memory_reserved() + nbytes);
    set_memory_allocated(memory_allocated() + nbytes);
    return {ptr, new Context(this, ptr, size, nbytes),
            deleteRawCachingAllocatorContext, device()};
  }

  void empty_cache() const override {
    DIPU_DEBUG_ALLOCATOR(8, "RawCachingAllocator: empty_cache");

    auto& pool = *async_mem_pool();
    for (auto item = pool.pop(); item; item = pool.pop()) {
      auto [ptr, size] = item.value();
      auto nbytes = getAllocateSize(size);
      raw_allocator()->raw_deallocate(ptr);
      set_memory_reserved(memory_reserved() - nbytes);
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
// TODO(allocator) - Refactor it!
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-bind)
DIPU_REGISTER_ALLOCATOR(RAW, DIPU_DEVICE_TYPE_MACRO, RawCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(RAW, CPU, RawCachingAllocator, 0);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-bind)

}  // namespace dipu
