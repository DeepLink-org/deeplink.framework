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
      std::deque<DIPUEvent> events;
      for (const auto& item : streams()) {
        events.emplace_back();
        events.back().record(item);
      }
      auto allocator_ = static_cast<const RawCachingAllocator*>(allocator());
      allocator_->async_mem_pool()->add(std::make_tuple(ptr(), size()), events);
      allocator_->set_memory_allocated(allocator_->memory_allocated() -
                                       real_size_);
      allocator_->empty_cache();
    }
    size_t real_size_ = 0;
  };

  static size_t getAllocateSize(size_t nbytes) {
    static const size_t kMinAllocationSize = []() {
      const int kBytesNum = 512;
      size_t size = kBytesNum;
      const char* env = std::getenv("DIPU_RAW_ALLOCATOR_MIN_ALLOCATE_SIZE");
      if (env != nullptr) {
        size = std::atoi(env);
      }
      return size;
    }();
    size_t allocateSize = ((nbytes - 1) | (kMinAllocationSize - 1)) + 1;
    return allocateSize;
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
    while (async_mem_pool()->size() > 0) {
      if (async_mem_pool()->ready()) {
        auto mem = async_mem_pool()->get();
        void* ptr = std::get<0>(mem);
        size_t size = std::get<1>(mem);
        size_t nbytes = getAllocateSize(size);
        raw_allocator()->raw_deallocate(ptr);
        set_memory_reserved(memory_reserved() - nbytes);
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DIPU_REGISTER_ALLOCATOR(RAW, DIPU_DEVICE_TYPE_MACRO, RawCachingAllocator, 0);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DIPU_REGISTER_ALLOCATOR(RAW, CPU, RawCachingAllocator, 0);

}  // namespace dipu
