// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <stdint.h>
#include <map>
#include <set>
#include <unordered_map>
#include <list>
#include <deque>
#include <mutex>

namespace dipu {

static void deleteBSContext(void*);

class BSCachingAllocator: public CacheAllocator {
  struct Impl {
    std::unordered_map<size_t, std::list<void*>> idel_blocks_;
    std::set<void*> allocated_;
    size_t total_alocated_bytes_ = 0;
    size_t total_idel_bytes_ = 0;
  };
  mutable std::unique_ptr<Impl> impl;
  using mutex_t = std::recursive_mutex;
  mutable mutex_t mutex_;
public:
  BSCachingAllocator() {
    impl.reset(new Impl());
  }

  ~BSCachingAllocator() {
    release_all_memory();
  }

  // Better adaptability to memory blocks of various sizes, but internal fragmentation will be larger
  size_t getAllocateSizeMoreAdaptable(size_t nbytes) const{
    static const int kMinAllocationSizeExp = [](){
      size_t size = 511;
      const char* env = std::getenv("DIPU_BS_ALLOCATOR_MIN_ALLOCATE_SIZE");
      if (env != nullptr) {
        size = std::atoi(env);
      }
      int exp = 32 - __builtin_clz(size);
      return exp;
    }();
    auto r = std::max(32 - __builtin_clz(nbytes), kMinAllocationSizeExp);
    size_t allocateSize = 1 << r;
    return allocateSize;
  }

  // The internal fragments are smaller, but are less adaptable to scenes with frequent and drastic changes in size.
  size_t getAllocateSizeLessFragmentation(size_t nbytes) const{
    static const size_t kMinAllocationSize = [](){
      size_t size = 512;
      const char* env = std::getenv("DIPU_BS_ALLOCATOR_MIN_ALLOCATE_SIZE");
      if (env != nullptr) {
        size = std::atoi(env);
      }
      return size;
    }();
    size_t allocateSize = ((nbytes - 1) | (kMinAllocationSize - 1)) + 1;
    return allocateSize;
  }

  size_t getAllocateSize(size_t nbytes) const{
    static bool less_fragmentation = std::getenv("DIPU_BS_MORE_ADAPTABLE") == nullptr;
    return less_fragmentation ? getAllocateSizeLessFragmentation(nbytes) : getAllocateSizeMoreAdaptable(nbytes);
  }

  c10::DataPtr allocate(size_t size) const override{
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::allocate " << size << ",allocator:" << this <<", memory-usage" << memory_allocated() << "/" << memory_reserved());
    std::lock_guard<mutex_t> lk(mutex_);
    flush_mem_pool();
    size_t nbytes = getAllocateSize(size);
    void* ptr = nullptr;
    auto& idel_blocks = impl->idel_blocks_[nbytes];
    if (idel_blocks.size() <= 0) {
      empty_resource_pool();
    }
    for (size_t i = 0; i < 2; i++) {
      if (idel_blocks.size() > 0) {
        ptr = idel_blocks.front();
        idel_blocks.pop_front();
        impl->total_idel_bytes_ -= nbytes;
        DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator::reuse " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
        break;
      } else {
        try {
          auto data_ptr = raw_allocator()->allocate(nbytes);
          ptr = data_ptr.get();
          device() = data_ptr.device();
          data_ptr.release_context();
          set_memory_reserved(memory_reserved() + nbytes);

          impl->allocated_.insert(ptr);
          impl->total_alocated_bytes_+= nbytes;
          DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator::allocate " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
          break;
        }
        catch(...) {
          if (i == 0) {
            empty_cache();
          } else {
            TORCH_CHECK(false, "no memory available")
          }
        }
      }
    }
    set_memory_allocated(memory_allocated() + nbytes);
    c10::DataPtr data_ptr(ptr, makeContext(ptr, size, nbytes), deleteBSContext, device());
    return data_ptr;
  }

  void restore(size_t size, void* ptr) const{
    size_t nbytes = getAllocateSize(size);
    std::lock_guard<mutex_t> lk(mutex_);
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::restore " << nbytes << " bytes, ptr:" << ptr << ",allocator:" << this);
    impl->idel_blocks_[nbytes].push_back(ptr);
    impl->total_idel_bytes_ += nbytes;
  }

  void empty_resource_pool() const {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::empty_resource_pool ,allocator:"  << this);
    while(async_mem_pool()->size() > 0) {
      if (async_mem_pool()->ready()) {
        flush_mem_pool();
      } else {
        std::this_thread::yield();
      }
    }
  }

  void empty_cache() const override {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::empty_cache ,allocator:"  << this);
    empty_resource_pool();
    std::lock_guard<mutex_t> lk(mutex_);
    for(auto iter = impl->idel_blocks_.begin(); iter != impl->idel_blocks_.end(); ++iter) {
      auto& idel_blocks = iter->second;
      const size_t size = iter->first;
      while (!idel_blocks.empty()) {
        void* ptr = idel_blocks.front();
        idel_blocks.pop_front();
        impl->total_alocated_bytes_ -= size;
        set_memory_reserved(memory_reserved() - size);
        impl->allocated_.erase(ptr);
        raw_allocator()->raw_deallocate(ptr);
      }
    }
  }

  void release_all_memory() const {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::release_all_memory allocator:"  << this);
    empty_cache();
  }

  void flush_mem_pool() const {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::flush_mem_pool allocator:"  << this);
    while (async_mem_pool()->ready()) {
        auto mem = async_mem_pool()->get();
        restore(std::get<1>(mem), std::get<0>(mem));
    }
  }

  struct Context: public DataPtrContextBase {
    Context(void* ptr, size_t size, size_t real_size, const BSCachingAllocator* allocator):DataPtrContextBase(allocator, ptr, size), real_size_(real_size) {

    }

    ~Context() {
      auto allocator_ = static_cast<const BSCachingAllocator*>(allocator());
      DIPU_DEBUG_ALLOCATOR(8, __FUNCTION__ << " allocator:" << allocator_ << ", ptr:" << ptr() << ", size_:" << size());
      if (allocator_->impl) {
        std::deque<DIPUEvent> events;
        for (auto iter = streams().begin(); iter != streams().end(); iter++) {
          events.emplace_back();
          events.back().record(*iter);
        }

        allocator_->async_mem_pool()->add(std::make_tuple(ptr(), size()), events);
        allocator_->set_memory_allocated(allocator_->memory_allocated() - real_size_);
        allocator_->flush_mem_pool();
      }
    }
    size_t real_size_ = 0;
  };


  void* makeContext(void* ptr, size_t size, size_t real_size) const{
    auto ctx = new Context(ptr, size, real_size, this);
    return ctx;
  }

};

static void deleteBSContext(void* ptr) {
  auto ctx = static_cast<BSCachingAllocator::Context*>(ptr);
  delete ctx;
}


DIPU_REGISTER_ALLOCATOR(BS, dipu::DIPU_DEVICE_TYPE, BSCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(BS, at::DeviceType::CPU, BSCachingAllocator, 0);

} // namespace dipu