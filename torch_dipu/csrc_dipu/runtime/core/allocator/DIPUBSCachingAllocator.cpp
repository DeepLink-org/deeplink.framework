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

  size_t getAllocateSize(size_t nbytes) const{
    static constexpr size_t kMinAllocationSize = 512;
    size_t allocateSize = ((nbytes - 1) | (kMinAllocationSize - 1)) + 1;
    return allocateSize;
  }

  c10::DataPtr allocate(size_t size) const override{
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::allocate " << size << ",allocator:" << this <<", memory-usage" << memory_allocated() << "/" << memory_reserved());
    flush_mem_pool();
    std::lock_guard<mutex_t> lk(mutex_);
    size_t nbytes = getAllocateSize(size);
    void* ptr = nullptr;
    auto& idel_blocks = impl->idel_blocks_[nbytes];
    if (idel_blocks.size() > 0) {
      ptr = idel_blocks.front();
      idel_blocks.pop_front();
      impl->total_idel_bytes_ -= nbytes;
      DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator::reuse " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    }
    if (ptr == nullptr){
      for (size_t i = 0; i < 2; i++) {
        try {
          auto data_ptr = raw_allocator()->allocate(nbytes);
          ptr = data_ptr.get();
          device() = data_ptr.device();
          data_ptr.release_context();
          set_memory_reserved(memory_reserved() + nbytes);
          break;
        }
        catch(...) {
          if (i == 0) {
            empty_cache();
          } else {
            throw std::runtime_error("no device memory available");
          }
        }
      }
      impl->allocated_.insert(ptr);
      impl->total_alocated_bytes_+= nbytes;
      DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator::allocate " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    }
    set_memory_allocated(memory_allocated() + nbytes);
    c10::DataPtr data_ptr(ptr, makeContext(ptr, size), deleteBSContext, device());
    return data_ptr;
  }

  void restore(size_t size, void* ptr) const{
    size_t nbytes = getAllocateSize(size);
    std::lock_guard<mutex_t> lk(mutex_);
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::restore " << nbytes << " bytes, ptr:" << ptr << ",allocator:" << this);
    impl->idel_blocks_[nbytes].push_back(ptr);
    impl->total_idel_bytes_ += nbytes;
    set_memory_allocated(memory_allocated() - nbytes);
  }

  void empty_cache() const override {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::empty_cache ,allocator:"  << this);
    while(async_mem_pool()->size() > 0) {
      if (async_mem_pool()->ready()) {
        flush_mem_pool();
      } else {
        std::this_thread::yield();
      }
    }
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
    impl->idel_blocks_.clear();
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
    Context(void* ptr, size_t size, const BSCachingAllocator* allocator):DataPtrContextBase(allocator, ptr, size) {

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
        allocator_->flush_mem_pool();
      }
    }
  };


  void* makeContext(void* ptr, size_t size) const{
    auto ctx = new Context(ptr, size, this);
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