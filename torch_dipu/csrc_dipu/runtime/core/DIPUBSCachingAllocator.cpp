// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include "DIPUDeviceAllocator.h"
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
    c10::Device device_ = c10::DeviceType::CPU;
  };
  mutable std::unique_ptr<Impl> impl;
  using mutex_t = std::mutex;
  mutable mutex_t mutex_;
public:
  BSCachingAllocator() {
    impl.reset(new Impl());
  }

  ~BSCachingAllocator() {
    // The allocator cannot be destructed before all tensors are destructed
    DIPU_DEBUG_ALLOCATOR(8, "~BSCachingAllocator: allocator:"  << this << "allocated_.size:" << impl->allocated_.size() << ", idel_blocks_.size:"  << impl->idel_blocks_.size());
    while (!impl->allocated_.empty()) {
      flush_mem_pool();
      empty_cache();
    }
    DIPU_DEBUG_ALLOCATOR(8, "~BSCachingAllocator: allocator:"  << this << "allocated_.size:" << impl->allocated_.size() << ", idel_blocks_.size:"  << impl->idel_blocks_.size());
    impl.reset(nullptr);
  }

  size_t getAllocateSize(size_t nbytes) const{
    static constexpr size_t kMinAllocationSize = 512;
    size_t allocateSize = ((nbytes + kMinAllocationSize - 1) / kMinAllocationSize) * kMinAllocationSize;
    return allocateSize;
  }

  c10::DataPtr allocate(size_t size) const override{
    flush_mem_pool();
    std::lock_guard<mutex_t> lk(mutex_);
    const size_t nbytes = getAllocateSize(size);
    void* ptr = nullptr;
    auto& idel_blocks = impl->idel_blocks_[nbytes];
    if (idel_blocks.size() > 0) {
      ptr = idel_blocks.front();
      idel_blocks.pop_front();
      impl->total_idel_bytes_ -= nbytes;
    }
    if (ptr == nullptr){
      auto data_ptr = raw_allocator()->allocate(nbytes);
      ptr = data_ptr.get();
      impl->device_ = data_ptr.device();
      data_ptr.release_context();
      impl->allocated_.insert(ptr);
      impl->total_alocated_bytes_+= nbytes;
      DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator::allocate " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    }

    c10::DataPtr data_ptr(ptr, makeContext(ptr, size), deleteBSContext, impl->device_);
    return data_ptr;
  }

  void restore(size_t size, void* ptr) const{
    std::lock_guard<mutex_t> lk(mutex_);
    const size_t nbytes = getAllocateSize(size);
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::restore " << nbytes << ", used:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    impl->idel_blocks_[nbytes].push_back(ptr);
    impl->total_idel_bytes_ += nbytes;
  }

  void empty_cache() override {
    flush_mem_pool();
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::empty_cache ,allocator:"  << this);
    std::lock_guard<mutex_t> lk(mutex_);
    for(auto iter = impl->idel_blocks_.begin(); iter != impl->idel_blocks_.end(); ++iter) {
      auto& idel_blocks = iter->second;
      const size_t size = iter->first;
      while (!idel_blocks.empty()) {
        void* ptr = idel_blocks.front();
        idel_blocks.pop_front();
        impl->total_alocated_bytes_ -= size;
        impl->allocated_.erase(ptr);
        raw_allocator()->raw_deallocate(ptr);
      }
    }
    impl->idel_blocks_.clear();
  }

  void flush_mem_pool() const {
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator::flush_mem_pool allocator:"  << this);
    while (async_mem_pool()->ready()) {
        auto mem = async_mem_pool()->get();
        restore(std::get<1>(mem), std::get<0>(mem));
    }
  }

  struct Context {
    void* ptr_;
    size_t size_;
    const BSCachingAllocator* allocator_;
    Context(void* ptr, size_t size, const BSCachingAllocator* allocator):ptr_(ptr), size_(size), allocator_(allocator) {

    }

    ~Context() {
      allocator_->async_mem_pool()->add(std::make_tuple(ptr_, size_));
      allocator_->flush_mem_pool();
    }
  };


  void* makeContext(void* ptr, size_t size) const{
    auto* ctx = new Context(ptr, size, this);
    return ctx;
  }

};

static void deleteBSContext(void* ptr) {
  auto ctx = static_cast<BSCachingAllocator::Context*>(ptr);
  delete ctx;
}


DIPU_REGISTER_ALLOCATOR("BS", dipu::DIPU_DEVICE_TYPE, BSCachingAllocator, 0);

} // namespace dipu