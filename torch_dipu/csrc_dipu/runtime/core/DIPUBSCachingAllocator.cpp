// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include "allocator.h"
#include <stdint.h>
#include <map>
#include <unorder_map>
#include <list>
#include <deque>
#include <mutex>

namespace dipu {

static void deleteBSContext(void*);

class BSCachingAllocator: public CacheAllocator {


public:
  BSCachingAllocator(c10::Allocator* raw_allocator): CacheAllocator(raw_allocator), device_(c10::DeviceType::CPU) {

  }

  ~BSCachingAllocator() {
    empty_cache();
  }

  size_t getAllocateSize(size_t nbytes) const{
    static constexpr size_t kMinAllocationSize = 16;
    size_t allocateSize = ((nbytes + kMinAllocationSize - 1) / kMinAllocationSize) * kMinAllocationSize;
    return allocateSize;
  }

  mutable std::map<int, std::map<bool, std::deque<void*>>> allCached; // size, using, void*
  //mutable std::unorder_map<size_t, std::deque<void*>> idel_blocks_;

  mutable c10::Device device_;
  using mutex_t = std::recursive_mutex;
  mutable mutex_t mutex_;

  c10::DataPtr allocate(size_t size) const override{
    const size_t nbytes = getAllocateSize(size);
    //std::lock_guard<mutex_t> lk(mutex_);
    auto& blocks = allCached[nbytes];
    auto& using_blocks = blocks[true];
    auto& idel_blocks = blocks[false];
    void* ptr = nullptr;
    if (!idel_blocks.empty()) {
      ptr = idel_blocks.front();
      idel_blocks.pop_front();
      using_blocks.push_back(ptr);
      DIPU_DEBUG_ALLOCATOR("BSCachingAllocator: reuse " << size << " bytes, ptr:" << ptr << ",block size:" << nbytes << ",allocator:" << this);
    } else {
      //ptr = raw_allocator()->raw_allocate(nbytes);
      auto data_ptr = raw_allocator()->allocate(nbytes);
      ptr = data_ptr.get();
      device_ = data_ptr.device();
      data_ptr.release_context();
      using_blocks.push_back(ptr);
      DIPU_DEBUG_ALLOCATOR("BSCachingAllocator: allocate " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    }

    c10::DataPtr data_ptr(ptr, makeContext(ptr, size), deleteBSContext, device_);
    return data_ptr;
  }

  void restore(size_t size, void* ptr) const {
    const size_t nbytes = getAllocateSize(size);
    DIPU_DEBUG_ALLOCATOR("BSCachingAllocator: restore " << nbytes << ", used:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    //std::lock_guard<mutex_t> lk(mutex_);
    auto& blocks = allCached[nbytes];
    auto& using_blocks = blocks[true];
    auto& idel_blocks = blocks[false];
    idel_blocks.push_back(ptr);
    std::remove(using_blocks.begin(), using_blocks.end(), ptr);
  }

  void empty_cache() {
    for (auto size_iter = allCached.begin(); size_iter != allCached.end(); size_iter++) {
      const auto nbytes = size_iter->first;
      auto& blocks = allCached[nbytes];
      auto& using_blocks = blocks[true];
      auto& idel_blocks = blocks[false];
      while (!idel_blocks.empty()) {
        auto ptr = idel_blocks.front();
        raw_allocator()->raw_deallocate(ptr);
        idel_blocks.pop_front();
      }
    }
  }

  struct Context {
    void* ptr_;
    size_t size_;
    const BSCachingAllocator* allocator_;
    Context(void* ptr, size_t size, const BSCachingAllocator* allocator):ptr_(ptr), size_(size), allocator_(allocator) {

    }
    ~Context() {
      allocator_->restore(size_, ptr_);
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


DIPU_REGISTER_ALLOCATOR("BS", dipu::DIPU_DEVICE_TYPE, DIPUAllocator, BSCachingAllocator, 0);

} // namespace dipu