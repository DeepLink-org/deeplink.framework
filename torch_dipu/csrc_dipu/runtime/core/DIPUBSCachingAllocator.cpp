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

  mutable std::unordered_map<size_t, std::deque<void*>> idel_blocks_;
  mutable std::set<void*> allocated_;
  mutable size_t idel_blocks_num_ = 0;
  mutable size_t total_blocks_num_ = 0;
  mutable size_t total_alocated_bytes_ = 0;
  mutable size_t total_idel_bytes_ = 0;
  mutable c10::Device device_;
  using mutex_t = std::recursive_mutex;
  mutable mutex_t mutex_;

public:
  BSCachingAllocator(): device_(c10::DeviceType::CPU)  {

  }

  ~BSCachingAllocator() {
    // The allocator cannot be destructed before all tensors are destructed
    while (!allocated_.empty()) {
      empty_cache();
    }
  }

  size_t getAllocateSize(size_t nbytes) const{
    static constexpr size_t kMinAllocationSize = 512;
    size_t allocateSize = ((nbytes + kMinAllocationSize - 1) / kMinAllocationSize) * kMinAllocationSize;
    return allocateSize;
  }

  c10::DataPtr allocate(size_t size) const override{
    const size_t nbytes = getAllocateSize(size);
    flush_mem_pool();
    std::lock_guard<mutex_t> lk(mutex_);
    void* ptr = nullptr;
    auto& idel_blocks = idel_blocks_[nbytes];
    const int max_find_count = idel_blocks.size();
    if (idel_blocks.size() > 0) {
      ptr = idel_blocks.front();
      idel_blocks.pop_front();
      total_idel_bytes_ -= nbytes;
    }
    if (ptr == nullptr){
      auto data_ptr = raw_allocator()->allocate(nbytes);
      ptr = data_ptr.get();
      device_ = data_ptr.device();
      data_ptr.release_context();
      allocated_.insert(ptr);
      total_blocks_num_++;
      total_alocated_bytes_+= nbytes;
      DIPU_DEBUG_ALLOCATOR(4, "BSCachingAllocator: allocate " << nbytes << ", requires:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    }

    c10::DataPtr data_ptr(ptr, makeContext(ptr, size), deleteBSContext, device_);
    return data_ptr;
  }

  void restore(size_t size, void* ptr) const {
    const size_t nbytes = getAllocateSize(size);
    DIPU_DEBUG_ALLOCATOR(8, "BSCachingAllocator: restore " << nbytes << ", used:" << size << " bytes, ptr:" << ptr << ",allocator:" << this);
    std::lock_guard<mutex_t> lk(mutex_);
    idel_blocks_[nbytes].push_back(ptr);
    idel_blocks_num_++;
    total_idel_bytes_ += nbytes;
    if ((total_idel_bytes_ > (512 << 20)) && ((1.0 * total_idel_bytes_ / total_alocated_bytes_) > 0.7)) {
      empty_cache();
    }

  }

  void empty_cache() const {
    std::lock_guard<mutex_t> lk(mutex_);
    for(auto iter = idel_blocks_.begin(); iter != idel_blocks_.end(); ++iter) {
      auto& idel_blocks = iter->second;
      while (!idel_blocks.empty()) {
        auto ptr = idel_blocks.front();
        raw_allocator()->raw_deallocate(ptr);
        total_alocated_bytes_ -= iter->first;
        total_blocks_num_--;
        idel_blocks.pop_front();
        allocated_.erase(ptr);
      }
    }
  }

  void flush_mem_pool() const {
    std::lock_guard<mutex_t> lk(mutex_);
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