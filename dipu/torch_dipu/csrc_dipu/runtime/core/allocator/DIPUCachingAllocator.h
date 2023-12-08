// Copyright (c) 2023, DeepLink.
#pragma once

#include <list>
#include <map>
#include <set>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "csrc_dipu/runtime/core/DIPUEvent.h"

#include "DIPUAsyncResourcePool.h"
#include "DIPUCachingAllocatorUtils.h"
#include "DIPURawAllocator.h"

namespace dipu {

using AsyncMemPool = AsyncResourcePool<std::tuple<void*, size_t>>;

class MemStats {
 private:
  mutable size_t reserved_in_bytes_ = 0;
  mutable size_t allocated_in_bytes_ = 0;
  mutable size_t max_reserved_in_bytes_ = 0;
  mutable size_t max_allocated_in_bytes_ = 0;

 protected:
  void set_memory_reserved(size_t reserved_in_bytes) const {
    reserved_in_bytes_ = reserved_in_bytes;
    max_reserved_in_bytes_ = max_reserved_in_bytes_ > reserved_in_bytes
                                 ? max_reserved_in_bytes_
                                 : reserved_in_bytes;
  }

  void set_memory_allocated(size_t allocated_in_bytes) const {
    allocated_in_bytes_ = allocated_in_bytes;
    max_allocated_in_bytes_ = max_allocated_in_bytes_ > allocated_in_bytes
                                  ? max_allocated_in_bytes_
                                  : allocated_in_bytes;
  }

 public:
  MemStats() = default;

  ~MemStats() {
    if (allocated_in_bytes_ != 0) {
      DIPU_DEBUG_ALLOCATOR(
          8, "~MemStats: allocated_in_bytes_:" << allocated_in_bytes_);
    }
    if (reserved_in_bytes_ != 0) {
      DIPU_DEBUG_ALLOCATOR(
          2, "~MemStats: reserved_in_bytes_:" << reserved_in_bytes_);
    }
  }

  size_t memory_allocated() const { return allocated_in_bytes_; }

  size_t memory_reserved() const { return reserved_in_bytes_; }

  size_t max_memory_allocated() const { return max_allocated_in_bytes_; }

  size_t max_memory_reserved() const { return max_reserved_in_bytes_; }
};

class DIPU_API CacheAllocator : public c10::Allocator, public MemStats {
  c10::Allocator* raw_allocator_ = nullptr;
  AsyncMemPool* async_mem_pool_ = nullptr;
  mutable c10::Device device_ = c10::DeviceType::CPU;

 protected:
  c10::Allocator* raw_allocator() const { return raw_allocator_; }

  AsyncMemPool* async_mem_pool() const { return async_mem_pool_; }

  void* allocate_raw(size_t n) { return raw_allocator()->raw_allocate(n); }

  void free_raw(void* ptr) { return raw_allocator()->raw_deallocate(ptr); }

 public:
  CacheAllocator() = default;

  void set_raw_allocator(c10::Allocator* raw_allocator) {
    raw_allocator_ = raw_allocator;
    device_ = raw_allocator_->allocate(0).device();
  }

  void set_async_mem_pool(AsyncMemPool* async_mem_pool) {
    async_mem_pool_ = async_mem_pool;
  }

  ~CacheAllocator() override = default;

  virtual void empty_cache() const = 0;

  virtual void release_all_memory() const = 0;

  c10::Device& device() const { return device_; }

  class DataPtrContextBase {
   private:
    std::set<DIPUStream> streams_;
    mutable const CacheAllocator* allocator_ = nullptr;
    void* ptr_ = nullptr;
    size_t size_ = 0;

   public:
    DataPtrContextBase(const CacheAllocator* allocator, void* ptr, size_t size)
        : allocator_(allocator), ptr_(ptr), size_(size) {
      if (allocator_->device().type() == dipu::DIPU_DEVICE_TYPE) {
        auto current_stream = getCurrentDIPUStream();
        // If current stream is the default stream, we don't need to synchronize
        // But before releasing the memory we must synchronize the default
        // stream
        if (getDefaultDIPUStream() != current_stream) {
          streams_.insert(current_stream);
        }
      }
      MemChecker::instance().insert(ptr, size);
    }

    ~DataPtrContextBase() { MemChecker::instance().erase(ptr_); }

    std::set<DIPUStream>& streams() { return streams_; }

    const CacheAllocator* allocator() { return allocator_; }

    void* ptr() { return ptr_; }

    size_t size() const { return size_; }
  };
};

void setAllocator(const std::string &name, c10::DeviceType device_type,
                  const std::function<c10::Allocator*(int)> &allocator_geter,
                  uint8_t priority = 0);

c10::Allocator* getAllocator(c10::DeviceType device_type);

namespace allocator_details {  // For internal implementation only

struct AllocatorRegisterer {
  explicit AllocatorRegisterer(
      const std::string &name, c10::DeviceType device_type,
      const std::function<c10::Allocator*(int)> &allocator_get_fn,
      uint8_t priority = 0) {
    setAllocator(name, device_type, allocator_get_fn, priority);
  }
};

template <c10::DeviceType>
struct RawAllocator;

template <>
struct RawAllocator<dipu::DIPU_DEVICE_TYPE> {
  using type = DIPURawDeviceAllocator;
};

template <>
struct RawAllocator<at::DeviceType::CPU> {
  using type = DIPURawHostAllocator;
};

template <typename AllocatorImpl, class AsyncMemPoolImpl, int device_id>
c10::Allocator* get_allocator_impl(c10::Allocator* raw_allocator) {
  // Construct when really needed
  // async_mem_pool is used when cache_allocator being destructed so it should
  // be destructed after cache_allocator
  static AsyncMemPoolImpl async_mem_pool;
  static AllocatorImpl cache_allocator;
  static int n = [&]() {
    cache_allocator.set_raw_allocator(raw_allocator);
    cache_allocator.set_async_mem_pool(&async_mem_pool);
    return 0;
  }();
  return &cache_allocator;
}

template <class AllocatorImpl, class AsyncMemPoolImpl>
c10::Allocator* get_allocator(int device_id, c10::Allocator* raw_allocator) {
#define allocator_dispatch_device_id(id)                            \
  if (device_id == (id)) {                                            \
    return get_allocator_impl<AllocatorImpl, AsyncMemPoolImpl, id>( \
        raw_allocator);                                             \
  }

  allocator_dispatch_device_id(0);
  allocator_dispatch_device_id(1);
  allocator_dispatch_device_id(2);
  allocator_dispatch_device_id(3);
  allocator_dispatch_device_id(4);
  allocator_dispatch_device_id(5);
  allocator_dispatch_device_id(6);
  allocator_dispatch_device_id(7);
  allocator_dispatch_device_id(8);
  allocator_dispatch_device_id(9);
  allocator_dispatch_device_id(10);
  allocator_dispatch_device_id(11);
  allocator_dispatch_device_id(12);
  allocator_dispatch_device_id(13);
  allocator_dispatch_device_id(14);
  allocator_dispatch_device_id(15);
  TORCH_CHECK(false, "support up to 16 cards");
}

#define DIPU_REGISTER_ALLOCATOR(name, device_type, CachingAllocator, priority) \
  namespace name##device_type {  \
    static allocator_details::RawAllocator<at::DeviceType::device_type>::type raw_allocator;        \
    using AsyncMemPool = AsyncResourcePoolImpl<std::tuple<void*, size_t>, at::DeviceType::device_type, priority>;         \
    static const std::function<c10::Allocator*(int)> allocator_get_fn =              \
      std::bind(allocator_details::get_allocator<CachingAllocator, AsyncMemPool>,               \
                  std::placeholders::_1, &raw_allocator); \
    static const allocator_details::AllocatorRegisterer g_allocator(#name, at::DeviceType::device_type,      \
                                           allocator_get_fn, priority);        \
  }
}  // namespace allocator_details

}  // namespace dipu
