// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include "DIPUAllocator.h"
#include "DIPUAsyncResoursePool.h"
#include "DIPUEvent.h"

#include "DIPUStream.h"
#include <map>
#include <list>

namespace dipu {

using AsyncMemPool = AsyncResoursePool<std::tuple<void*, size_t>>;

class DIPU_API CacheAllocator: public c10::Allocator {
  c10::Allocator* raw_allocator_ = nullptr;
  AsyncMemPool* async_mem_pool_ = nullptr;

  protected:
    c10::Allocator* raw_allocator() const {
      return raw_allocator_;
    }

    AsyncMemPool* async_mem_pool() const {
       return async_mem_pool_;
    }

    void* allocate_raw(size_t n) {
      return raw_allocator()->raw_allocate(n);
    }

    void free_raw(void* ptr) {
      return raw_allocator()->raw_deallocate(ptr);
    }

  public:
    CacheAllocator() {

    }

    void set_raw_allocator(c10::Allocator* raw_allocator) {
      raw_allocator_ = raw_allocator;
    }

    void set_async_mem_pool(AsyncMemPool* async_mem_pool) {
      async_mem_pool_ = async_mem_pool;
    }

    virtual ~CacheAllocator() {

    };

    virtual void empty_cache() const = 0 ;
    virtual void release_all_memory() const = 0 ;
};

void setAllocator(const std::string name, c10::DeviceType device_type, std::function<c10::Allocator*(int)> allocator_get_fn, uint8_t priority = 0);

c10::Allocator* getAllocator(c10::DeviceType device_type);

void emptyCachedMem();

void releaseAllDeviceMem();

struct AllocatorRegisterer {
  explicit AllocatorRegisterer(const std::string name, c10::DeviceType device_type, std::function<c10::Allocator*(int)> allocator_get_fn, uint8_t priority = 0) {
    setAllocator(name, device_type, allocator_get_fn, priority);
  }
};

template<c10::DeviceType>
struct RawAllocator;

template<>
struct RawAllocator<dipu::DIPU_DEVICE_TYPE> {
  using type = DIPUDeviceAllocator;
};

template<>
struct RawAllocator<at::DeviceType::CPU> {
  using type = DIPUHostAllocator;
};

template<typename AllocatorImpl, int device_id>
c10::Allocator* get_allocator_impl(c10::Allocator* raw_allocator, AsyncMemPool* asyncMemPool) {
    // Construct when really needed
    static AllocatorImpl cache_allocator;
    static int n = [&](){
      cache_allocator.set_raw_allocator(raw_allocator);
      cache_allocator.set_async_mem_pool(asyncMemPool);
      return 0;
    }();
    return &cache_allocator;
}

template<class AllocatorImpl>
c10::Allocator* get_allocator(int device_id, c10::Allocator* raw_allocator, AsyncMemPool* asyncMemPool) {
  #define allocator_dispatch_device_id(id)                                        \
    if (device_id == id){                                                         \
      return get_allocator_impl<AllocatorImpl, id>(raw_allocator, asyncMemPool);  \
    }                                                                             \

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

#define DIPU_REGISTER_ALLOCATOR(name, device_type, CachingAllocator, priority)                                                                                      \
  namespace name##device_type{                                                                                                                                      \
  static RawAllocator<device_type>::type raw_allocator;                                                                                                             \
  static AsyncResoursePoolImpl<std::tuple<void*, size_t>, device_type, priority>  asyncMemPool;                                                                     \
  static std::function<c10::Allocator*(int)> allocator_get_fn = std::bind(get_allocator<CachingAllocator>, std::placeholders::_1, &raw_allocator, &asyncMemPool);   \
  static AllocatorRegisterer g_allocator(#name, device_type, allocator_get_fn,  priority);                                                                          \
  }




    //static AllocatorRegisterer g_allocator(#name, device_type, allocator_getter,  priority);                                                    \
  //static std::function<c10::Allocator*(int)> allocator_getter = get_allocator<CachingAllocator, (c10::Allocator*)&raw_allocator, (AsyncMemPool*)&asyncMemPool>;                 \
  //static AllocatorRegisterer g_allocator(#name, device_type, allocator_getter,  priority); \
  //  std::bind(&BFCachingAllocator::free_raw, (BFCachingAllocator*)this, std::placeholders::_1);


}  // namespace dipu

