// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include "DIPUDeviceAllocator.h"
#include "DIPUHostAllocator.h"
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

    virtual void empty_cache() {};
};

void setAllocator(const std::string name, c10::DeviceType device_type, c10::Allocator* allocator, uint8_t priority = 0);
c10::Allocator* getAllocator(c10::DeviceType device_type);

struct AllocatorRegisterer {
  explicit AllocatorRegisterer(const std::string name, c10::DeviceType device_type, c10::Allocator* allocator, uint8_t priority = 0) {
    setAllocator(name, device_type, allocator, priority);
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
  //using type = DIPUHostAllocator;  // TODO: support pin memory
};


#define DIPU_REGISTER_ALLOCATOR(name, device_type, CachingAllocator, priority)                                   \
  namespace {                                                                                                    \
  static RawAllocator<device_type>::type raw_allocator;                                                          \
  static AsyncResoursePoolImpl<std::tuple<void*, size_t>, device_type, priority>  asyncMemPool;                  \
  static CachingAllocator cache_allocator;                                                                       \
  static int n = [&](){                                                                                          \
    cache_allocator.set_raw_allocator(&raw_allocator);                                                           \
    cache_allocator.set_async_mem_pool(&asyncMemPool);                                                           \
    return 0;                                                                                                    \
  }();                                                                                                           \
  static AllocatorRegisterer g_allocator_d(name, device_type, &cache_allocator,  priority);                      \
  }


}  // namespace dipu

