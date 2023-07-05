// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include "DIPUDeviceAllocator.h"
#include "DIPUHostAllocator.h"
#include "DIPUEvent.h"

#include "DIPUStream.h"
#include <map>

namespace dipu {

class DIPU_API CacheAllocator: public c10::Allocator {
  c10::Allocator* raw_allocator_ = nullptr;

  protected:
    c10::Allocator* raw_allocator() const {
      return raw_allocator_;
    }

    void* allocate_raw(size_t n) {
      return raw_allocator()->raw_allocate(n);
    }

    void free_raw(void* ptr) {
      return raw_allocator()->raw_deallocate(ptr);
    }

  public:
    explicit CacheAllocator(c10::Allocator* raw_allocator):raw_allocator_(raw_allocator) {
      TORCH_CHECK(raw_allocator_);
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


#define DIPU_REGISTER_ALLOCATOR(name, device_type, CachingAllocator, priority)                      \
  namespace {                                                                                       \
  static RawAllocator<device_type>::type raw_allocator;                                             \
  static CachingAllocator cache_allocator(&raw_allocator);                                          \
  static AllocatorRegisterer g_allocator_d(name, device_type, &cache_allocator, priority);          \
  }


}  // namespace dipu

