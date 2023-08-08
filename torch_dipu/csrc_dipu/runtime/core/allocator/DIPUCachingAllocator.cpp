// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <map>
#include <tuple>
#include <set>
#include <atomic>


namespace dipu {

std::mutex DIPURawDeviceAllocator::mutex_;

namespace {

//using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<c10::Allocator*, uint8_t>>>;
//using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<std::array<c10::Allocator*, 16>, uint8_t>>>;

using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<std::function<c10::Allocator*(int)>, uint8_t>>>;

static std::unique_ptr<RegisteredAllocator> gDIPURegisterdAllocatorPtr;

static std::mutex dipu_register_allocator_mutex;

static std::set<c10::Allocator*> used_allocator;

}  // namespace

constexpr const char* dipu_default_memcaching_algorithm = "BS";

std::string dipu_device_memcaching_algorithm = []() {
  const char* env = std::getenv("DIPU_DEVICE_MEMCACHING_ALGORITHM");
  return env ? env : dipu_default_memcaching_algorithm;
}();

std::string dipu_host_memcaching_algorithm = []() {
  const char* env = std::getenv("DIPU_HOST_MEMCACHING_ALGORITHM");
  return env ? env : dipu_default_memcaching_algorithm;
}();

void setAllocator(const std::string name, c10::DeviceType device_type, std::function<c10::Allocator*(int)> allocator_geter, uint8_t priority) {
  std::lock_guard<std::mutex> lock(dipu_register_allocator_mutex);
  if (!gDIPURegisterdAllocatorPtr) {
    gDIPURegisterdAllocatorPtr = std::make_unique<RegisteredAllocator>();
  }
  auto& gDIPURegisterdAllocator = *gDIPURegisterdAllocatorPtr;
  if (gDIPURegisterdAllocator[device_type].count(name) <= 0) {
    gDIPURegisterdAllocator[device_type][name] = std::make_tuple(allocator_geter, priority);
  } else {
    if (std::get<1>(gDIPURegisterdAllocator[device_type][name]) < priority) {
      gDIPURegisterdAllocator[device_type][name] = std::make_tuple(allocator_geter, priority);
    } else {
        TORCH_CHECK(false, "A higher priority allocator is already registered for the same device:", device_type, name, priority);
    }
  }
}

c10::Allocator*  getAllocator(const c10::Device& device) {
  c10::DeviceType device_type = device.type();
  c10::Allocator* result = nullptr;
  auto& gDIPURegisterdAllocator = *gDIPURegisterdAllocatorPtr;
  const std::string algorithm = (device_type == dipu::DIPU_DEVICE_TYPE ? dipu_device_memcaching_algorithm : dipu_host_memcaching_algorithm);
  if (gDIPURegisterdAllocator[device_type].count(algorithm) > 0) {
    auto allocator_geter = std::get<0>(gDIPURegisterdAllocator[device_type][algorithm]);
    int device_index = 0;
    if (device_type == dipu::DIPU_DEVICE_TYPE) {
      device_index = device.has_index() ? device.index() : devproxy::current_device();
    }

    auto allocator = allocator_geter(device_index);
    if(device_type == dipu::DIPU_DEVICE_TYPE) {
      used_allocator.insert(allocator);
    }
    return allocator;
  }
  TORCH_CHECK(false, "No allocator found for the device using the given algorithm:", device_type, dipu_device_memcaching_algorithm);
  return nullptr;
}

c10::Allocator* getAllocator(c10::DeviceType device_type) {
  return getAllocator(c10::Device(device_type));
}

void emptyCachedMem() {
  auto empty_allocator_cache = [](auto allocator) {
    auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
    DIPU_DEBUG_ALLOCATOR(8, __FUNCTION__ << " allocator:" << allocator << ", cached_allocator:" << cached_allocator);
    if (cached_allocator != nullptr) {
      cached_allocator->empty_cache();
    }
  };
  for (auto& allocator : used_allocator) {
    empty_allocator_cache(allocator);
  }
}

void releaseAllDeviceMem() {
  auto release_allocator_memory = [](auto allocator) {
    auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
    DIPU_DEBUG_ALLOCATOR(8, "release_allocator_memory: allocator:" << allocator << ", cached_allocator:" << cached_allocator);
    if (cached_allocator != nullptr) {
      cached_allocator->release_all_memory();
    }
  };
  for (auto& allocator : used_allocator) {
    release_allocator_memory(allocator);
  }
}

size_t memoryReserved(const c10::Device& device) {
  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
      return cached_allocator->memory_reserved();
  }
  return 0;
}

size_t memoryAllocated(const c10::Device& device) {
  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
      return cached_allocator->memory_allocated();
  }
  return 0;
}

size_t maxMemoryReserved(const c10::Device& device) {
  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
      return cached_allocator->max_memory_reserved();
  }
  return 0;
}

size_t maxMemoryAllocated(const c10::Device& device) {
  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
      return cached_allocator->max_memory_allocated();
  }
  return 0;
}

void recordStream(const c10::DataPtr& ptr, DIPUStream stream) {
  void* ctx = ptr.get_context();
  if(ctx == nullptr) {
    return;
  }
  auto base_cxt = static_cast<CacheAllocator::DataPtrContextBase*>(ctx);
  if (base_cxt) {
    base_cxt->streams().insert(stream);
  }
}

void recordStream(const at::Tensor& tensor, DIPUStream stream) {
   dipu::recordStream(tensor.storage().data_ptr(), stream);
}

namespace {
  class DIPUDeviceCachingProxy: public c10::Allocator {
    c10::DeviceType device_type_;
  public:
    DIPUDeviceCachingProxy(c10::DeviceType device_type):device_type_(device_type) {

    }

    ~DIPUDeviceCachingProxy() {

    }

    c10::DataPtr allocate(size_t size) const {
      return getAllocator(device_type_)->allocate(size);
    }

    c10::DeleterFnPtr raw_deleter() const override {
      return getAllocator(device_type_)->raw_deleter();
    }
  };

  // Make the c10::GetAllocator interface available
  static DIPUDeviceCachingProxy dipu_default_device_allocator(dipu::DIPU_DEVICE_TYPE);
  static int m = [&]() {
    c10::SetAllocator(dipu::DIPU_DEVICE_TYPE, &dipu_default_device_allocator, 255);
    c10::SetAllocator(c10::DeviceType::CUDA, &dipu_default_device_allocator, 255);


    //allocator.store((CUDAAllocator*)&dipu_default_device_allocator);

    return 0;
  }();

};


}  // namespace dipu
