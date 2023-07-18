// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <map>
#include <tuple>
#include <set>

namespace dipu {

std::mutex DIPURawDeviceAllocator::mutex_;

namespace {

//using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<c10::Allocator*, uint8_t>>>;
//using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<std::array<c10::Allocator*, 16>, uint8_t>>>;

using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<std::function<c10::Allocator*(int)>, uint8_t>>>;

static std::unique_ptr<RegisteredAllocator> gDIPURegisterdAllocatorPtr;

static std::mutex dipu_register_allocator_mutex;

static DIPURawDeviceAllocator lowest_priority_device_allocator;
static int n = [&]() {
  c10::SetAllocator(dipu::DIPU_DEVICE_TYPE, &lowest_priority_device_allocator, 0);
  return 0;
}();

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

c10::Allocator*  getAllocator(c10::DeviceType device_type) {
  c10::Allocator* result = nullptr;
  auto& gDIPURegisterdAllocator = *gDIPURegisterdAllocatorPtr;
  const std::string algorithm = (device_type == dipu::DIPU_DEVICE_TYPE ? dipu_device_memcaching_algorithm : dipu_host_memcaching_algorithm);
  if (gDIPURegisterdAllocator[device_type].count(algorithm) > 0) {
    auto allocator_geter = std::get<0>(gDIPURegisterdAllocator[device_type][algorithm]);
    int device_index =  (device_type == dipu::DIPU_DEVICE_TYPE) ? devapis::current_device() : 0;
    auto allocator = allocator_geter(device_index);
    if(device_type == dipu::DIPU_DEVICE_TYPE) {
      used_allocator.insert(allocator);
    }
    return allocator;
  }
  TORCH_CHECK(false, "No allocator found for the device using the given algorithm:", device_type, dipu_device_memcaching_algorithm);
  return nullptr;
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


}  // namespace dipu
