// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <map>
#include <tuple>

namespace dipu {

std::mutex DIPUDeviceAllocator::mutex_;

namespace {

static DIPUDeviceAllocator allocator;

using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string, std::tuple<c10::Allocator*, uint8_t>>>;

static std::unique_ptr<RegisteredAllocator> gDIPURegisterdAllocatorPtr;

static std::mutex dipu_register_allocator_mutex;

REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, &allocator);

}  // namespace

void setAllocator(const std::string name, c10::DeviceType device_type, c10::Allocator* allocator, uint8_t priority) {
  TORCH_CHECK(allocator);
  std::lock_guard<std::mutex> lock(dipu_register_allocator_mutex);
  if (!gDIPURegisterdAllocatorPtr) {
    gDIPURegisterdAllocatorPtr = std::make_unique<RegisteredAllocator>();
  }
  auto& gDIPURegisterdAllocator = *gDIPURegisterdAllocatorPtr;
  if (gDIPURegisterdAllocator[device_type].count(name) <= 0) {
    gDIPURegisterdAllocator[device_type][name] = std::make_tuple(allocator, priority);
  } else {
    if (std::get<1>(gDIPURegisterdAllocator[device_type][name]) < priority) {
        gDIPURegisterdAllocator[device_type][name] = std::make_tuple(allocator, priority);
    } else {
        TORCH_CHECK(false, "A higher priority allocator is already registered for the same device:", device_type, name, priority);
    }
  }
}

constexpr const char* dipu_default_memcaching_algorithm = "BS";

std::string dipu_memcaching_algorithm = []() {
  const char* env = std::getenv("DIPU_MEMCACHING_ALGORITHM");
  return env ? env : dipu_default_memcaching_algorithm;
}();

c10::Allocator*  getAllocator(c10::DeviceType device_type) {
  std::lock_guard<std::mutex> lock(dipu_register_allocator_mutex);
  c10::Allocator* result = nullptr;
  auto& gDIPURegisterdAllocator = *gDIPURegisterdAllocatorPtr;

  if (gDIPURegisterdAllocator[device_type].count(dipu_memcaching_algorithm) > 0) {
    return std::get<0>(gDIPURegisterdAllocator[device_type][dipu_memcaching_algorithm]);
  }
  TORCH_CHECK(false, "No allocator found for the device using the given algorithm:", device_type, dipu_memcaching_algorithm);
  return nullptr;
}

void emptyCachedMem() {
  auto allocator = getAllocator(dipu::DIPU_DEVICE_TYPE);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  DIPU_DEBUG_ALLOCATOR(10, __FUNCTION__ << " allocator:" << allocator << ", cached_allocator:" << cached_allocator);
  if (cached_allocator != nullptr) {
    cached_allocator->empty_cache();
  }
}


}  // namespace dipu
