// Copyright (c) 2023, DeepLink.
#include "allocator.h"
#include <map>
#include <tuple>

namespace dipu {

std::mutex DIPUAllocator::mutex_;

namespace {

static DIPUAllocator allocator;

using RegisteredAllocator = std::map<std::string, std::tuple<c10::Allocator*, c10::DeviceType, uint8_t>>;

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
  (*gDIPURegisterdAllocatorPtr)[name] = std::make_tuple(allocator, device_type, priority);
}

constexpr const char* dipu_default_memcaching_algorithm = "RAW";

std::string dipu_memcaching_algorithm = []() {
  const char* env = std::getenv("DIPU_MEMCACHING_ALGORITHM");
  return env ? env : dipu_default_memcaching_algorithm;
}();

c10::Allocator*  getAllocator(c10::DeviceType device_type) {
  std::lock_guard<std::mutex> lock(dipu_register_allocator_mutex);
  c10::Allocator* result = nullptr;
  if (!gDIPURegisterdAllocatorPtr) {
    if (device_type == dipu::DIPU_DEVICE_TYPE) {
      return &allocator;
    }
  } else {
    auto iter = gDIPURegisterdAllocatorPtr->find(dipu_memcaching_algorithm);
    if (iter != gDIPURegisterdAllocatorPtr->end()) {
      return std::get<0>(iter->second);
    }
  }
  TORCH_CHECK(false, "No suitable allocators registered");
  return nullptr;
}

}  // namespace dipu
