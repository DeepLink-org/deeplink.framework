// Copyright (c) 2023, DeepLink.
#include "allocator.h"
#include <map>

namespace dipu {

std::mutex DIPUAllocator::mutex_;

namespace {

static DIPUAllocator allocator;

static std::map<std::string, c10::Allocator*> gDIPURegisterdAllocator;

}  // namespace

REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, &allocator);


void setAllocator(const std::string name, c10::DeviceType device_type, c10::Allocator* allocator, uint8_t priority) {
  TORCH_CHECK(allocator);
  gDIPURegisterdAllocator[name] = allocator;
}

c10::Allocator*  getAllocator(c10::DeviceType device_type) {
  if (gDIPURegisterdAllocator.size() <= 0 && device_type == dipu::DIPU_DEVICE_TYPE) {
    return &allocator;
  }
  TORCH_CHECK(false, "No suitable allocators registered");
  return nullptr;
}

}  // namespace dipu
