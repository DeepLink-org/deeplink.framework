// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/base/environ.hpp"
#include "csrc_dipu/runtime/core/DIPUEvent.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include "csrc_dipu/utils/env.hpp"

#include "DIPUCachingDeviceAllocator.h"
#include "DIPUCachingHostAllocator.h"

namespace dipu {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex DIPURawDeviceAllocator::mutex_;

constexpr size_t kDefaultMaxAsyncResourcePoolLength = 96;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const size_t kMaxAsyncResourcePoolLength = get_env_or_default(
    "DIPU_MAX_ASYNC_RESOURCE_POOL_LENGTH", kDefaultMaxAsyncResourcePoolLength);

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MemoryAlignmentStrategy gDefaultMemoryAlignmentStrategy;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const MemoryAlignmentStrategy* memoryAlignmentStrategy =
    &gDefaultMemoryAlignmentStrategy;

}  // namespace

const MemoryAlignmentStrategy* getMemoryAlignmentStrategy() {
  return memoryAlignmentStrategy;
}

void setMemoryAlignmentStrategy(
    const MemoryAlignmentStrategy* memoryAlignStrategy) {
  memoryAlignmentStrategy = memoryAlignStrategy;
}

namespace {

// using RegisteredAllocator = std::map<c10::DeviceType, std::map<std::string,
// std::tuple<c10::Allocator*, uint8_t>>>; using RegisteredAllocator =
// std::map<c10::DeviceType, std::map<std::string,
// std::tuple<std::array<c10::Allocator*, 16>, uint8_t>>>;

using RegisteredAllocator = std::map<
    c10::DeviceType,
    std::map<std::string,
             std::tuple<std::function<c10::Allocator*(int)>, uint8_t>>>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<RegisteredAllocator> gDIPURegisteredAllocatorPtr;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex dipu_register_allocator_mutex;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::set<c10::Allocator*> used_allocator;

}  // namespace

void setAllocator(const std::string& name, c10::DeviceType device_type,
                  const std::function<c10::Allocator*(int)>& allocator_getter,
                  uint8_t priority) {
  std::lock_guard<std::mutex> lock(dipu_register_allocator_mutex);
  if (!gDIPURegisteredAllocatorPtr) {
    gDIPURegisteredAllocatorPtr = std::make_unique<RegisteredAllocator>();
  }
  auto& gDIPURegisteredAllocator = *gDIPURegisteredAllocatorPtr;
  if (gDIPURegisteredAllocator[device_type].count(name) <= 0) {
    gDIPURegisteredAllocator[device_type][name] =
        std::make_tuple(allocator_getter, priority);
  } else {
    if (std::get<1>(gDIPURegisteredAllocator[device_type][name]) < priority) {
      gDIPURegisteredAllocator[device_type][name] =
          std::make_tuple(allocator_getter, priority);
    } else {
      TORCH_CHECK(false,
                  "A higher priority allocator is already registered for the "
                  "same device:",
                  device_type, name, priority);
    }
  }
}

namespace {

int getDeviceIndex(const c10::Device& device, int host_index) {
  if (device.is_cpu()) {
    return host_index;
  }
  if (device.has_index()) {
    return device.index();
  }
  return devproxy::current_device();
}

c10::Allocator* createAllocator(const c10::Device& device) {
  c10::DeviceType device_type = device.type();
  c10::Allocator* result = nullptr;
  auto& gDIPURegisteredAllocator = *gDIPURegisteredAllocatorPtr;
  const std::string algorithm = (device_type == dipu::DIPU_DEVICE_TYPE
                                     ? environ::deviceMemCachingAlgorithm()
                                     : environ::hostMemCachingAlgorithm());
  if (gDIPURegisteredAllocator[device_type].count(algorithm) > 0) {
    auto allocator_geter =
        std::get<0>(gDIPURegisteredAllocator[device_type][algorithm]);
    int device_index = getDeviceIndex(device, 0);

    auto allocator = allocator_geter(device_index);
    if (device_type == dipu::DIPU_DEVICE_TYPE) {
      used_allocator.insert(allocator);
    }
    return allocator;
  }
  TORCH_CHECK(false,
              "No allocator found for the device using the given algorithm:",
              device_type, environ::deviceMemCachingAlgorithm());
  return nullptr;
}

}  // namespace

bool isTorchAllocator() {
  static bool is_torch_allocator =
      (environ::deviceMemCachingAlgorithm() == environ::kTorchAllocatorName);
  return is_torch_allocator;
}

c10::Allocator* getAllocator(const c10::Device& device) {
  // allocator_lookup_table[device_index] == device allocator
  // allocator_lookup_table[device_count] == host allocator
  if (!device.is_cpu() && isTorchAllocator()) {
    return allocator::getTorchAllocator();
  }
  if (device.is_cpu() && isTorchAllocator()) {
    return allocator::getCachingHostAllocator();
  }

  static const int device_count = devproxy::getDeviceCount();
  static const int host_index = device_count;
  static std::vector<c10::Allocator*> allocator_lookup_table(device_count + 1);
  int device_index = getDeviceIndex(device, host_index);
  auto& allocator = allocator_lookup_table[device_index];
  if (allocator == nullptr) {
    allocator = createAllocator(device);
  }
  return allocator;
}

c10::Allocator* getAllocator(c10::DeviceType device_type) {
  return getAllocator(c10::Device(device_type));
}

void emptyCachedMem() {
  if (isTorchAllocator()) {
    allocator::emptyCache();
    allocator::CachingHostAllocator_emptyCache();
    return;
  }

  auto function_name = __FUNCTION__;
  auto empty_allocator_cache = [&function_name](auto allocator) {
    auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
    DIPU_DEBUG_ALLOCATOR(8, function_name
                                << " allocator:" << allocator
                                << ", cached_allocator:" << cached_allocator);
    if (cached_allocator != nullptr) {
      cached_allocator->empty_cache();
    }
  };
  for (auto& allocator : used_allocator) {
    empty_allocator_cache(allocator);
  }
}

void releaseAllDeviceMem() {
  if (isTorchAllocator()) {
    allocator::emptyCache();
    allocator::CachingHostAllocator_emptyCache();
    return;
  }

  auto release_allocator_memory = [](auto allocator) {
    auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
    DIPU_DEBUG_ALLOCATOR(8, "release_allocator_memory: allocator:"
                                << allocator
                                << ", cached_allocator:" << cached_allocator);
    if (cached_allocator != nullptr) {
      cached_allocator->release_all_memory();
    }
  };
  for (auto& allocator : used_allocator) {
    release_allocator_memory(allocator);
  }
}

size_t memoryReserved(const c10::Device& device) {
  if (!device.is_cpu() && isTorchAllocator()) {
    allocator::DeviceStats stats = allocator::getDeviceStats(device.index());
    return stats
        .reserved_bytes[static_cast<int64_t>(allocator::StatType::AGGREGATE)]
        .current;
  }

  if (device.is_cpu() && isTorchAllocator()) {
    return 0;
  }

  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
    return cached_allocator->memory_reserved();
  }
  return 0;
}

size_t memoryAllocated(const c10::Device& device) {
  if (!device.is_cpu() && isTorchAllocator()) {
    allocator::DeviceStats stats = allocator::getDeviceStats(device.index());
    return stats
        .allocated_bytes[static_cast<int64_t>(allocator::StatType::AGGREGATE)]
        .current;
  }

  if (device.is_cpu() && isTorchAllocator()) {
    return 0;
  }

  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
    return cached_allocator->memory_allocated();
  }
  return 0;
}

size_t maxMemoryReserved(const c10::Device& device) {
  if (!device.is_cpu() && isTorchAllocator()) {
    allocator::DeviceStats stats = allocator::getDeviceStats(device.index());
    return stats
        .reserved_bytes[static_cast<int64_t>(allocator::StatType::AGGREGATE)]
        .peak;
  }

  if (device.is_cpu() && isTorchAllocator()) {
    return 0;
  }

  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
    return cached_allocator->max_memory_reserved();
  }
  return 0;
}

size_t maxMemoryAllocated(const c10::Device& device) {
  if (!device.is_cpu() && isTorchAllocator()) {
    allocator::DeviceStats stats = allocator::getDeviceStats(device.index());
    return stats
        .allocated_bytes[static_cast<int64_t>(allocator::StatType::AGGREGATE)]
        .peak;
  }

  if (device.is_cpu() && isTorchAllocator()) {
    return 0;
  }

  c10::Allocator* allocator = getAllocator(device);
  auto cached_allocator = dynamic_cast<CacheAllocator*>(allocator);
  if (cached_allocator != nullptr) {
    return cached_allocator->max_memory_allocated();
  }
  return 0;
}

void resetPeakStats(const c10::Device& device) {
  if (!device.is_cpu() && isTorchAllocator()) {
    return allocator::resetPeakStats(device.index());
  }
}

void recordStream(const c10::DataPtr& ptr, const DIPUStream& stream) {
  if (isTorchAllocator()) {
    allocator::recordStream(ptr, stream);
    return;
  }

  using pointer = CacheAllocator::DataPtrContextBase*;
  if (auto ctx = static_cast<pointer>(ptr.get_context())) {
    ctx->streams().insert(stream);
  }
}

void recordStream(const at::Tensor& tensor, const DIPUStream& stream) {
  dipu::recordStream(tensor.storage().data_ptr(), stream);
}

namespace {
class DIPUDeviceCachingProxy : public c10::Allocator {
  c10::DeviceType device_type_;

 public:
  explicit DIPUDeviceCachingProxy(c10::DeviceType device_type)
      : device_type_(device_type) {}

  ~DIPUDeviceCachingProxy() override = default;

  c10::DataPtr allocate(size_t size) const override {
    return getAllocator(device_type_)->allocate(size);
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return getAllocator(device_type_)->raw_deleter();
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DIPUDeviceCachingProxy dipu_default_device_allocator(dipu::DIPU_DEVICE_TYPE);
};  // namespace

void initCachedAllocator() {
  // Make the c10::GetAllocator interface available
  constexpr int kPriority = 255;
  c10::SetAllocator(dipu::DIPU_DEVICE_TYPE, &dipu_default_device_allocator,
                    kPriority);
  c10::SetAllocator(c10::DeviceType::CUDA, &dipu_default_device_allocator,
                    kPriority);
}

}  // namespace dipu
