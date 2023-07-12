// Copyright (c) 2023, DeepLink.

#include "DIPUAllocator.h"

#include <mutex>
#include <unordered_set>
#include <utility>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/MemChecker.h>

namespace dipu {

static void DIPUDeviceAllocatorDeleter(void *ptr) {
    if (ptr) {
      MemChecker::instance().erase(ptr);
      auto device = devapis::current_device();
      devapis::setDevice(device);
      DIPU_DEBUG_ALLOCATOR(2, "devapis::freeDevice: free " << ptr);
      devapis::freeDevice(ptr);
      ptr = nullptr;
    }
}

DIPUDeviceAllocator::DIPUDeviceAllocator() {
  auto device = devapis::current_device();
  devapis::setDevice(device);
}

c10::DataPtr DIPUDeviceAllocator::allocate(size_t size) const {
  auto idx = devapis::current_device();
  devapis::setDevice(idx);
  return this->allocate(size, idx);
}

c10::DeleterFnPtr DIPUDeviceAllocator::raw_deleter() const {
  return &DIPUDeviceAllocatorDeleter;
}

c10::DataPtr DIPUDeviceAllocator::allocate(size_t nbytes, c10::DeviceIndex device_index) const {
      std::lock_guard<std::mutex> lock(mutex_);
      void *data = nullptr;
      if (nbytes > 0) {
        devapis::mallocDevice(&data, nbytes);
      }
      DIPU_DEBUG_ALLOCATOR(1, "devapis::mallocDevice: malloc " << nbytes << " nbytes, ptr:" << data);
      MemChecker::instance().insert(data, nbytes);
      return {data, data, &DIPUDeviceAllocatorDeleter, c10::Device(dipu::DIPU_DEVICE_TYPE, device_index)};
}

class DIPUHostAllocatorImpl final {
public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    void* data = nullptr;
    devapis::mallocHost(&data, size);
    DIPU_DEBUG_ALLOCATOR(1, "devapis::mallocHost: malloc " << size << " nbytes, ptr:" << data);
    MemChecker::instance().insert(data, size);
    {
      std::lock_guard<std::mutex> lck(mtx_);
      blocks_.insert(data);
    }
    return {data, data};
  }

  void free(void* ctx) {
    if (ctx == nullptr) {
      return;
    }

    {
      std::lock_guard<std::mutex> lck(mtx_);
      blocks_.erase(ctx);
    }
    MemChecker::instance().erase(ctx);
    devapis::freeHost(ctx);
    DIPU_DEBUG_ALLOCATOR(2, "devapis::freeHost: free " << ctx);
    ctx = nullptr;
  }

  bool isPinnedPtr(const void *p) {
    bool is_pinned = false;
    {
      std::lock_guard<std::mutex> lck(mtx_);
      is_pinned = (blocks_.find(p) != blocks_.end());
    }
    return is_pinned;
  }

private:
  std::mutex mtx_;
  std::unordered_set<const void*> blocks_;
};

namespace {

static DIPUHostAllocatorImpl dipu_host_allocator;

static void DIPUHostAllocatorDeleter(void* ctx) {
  dipu_host_allocator.free(ctx);
}

}

c10::DeleterFnPtr DIPUHostAllocator::raw_deleter() const {
      return &DIPUHostAllocatorDeleter;
}

 c10::DataPtr DIPUHostAllocator::allocate(size_t size) const {
    auto ptr_and_ctx = dipu_host_allocator.allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &DIPUHostAllocatorDeleter,
        at::DeviceType::CPU};
  }

bool isPinnedPtr(const void* ptr) {
  dipu_host_allocator.isPinnedPtr(ptr);
}

}  // namespace dipu
