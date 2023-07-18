// Copyright (c) 2023, DeepLink.

#include "DIPURawAllocator.h"

#include <mutex>
#include <unordered_set>
#include <utility>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/MemChecker.h>

namespace dipu {

static void DIPURawDeviceAllocatorDeleter(void *ptr) {
    if (ptr) {
      MemChecker::instance().erase(ptr);
      auto device = devapis::current_device();
      devapis::setDevice(device);
      DIPU_DEBUG_ALLOCATOR(2, "devapis::freeDevice: free " << ptr);
      devapis::freeDevice(ptr);
      ptr = nullptr;
    }
}

DIPURawDeviceAllocator::DIPURawDeviceAllocator() {
  auto device = devapis::current_device();
  devapis::setDevice(device);
}

c10::DataPtr DIPURawDeviceAllocator::allocate(size_t size) const {
  auto idx = devapis::current_device();
  devapis::setDevice(idx);
  return this->allocate(size, idx);
}

c10::DeleterFnPtr DIPURawDeviceAllocator::raw_deleter() const {
  return &DIPURawDeviceAllocatorDeleter;
}

c10::DataPtr DIPURawDeviceAllocator::allocate(size_t nbytes, c10::DeviceIndex device_index) const {
      std::lock_guard<std::mutex> lock(mutex_);
      void *data = nullptr;
      if (nbytes > 0) {
        devapis::mallocDevice(&data, nbytes);
        DIPU_DEBUG_ALLOCATOR(1, "devapis::mallocDevice: malloc " << nbytes << " nbytes, ptr:" << data);
      }
      MemChecker::instance().insert(data, nbytes);
      return {data, data, &DIPURawDeviceAllocatorDeleter, c10::Device(dipu::DIPU_DEVICE_TYPE, device_index)};
}

class DIPURawHostAllocatorImpl final {
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
      blocks_[data] = size;
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
      for (auto iter = blocks_.begin(); iter != blocks_.end(); iter++) {
        const void* ptr = iter->first;
        const size_t size = iter->second;
        const char* cptr = static_cast<const char*>(ptr);
        const char* cp = static_cast<const char*>(p);
        if (cp >= cptr && cp < (cptr + size)) {
          return true;
        }
      }
    }
    return is_pinned;
  }

private:
  static std::mutex mtx_;
  static std::unordered_map<void*, size_t> blocks_;
};

std::unordered_map<void*, size_t> DIPURawHostAllocatorImpl::blocks_;
std::mutex DIPURawHostAllocatorImpl::mtx_;

namespace {

static DIPURawHostAllocatorImpl dipu_host_allocator;

static void DIPURawHostAllocatorDeleter(void* ctx) {
  dipu_host_allocator.free(ctx);
}

}

c10::DeleterFnPtr DIPURawHostAllocator::raw_deleter() const {
      return &DIPURawHostAllocatorDeleter;
}

 c10::DataPtr DIPURawHostAllocator::allocate(size_t size) const {
    auto ptr_and_ctx = dipu_host_allocator.allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &DIPURawHostAllocatorDeleter,
        at::DeviceType::CPU};
  }

bool isPinnedPtr(const void* ptr) {
  return dipu_host_allocator.isPinnedPtr(ptr);
}

}  // namespace dipu
