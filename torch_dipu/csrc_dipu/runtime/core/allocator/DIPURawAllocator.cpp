// Copyright (c) 2023, DeepLink.

#include "DIPURawAllocator.h"

#include <mutex>
#include <unordered_set>
#include <utility>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {

static void DIPURawDeviceAllocatorDeleter(void *ptr) {
    if (ptr) {
      auto device = devproxy::current_device();
      DIPU_DEBUG_ALLOCATOR(2, "devproxy::freeDevice: free " << ptr);
      devproxy::freeDevice(ptr);
      ptr = nullptr;
    }
}

DIPURawDeviceAllocator::DIPURawDeviceAllocator() {

}

c10::DataPtr DIPURawDeviceAllocator::allocate(size_t size) const {
  auto idx = devproxy::current_device();
  return this->allocate(size, idx);
}

c10::DeleterFnPtr DIPURawDeviceAllocator::raw_deleter() const {
  return &DIPURawDeviceAllocatorDeleter;
}

c10::DataPtr DIPURawDeviceAllocator::allocate(size_t nbytes, c10::DeviceIndex device_index) const {
      std::lock_guard<std::mutex> lock(mutex_);
      void *data = nullptr;
      if (nbytes > 0) {
        devproxy::mallocDevice(&data, nbytes);
        DIPU_DEBUG_ALLOCATOR(1, "devproxy::mallocDevice: malloc " << nbytes << " nbytes, ptr:" << data);
      }
      return {data, data, &DIPURawDeviceAllocatorDeleter, c10::Device(dipu::DIPU_DEVICE_TYPE, device_index)};
}

class DIPURawHostAllocatorImpl final {
public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    void* data = nullptr;
    devproxy::mallocHost(&data, size);
    DIPU_DEBUG_ALLOCATOR(1, "devproxy::mallocHost: malloc " << size << " nbytes, ptr:" << data);
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
    devproxy::freeHost(ctx);
    DIPU_DEBUG_ALLOCATOR(2, "devproxy::freeHost: free " << ctx);
    ctx = nullptr;
  }

  bool isPinnedPtr(const void *p) {
    bool is_pinned = false;
    {
      //std::lock_guard<std::mutex> lck(mtx_);
      for (auto iter = blocks_.crbegin(); iter != blocks_.crend(); iter++) {
        const void* ptr = iter->first;
        const size_t size = iter->second;
        const char* cptr = static_cast<const char*>(ptr);
        const char* cp = static_cast<const char*>(p);
        const char* max_ptr = cptr + size;
        if (cp >= cptr && cp < max_ptr) {
          is_pinned = true;
          break;
        } else {
          if (cp >= max_ptr) {
            break;
          }
        }
      }
    }
    return is_pinned;
  }

private:
  static std::mutex mtx_;
  static std::map<void*, size_t> blocks_;
};

std::map<void*, size_t> DIPURawHostAllocatorImpl::blocks_;
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
