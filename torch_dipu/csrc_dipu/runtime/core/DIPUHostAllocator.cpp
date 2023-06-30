// Copyright (c) 2023, DeepLink.

#include "DIPUHostAllocator.h"

#include <mutex>
#include <unordered_set>
#include <utility>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/MemChecker.h>

namespace dipu {

class DIPUHostAllocator final {
public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    void* data = nullptr;
    devapis::mallocHost(&data, size);
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

static DIPUHostAllocator dipu_host_allocator;
static void DIPUHostAllocatorDeleter(void* ctx) {
  dipu_host_allocator.free(ctx);
}

class DIPUHostAllocatorWrapper : public c10::Allocator {
public:
  c10::DataPtr allocate(size_t size) const {
    auto ptr_and_ctx = dipu_host_allocator.allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &DIPUHostAllocatorDeleter,
        at::DeviceType::CPU};
  }
};

static DIPUHostAllocatorWrapper dipu_host_allocator_wrapper;

at::Allocator* getHostAllocator() {
  return &dipu_host_allocator_wrapper;
}

bool isPinnedPtr(const void* ptr) {
  dipu_host_allocator.isPinnedPtr(ptr);
}

}  // namespace dipu
