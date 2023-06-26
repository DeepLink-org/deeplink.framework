// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/MemChecker.h>

namespace dipu {

static std::mutex dipu_mutex;

static void DIPUDeleter(void* ptr) {
  if (ptr) {
    MemChecker::instance().erase(ptr);
    devapis::freeDevice(ptr);
    ptr = nullptr;
  }
}

class DIPU_API DIPUAllocator: public c10::Allocator {
public:
  inline virtual c10::DataPtr allocate(size_t size) const {
    auto idx = devapis::current_device();
    return this->allocate(size, idx);
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &DIPUDeleter;
  }
private:
  c10::DataPtr allocate(size_t nbytes, c10::DeviceIndex device_index) const {
    std::lock_guard<std::mutex> lock(dipu_mutex);
    void* data = nullptr;
    devapis::mallocDevice(&data, nbytes);
    MemChecker::instance().insert(data, nbytes);
    return {data, data, &DIPUDeleter, c10::Device(dipu::DIPU_DEVICE_TYPE, device_index)};
  }
};


}  // namespace torch_dipu
