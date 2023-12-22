// Copyright (c) 2023, DeepLink.

#pragma once

#include <limits>

#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/core/allocator/DIPUCachingAllocatorUtils.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>
namespace dipu {
struct DIPUGuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_type = dipu::DIPU_DEVICE_TYPE;
  DIPUGuardImpl() = default;
  explicit DIPUGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == dipu::DIPU_DEVICE_TYPE);
  }
  at::DeviceType type() const override { return dipu::DIPU_DEVICE_TYPE; }

  c10::Device exchangeDevice(c10::Device device) const override {
    AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
    c10::Device old_device = this->getDevice();
    if (old_device.index() != device.index()) {
      setDevice(device);
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return {dipu::DIPU_DEVICE_TYPE, devproxy::current_device()};
  }

  void setDevice(c10::Device device) const override {
    if (devproxy::current_device() < 0) {
      return;
    }
    AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
    devproxy::setDevice(device.index());
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if (devproxy::current_device() < 0) {
      return;
    }
    devproxy::setDevice(device.index());
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return getCurrentDIPUStream(device.index()).unwrap();
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    auto oldStream = getCurrentDIPUStream(s.device().index());
    auto stream = DIPUStream(s);
    setCurrentDIPUStream(stream);
    return c10::Stream(c10::Stream::UNSAFE, s.device(),
                       static_cast<c10::StreamId>(oldStream.id()));
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(devproxy::getDeviceCount());
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d,
                                      bool isHighPriority) const override {
    return getDIPUStreamFromPool(d.index()).unwrap();
  }

  c10::Stream getDefaultStream(c10::Device device) const override {
    return getDefaultDIPUStream(device.index()).unwrap();
  }

  void record(void** event, const c10::Stream& s,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == s.device_index(),
                "Event device index ", device_index,
                " does not match recording stream's device index ",
                s.device_index(), ".");

    auto dipu_event = static_cast<deviceEvent_t>(*event);
    auto stream = DIPUStream(s);

    // Moves to queue's device to record
    const c10::Device orig_device = this->getDevice();
    devproxy::setDevice(stream.device_index());

    // Create the Notifier
    if (!dipu_event) {
      devproxy::createEvent(&dipu_event);
    }
    devproxy::recordEvent(dipu_event, stream.rawstream());
    *event = dipu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const c10::Stream& s) const override {
    if (!event) {
      return;
    }
    auto dipu_event = static_cast<deviceEvent_t>(event);
    const auto orig_device = this->getDevice();
    setDevice(s.device());
    auto stream = DIPUStream(s);
    devproxy::streamWaitEvent(stream.rawstream(), dipu_event);
    setDevice(orig_device);
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event) {
      return;
    }
    auto dipu_event = static_cast<deviceEvent_t>(event);
    const c10::Device orig_device = this->getDevice();
    devproxy::setDevice(device_index);
    devproxy::destroyEvent(dipu_event);
    setDevice(orig_device);
  }
  // call from ivalue_inl.h  synchronizeWithCurrentStreams with 'current stream'
  // = default stream. it's useless in ddp, because output tensor is record with
  // comm stream in colletive(), but may be useful in other communication mode.
  void recordDataPtrOnStream(const c10::DataPtr& dataptr,
                             const c10::Stream& s) const override {
    auto stream = DIPUStream(s);
    if (stream != getDefaultDIPUStream()) {
      dipu::recordStream(dataptr, stream);
    }
  }
};

}  // namespace dipu
