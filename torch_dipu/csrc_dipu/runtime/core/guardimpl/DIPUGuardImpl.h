// Copyright (c) 2023, DeepLink.

#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

struct DIPUGuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_type = dipu::DIPU_DEVICE_TYPE;
  DIPUGuardImpl() {}
  explicit DIPUGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == dipu::DIPU_DEVICE_TYPE);
  }
  at::DeviceType type() const override {
    return dipu::DIPU_DEVICE_TYPE;
  }

  c10::Device exchangeDevice(c10::Device device) const override {
    AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
    c10::Device old_device = this->getDevice();
    if (old_device.index() != device.index()) {
      setDevice(device);
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(dipu::DIPU_DEVICE_TYPE, devproxy::current_device());
  }

  void setDevice(c10::Device device) const override {
    if  (devproxy::current_device() < 0) return;
    AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
    devproxy::setDevice(device.index());
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if (devproxy::current_device() < 0 ) return;
    devproxy::setDevice(device.index());
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    auto oldStream = getCurrentDIPUStream(s.device().index());
    DIPUStream stream(s);
    setCurrentDIPUStream(stream);
    return c10::Stream(c10::Stream::UNSAFE,
                       s.device(),
                       static_cast<c10::StreamId>(oldStream.id()));
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return devproxy::getDeviceCount();
  }

  c10::Stream getDefaultStream(c10::Device device) const override {
    DIPUStream stream = getCurrentDIPUStream(device.index());
    return c10::Stream(c10::Stream::UNSAFE,
                       device,
                       static_cast<c10::StreamId>(stream.id()));
  }

  void record(
    void** event,
    const c10::Stream& s,
    const c10::DeviceIndex device_index,
    const c10::EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == s.device_index(),
    "Event device index ",
    device_index,
    " does not match recording stream's device index ",
    s.device_index(),
    ".");

    deviceEvent_t dipu_event = static_cast<deviceEvent_t>(*event);
    DIPUStream stream(s);
    deviceStream_t raw_stream = stream.rawstream();

    // Moves to queue's device to record
    const c10::Device orig_device = this->getDevice();
    devproxy::setDevice(stream.device_index());

    // Create the Notifier
    if (!dipu_event) {
      devproxy::createEvent(&dipu_event);
    }
    devproxy::recordEvent(dipu_event, raw_stream);
    *event = dipu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(
    void* event,
    const c10::Stream& s) const override {
    if (!event) return;
    deviceEvent_t dipu_event = static_cast<deviceEvent_t>(event);
    const auto orig_device = this->getDevice();
    setDevice(s.device());
    DIPUStream stream(s);
    devproxy::streamWaitEvent(stream.rawstream(), dipu_event);
    setDevice(orig_device);
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto dipu_event = static_cast<deviceEvent_t>(event);
    const c10::Device orig_device = this->getDevice();
    devproxy::setDevice(device_index);

    devproxy::destroyEvent(dipu_event);
    setDevice(orig_device);
  }

  void recordDataPtrOnStream(const c10::DataPtr& dataptr, const c10::Stream& stream) const override {
    // todo: DIPUCachingAllocator::recordStream(dataptr, stream);
  }
};

}  // namespace dipu
