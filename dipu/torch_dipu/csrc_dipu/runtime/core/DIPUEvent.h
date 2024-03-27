// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

#include "DIPUGuard.h"
#include "DIPUStream.h"

namespace dipu {
/*
 * DIPUEvents are movable not copyable wrappers around DIPU's events.
 * DIPUEvents are constructed lazily when first recorded.
 *
 * DIPU does not support IpcEventHandle currently.
 */
class DIPU_API DIPUEvent {
  c10::DeviceIndex index{0};
  deviceEvent_t event{nullptr};

 public:
  DIPUEvent() = default;
  DIPUEvent(const DIPUEvent&) = delete;
  DIPUEvent& operator=(const DIPUEvent&) = delete;

  DIPUEvent(DIPUEvent&& other) noexcept
      : index(other.index), event(other.event) {
    other.unsafe_reset();
  }

  DIPUEvent& operator=(DIPUEvent&& other) noexcept {
    index = other.index;
    event = other.event;
    other.unsafe_reset();
    return *this;
  }

  ~DIPUEvent() {
    if (initialized()) {
      DIPUGuard _(index);
      devproxy::destroyEvent(event);
    }
  }

  explicit operator deviceEvent_t() const noexcept { return rawevent(); }

  c10::optional<at::Device> device() const noexcept {
    if (initialized()) {
      return at::Device(dipu::DIPU_DEVICE_TYPE, index);
    }
    return {};
  }

  c10::DeviceIndex device_index() const noexcept { return index; }
  deviceEvent_t rawevent() const noexcept { return event; }

  bool query() const {
    if (initialized()) {
      DIPUGuard _(index);
      return devproxy::getEventStatus(event) == devapis::EventStatus::READY;
    }
    return true;
  }

  void record(const DIPUStream& stream = getCurrentDIPUStream()) {
    auto stream_index = stream.device_index();
    auto match = true;
    {
      DIPUGuard _(stream_index);
      if (not initialized()) {
        index = stream_index;
        devproxy::createEvent(&event);
        devproxy::recordEvent(event, stream.rawstream());
      } else if (index == stream_index) {
        devproxy::recordEvent(event, stream.rawstream());
      } else {
        match = false;
      }
    }

    TORCH_CHECK(match, "Event device ", index,
                " does not match recording stream's device ",
                stream.device_index(), ".");
  }

  void wait(const DIPUStream& stream) const {
    if (initialized()) {
      DIPUGuard _(stream.device_index());
      devproxy::streamWaitEvent(stream.rawstream(), event);
    }
  }

  float elapsed_time(const DIPUEvent& other) const {
    TORCH_CHECK(
        initialized() && other.initialized(),
        "Both events must be recorded before calculating elapsed time.");

    auto milliseconds = 0.F;
    devproxy::eventElapsedTime(&milliseconds, event, other.event);
    return milliseconds;
  }

  void synchronize() const {
    if (initialized()) {
      devproxy::waitEvent(event);
    }
  }

 private:
  bool initialized() const noexcept { return event != nullptr; }
  void unsafe_reset() noexcept { event = nullptr; }
};

}  // namespace dipu
