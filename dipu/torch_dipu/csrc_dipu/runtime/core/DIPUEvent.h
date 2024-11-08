// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/Stream.h>

#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

#include "DIPUGuard.h"
#include "DIPUStream.h"

namespace dipu {
/*
 * DIPUEvent is a movable non-copyable wrapper around DIPU's events.
 * DIPUEvent is lazily constructed while "record".
 */
class DIPU_API DIPUEvent {
  deviceEvent_t event_{nullptr};
  c10::DeviceIndex device_index_{-1};
  c10::StreamId last_recorded_stream_id_{-1};
  bool use_pool_{true};

 public:
  DIPUEvent(const DIPUEvent&) = delete;
  DIPUEvent& operator=(const DIPUEvent&) = delete;

  constexpr DIPUEvent() noexcept = default;
  constexpr DIPUEvent(DIPUEvent&& other) noexcept
      : event_(other.event_),
        device_index_(other.device_index_),
        last_recorded_stream_id_(other.last_recorded_stream_id_),
        use_pool_(other.use_pool_) {
    other.unsafe_reset();
  }

  DIPUEvent& operator=(DIPUEvent&& other) noexcept(false /* release_event */) {
    if (this != std::addressof(other)) {
      release_event();
      event_ = other.event_;
      device_index_ = other.device_index_;
      last_recorded_stream_id_ = other.last_recorded_stream_id_;
      use_pool_ = other.use_pool_;
      other.unsafe_reset();
    }
    return *this;
  }

  ~DIPUEvent() { release_event(); }

  deviceEvent_t device_event() const noexcept { return event_; }

  c10::DeviceIndex device_index() const noexcept { return device_index_; }

  // Return the ID of the most recent recoreded stream.
  //
  // Note: no event-stream relation guaranteed. It means the fetched stream ID
  // may be invalid (-1) or out-dated.
  c10::StreamId last_recorded_stream_id() const noexcept {
    return last_recorded_stream_id_;
  }

  c10::optional<at::Device> device() const {
    if (initialized()) {
      return at::Device(dipu::DIPU_DEVICE_TYPE, device_index_);
    }
    return {};
  }

  bool query() const {
    if (!initialized()) {  // unlikely
      return true;
    }

    DIPUGuard guard(device_index_);
    return devproxy::getEventStatus(event_) == devapis::EventStatus::READY;
  }

  void record() { record(getCurrentDIPUStream()); }

  void record(const DIPUStream& stream, bool use_pool = true) {
    if (!initialized()) {
      use_pool_ = use_pool;
      create_event(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ",
                device_index_, " does not match recording stream's device ",
                stream.device_index(), ".");

    last_recorded_stream_id_ = stream.id();
    DIPUGuard guard(device_index_);
    devproxy::recordEvent(event_, stream.rawstream());
  }

  void wait(const DIPUStream& stream) {
    if (initialized()) {  // likely
      DIPUGuard guard(stream.device_index());
      devproxy::streamWaitEvent(stream.rawstream(), event_);
    }
  }

  float elapsed_time(const DIPUEvent& other) const {
    TORCH_CHECK(
        initialized() && other.initialized(),
        "Both events must be recorded before calculating elapsed time.");

    auto time_ms = 0.F;
    devproxy::eventElapsedTime(&time_ms, event_, other.event_);
    return time_ms;
  }

  void synchronize() const {
    if (initialized()) {  // likely
      devproxy::waitEvent(event_);
    }
  }

  // aclrtEvent do not support Less than operator until now
  // dipu do not support IpcEventHandle until now

 private:
  bool initialized() const noexcept { return event_ != nullptr; }

  void unsafe_reset() noexcept { event_ = nullptr; }

  void create_event(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    DIPUGuard guard(device_index_);
    if(use_pool_) {
      devproxy::createEvent(&event_);
    } else {
      devapis::createEvent(&event_);
    }
  }

  void release_event() {
    if (initialized()) {
      DIPUGuard guard(device_index_);
      if(use_pool_) {
        devproxy::destroyEvent(event_);
      } else {
        devapis::destroyEvent(event_);
      }
      event_ = nullptr;
      use_pool_ = true;
    }
  }
};

}  // namespace dipu
