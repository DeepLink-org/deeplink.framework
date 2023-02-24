#pragma once

#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <cstddef>

#include "./guardimpl/DIPUGuardImpl.h"


namespace dipu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for DIPU.  It accepts
/// integer indices (interpreting them as DIPU devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// DIPUSetDevice/DIPUGetDevice calls); however, it can only be used
/// from code that links against DIPU directly.
class DIPUGuard {
public:
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit DIPUGuard() = delete;

  /// Set the current DIPU device to the passed device index.
  explicit DIPUGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current DIPU device to the passed device.  Errors if the passed
  /// device is not a DIPU device.
  explicit DIPUGuard(c10::Device device) : guard_(device) {}

  // Copy is not allowed
  DIPUGuard(const DIPUGuard&) = delete;
  DIPUGuard& operator=(const DIPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  DIPUGuard(DIPUGuard&& other) = delete;
  DIPUGuard& operator=(DIPUGuard&& other) = delete;

  /// Sets the DIPU device to the given device.  Errors if the given device
  /// is not a DIPU device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the DIPU device to the given device.  Errors if the given device
  /// is not a DIPU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the DIPU device to the given device index.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  c10::Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  c10::Device current_device() const {
    return guard_.current_device();
  }

private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<dipu::DIPUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for DIPU.  See
/// DIPUGuard for when you can use this.
struct OptionalDIPUGuard {
  /// Create an uninitialized OptionalDIPUGuard.
  explicit OptionalDIPUGuard() : guard_() {}

  /// Set the current DIPU device to the passed Device, if it is not nullopt.
  explicit OptionalDIPUGuard(c10::optional<c10::Device> device_opt) : guard_(device_opt) {}

  /// Set the current DIPU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalDIPUGuard(c10::optional<c10::DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalDIPUGuard(const OptionalDIPUGuard&) = delete;
  OptionalDIPUGuard& operator=(const OptionalDIPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalDIPUGuard(OptionalDIPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalDIPUGuard& operator=(OptionalDIPUGuard&& other) = delete;

  /// Sets the DIPU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a DIPU
  /// device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the DIPU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a DIPU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the DIPU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  c10::optional<c10::Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  c10::optional<c10::Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original DIPU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

private:
  c10::impl::InlineOptionalDeviceGuard<dipu::DIPUGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for DIPU.  See DIPUGuard
/// for when you can use this.
struct DIPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit DIPUStreamGuard() = delete;

  /// Set the current DIPU device to the device associated with the passed
  /// stream, and set the current DIPU stream on that device to the passed
  /// stream. Errors if the Stream is not a DIPU stream.
  explicit DIPUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  DIPUStreamGuard(const DIPUStreamGuard&) = delete;
  DIPUStreamGuard& operator=(const DIPUStreamGuard&) = delete;

  /// Move is disallowed, as DIPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  DIPUStreamGuard(DIPUStreamGuard&& other) = delete;
  DIPUStreamGuard& operator=(DIPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a DIPU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on DIPU, use DIPUMultiStreamGuard instead.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the DIPU stream that was set at the time the guard was constructed.
  DIPUStream original_stream() const {
    return DIPUStream(DIPUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent DIPU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  DIPUStream current_stream() const {
    return DIPUStream(DIPUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent DIPU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the DIPU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const {
    return guard_.original_device();
  }

private:
  c10::impl::InlineStreamGuard<dipu::DIPUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for DIPU.  See DIPUGuard
/// for when you can use this.
struct OptionalDIPUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalDIPUStreamGuard() : guard_() {}

  /// Set the current DIPU device to the device associated with the passed
  /// stream, and set the current DIPU stream on that device to the passed
  /// stream. Errors if the Stream is not a DIPU stream.
  explicit OptionalDIPUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalDIPUStreamGuard(c10::optional<c10::Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalDIPUStreamGuard(const OptionalDIPUStreamGuard&) = delete;
  OptionalDIPUStreamGuard& operator=(const OptionalDIPUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalDIPUStreamGuard(OptionalDIPUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalDIPUStreamGuard& operator=(OptionalDIPUStreamGuard&& other) = delete;

  /// Resets the currently set DIPU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the DIPU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  c10::optional<DIPUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return c10::make_optional(DIPUStream(DIPUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Returns the most recent DIPU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  c10::optional<DIPUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return c10::make_optional(DIPUStream(DIPUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Restore the original DIPU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

private:
  c10::impl::InlineOptionalStreamGuard<dipu::DIPUGuardImpl> guard_;
};

} // namespace dipu