#pragma once

#include <cstdint>
#include <mutex>

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/SmallVector.h>
#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

class DIPU_API DIPUStream {
public:
  enum Unchecked { UNCHECKED };
  explicit DIPUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == dipu::DIPU_DEVICE_TYPE);
  }

  explicit DIPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  ~DIPUStream(){}

  bool operator==(const DIPUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const DIPUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to pytorch Stream.
  operator c10::Stream() const {
    return unwrap();
  }

  operator deviceStream_t() const {
    return rawstream();
  }

  /// Get the device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a device.
  c10::Device device() const {
    return c10::Device(dipu::DIPU_DEVICE_TYPE, device_index());
  }

  c10::StreamId id() const {
    return stream_.id();
  }

  void synchronize() const {
    c10::DeviceGuard guard{stream_.device()};
    devapis::syncStream(rawstream());
  }

  bool isStreamEmpty() const {
    c10::DeviceGuard guard{device()};
    return devapis::isStreamEmpty(rawstream());
  }

  /// Explicit conversion to rtStream_t.
  deviceStream_t rawstream() const;

  /// Explicit conversion to Stream.
  c10::Stream unwrap() const {
    return stream_;
  }

  uint64_t pack() const noexcept {
    return stream_.id();
  }

  struct c10::StreamData3 pack3() const {
    return {id(), device_index(), dipu::DIPU_DEVICE_TYPE};
  }

  static DIPUStream unpack3(
      c10::StreamId stream_id,
      c10::DeviceIndex device_index,
      c10::DeviceType device_type) {
    return DIPUStream(c10::Stream::unpack3(stream_id, device_index, device_type));
  }

private:
  c10::Stream stream_;
};

DIPU_API DIPUStream getDIPUStreamFromPool(c10::DeviceIndex device = -1);

DIPU_API DIPUStream getDefaultDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API DIPUStream getCurrentDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API void dipuSynchronizeDevice();

DIPU_API void setCurrentDIPUStream(DIPUStream stream);

std::ostream& operator<<(std::ostream& stream, const DIPUStream& s);
} // namespace dipu

namespace std {
template <>
struct hash<dipu::DIPUStream> {
  size_t operator()(dipu::DIPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
