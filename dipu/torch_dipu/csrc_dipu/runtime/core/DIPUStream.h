// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>
#include <mutex>

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {

class DIPU_API DIPUStream {
 public:
  enum Unchecked { UNCHECKED };
  explicit DIPUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == dipu::DIPU_DEVICE_TYPE);
  }

  explicit DIPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  explicit DIPUStream(devapis::deviceId_t devidx, c10::StreamId stream_id)
      : DIPUStream(Unchecked::UNCHECKED,
                   c10::Stream(c10::Stream::UNSAFE,
                               c10::Device(dipu::DIPU_DEVICE_TYPE, devidx),
                               stream_id)) {}

  ~DIPUStream() {}

  bool operator==(const DIPUStream &other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const DIPUStream &other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to pytorch Stream.
  operator c10::Stream() const { return unwrap(); }

  operator deviceStream_t() const { return rawstream(); }

  /// Get the device index that this stream is associated with.
  c10::DeviceIndex device_index() const { return stream_.device_index(); }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a device.
  c10::Device device() const {
    return c10::Device(dipu::DIPU_DEVICE_TYPE, device_index());
  }

  c10::StreamId id() const { return stream_.id(); }

  void synchronize() const {
    c10::DeviceGuard guard{stream_.device()};
    devproxy::syncStream(rawstream());
  }

  bool isStreamEmpty() const {
    c10::DeviceGuard guard{device()};
    return devproxy::isStreamEmpty(rawstream());
  }

  /// Explicit conversion to rtStream_t.
  deviceStream_t rawstream() const;

  /// Explicit conversion to Stream.
  c10::Stream unwrap() const { return stream_; }

  c10::StreamData3 pack3() const noexcept { return stream_.pack3(); }

  static DIPUStream unpack3(c10::StreamId stream_id,
                            c10::DeviceIndex device_index,
                            c10::DeviceType device_type) {
    TORCH_CHECK(device_type == dipu::DIPU_DEVICE_TYPE);
    return DIPUStream(device_index, stream_id);
  }

 private:
  c10::Stream stream_;
};

DIPU_API DIPUStream getDIPUStreamFromPool(c10::DeviceIndex device = -1);

DIPU_API DIPUStream getDefaultDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API DIPUStream getCurrentDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API void setCurrentDIPUStream(DIPUStream stream);

DIPU_API DIPUStream getStreamFromExternal(deviceStream_t ext_stream,
                                          c10::DeviceIndex device_index);

std::ostream &operator<<(std::ostream &stream, const DIPUStream &s);
}  // namespace dipu

namespace std {
template <>
struct hash<dipu::DIPUStream> {
  size_t operator()(dipu::DIPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
}  // namespace std
