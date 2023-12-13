// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {

class DIPU_API DIPUStream {
 private:
  c10::Stream stream_;

 public:
  // Need more discussion to handle empty DIPUStream.
  explicit DIPUStream() : DIPUStream(-1, 0) {}

  explicit DIPUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == dipu::DIPU_DEVICE_TYPE);
  }

  explicit DIPUStream(devapis::deviceId_t device_id, c10::StreamId stream_id)
      : DIPUStream(c10::Stream(c10::Stream::UNSAFE,
                               c10::Device(dipu::DIPU_DEVICE_TYPE, device_id),
                               stream_id)) {}

  bool operator==(const DIPUStream& other) const noexcept {
    return static_cast<c10::Stream>(*this) == static_cast<c10::Stream>(other);
  }

  bool operator!=(const DIPUStream& other) const noexcept {
    return not operator==(other);
  }

  explicit operator c10::Stream() const { return stream_; }
  operator deviceStream_t() const { return rawstream(); }

  /// Get the device index that this stream is associated with.
  c10::DeviceIndex device_index() const { return stream_.device_index(); }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a device.
  c10::Device device() const {
    return {dipu::DIPU_DEVICE_TYPE, device_index()};
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
};

DIPU_API DIPUStream getDIPUStreamFromPool(c10::DeviceIndex device_index = -1);

DIPU_API DIPUStream getDefaultDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API DIPUStream getCurrentDIPUStream(c10::DeviceIndex device_index = -1);

DIPU_API void setCurrentDIPUStream(DIPUStream stream);

DIPU_API DIPUStream getStreamFromExternal(deviceStream_t ext_stream,
                                          c10::DeviceIndex device_index);

template <typename O>
inline O& operator<<(O& oss, const dipu::DIPUStream& stream) {
  oss << static_cast<c10::Stream>(stream);
}
}  // namespace dipu

template <>
struct std::hash<dipu::DIPUStream> {
  std::size_t operator()(dipu::DIPUStream const& s) const noexcept {
    return std::hash<c10::Stream>{}(static_cast<c10::Stream>(s));
  }
};
