// Copyright (c) 2023, DeepLink.
#include "DIPUStream.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include <c10/util/Exception.h>

#include "DIPUGuard.h"

namespace dipu {
namespace {

enum class StreamIdType : uint8_t {
  DEFAULT = 0,
  POOL = 1,
};

std::string to_string(StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      return "DEFAULT";
    case StreamIdType::POOL:
      return "POOL";
    default:
      return std::to_string(static_cast<uint8_t>(s));
  }
}

// follow old pytorch cuda, seems new version use an opposite strategy.
constexpr int kStreamsPerPoolBits = 3;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;

c10::StreamId makeC10StreamId(StreamIdType sType, size_t id) {
  return (static_cast<uint32_t>(static_cast<c10::StreamId>(sType)
                                << kStreamsPerPoolBits)) |
         static_cast<c10::StreamId>(id);
}

// manage per-device streams
struct DIPUStreamDevice {
 private:
  // Default streams
  std::once_flag pool_flag;
  std::once_flag default_flag;
  devapis::deviceId_t devidx_;
  // seems pytorch 2.0 giveup default stream and enable cuda per_thread stream
  // feature at compile time. it cannot be applied to othe device.
  deviceStream_t default_stream{};
  std::atomic<uint32_t> next_pool_pos{};
  std::array<deviceStream_t, kStreamsPerPool> pool_streams{};

  inline uint32_t getNextPoolIdx() {
    auto raw_idx = next_pool_pos++;
    return raw_idx % kStreamsPerPool;
  }

  static StreamIdType getStreamIdType(c10::StreamId s) {
    return static_cast<StreamIdType>(static_cast<uint32_t>(s) >>
                                     kStreamsPerPoolBits);
  }

  static size_t getStreamIdIndex(c10::StreamId s) {
    return static_cast<size_t>(static_cast<uint32_t>(s) &
                               ((1 << kStreamsPerPoolBits) - 1));
  }

  void _doInitPool() {
    DIPUGuard device_guard{devidx_};
    for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
      auto& raw_device_stream = pool_streams[i];
      devproxy::createStream(&raw_device_stream);
    }
  }

  void _doInitDeivce() {
    auto cur_device = devproxy::current_device();
    devproxy::setDevice(devidx_);
    devproxy::createStream(&default_stream);
    devproxy::setDevice(cur_device);
  }

 public:
  explicit DIPUStreamDevice(devapis::deviceId_t device_id)
      : devidx_(device_id) {}

  DIPUStream getDIPUStreamfromPool() {
    const auto idx = getNextPoolIdx();
    return DIPUStream(devidx_, makeC10StreamId(StreamIdType::POOL, idx));
  }

  DIPUStream getDefaultDIPUStream() const {
    return DIPUStream(devidx_, makeC10StreamId(StreamIdType::DEFAULT, 0));
  }

  // c10:StreamId -> rawStream saved in DIPUStreamDevice.
  deviceStream_t obtainRawStream(c10::StreamId stream_id) {
    StreamIdType st = getStreamIdType(stream_id);
    size_t sidx = getStreamIdIndex(stream_id);
    switch (st) {
      case StreamIdType::DEFAULT:
        AT_ASSERTM(
            sidx == 0, "Unrecognized stream ", stream_id,
            " (I think this should be the default stream, but I got a non-zero "
            "index ",
            sidx, ").",
            " Did you manufacture the StreamId yourself?  Don't do that; use "
            "the",
            " official API like c10::cuda::getStreamFromPool() to get a new "
            "stream.");
        return default_stream;
      case StreamIdType::POOL:
        return pool_streams[sidx];
      default:
        // TODO(assert): AT_ERROR is deprecated.
        AT_ERROR("Invalid stream", stream_id, " (type=", to_string(st), ")");
    }
  }
  void initPool() {
    std::call_once(pool_flag, &DIPUStreamDevice::_doInitPool, this);
  }
  void initDevice() {
    std::call_once(default_flag, &DIPUStreamDevice::_doInitDeivce, this);
  }
};

auto StreamDeviceList()
    -> std::vector<std::unique_ptr<DIPUStreamDevice>> const& {
  auto make_list = [] {
    auto number_of_device = devproxy::getDeviceCount();
    auto list = std::vector<std::unique_ptr<DIPUStreamDevice>>();
    list.reserve(number_of_device);
    for (auto i = 0; i < number_of_device; ++i) {
      list.emplace_back(std::make_unique<DIPUStreamDevice>(i));
    }
    return list;
  };

  auto static device_list = make_list();
  return device_list;
}

auto LocalStreams() -> std::vector<c10::StreamId>& {
  auto static thread_local streams = std::vector<c10::StreamId>(
      StreamDeviceList().size(), makeC10StreamId(StreamIdType::DEFAULT, 0));

  return streams;
}

c10::DeviceIndex setupDevice(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = devproxy::current_device();
  }

  auto& device_list = StreamDeviceList();
  auto number_of_device = static_cast<int>(device_list.size());
  // TODO(assert): AT_ASSERT is deprecated and TORCH_CHECK contains their own
  // help message. We need our version.
  AT_ASSERT(0 <= device_index && device_index < number_of_device);
  device_list[device_index]->initDevice();

  return device_index;
}

}  // end anonymous namespace

// api
deviceStream_t DIPUStream::rawstream() const {
  return StreamDeviceList()[stream_.device_index()]->obtainRawStream(
      stream_.id());
}

DIPUStream getDIPUStreamFromPool(c10::DeviceIndex device_index) {
  device_index = setupDevice(device_index);
  // Initializes the stream pools (once)
  auto& device = *StreamDeviceList()[device_index];
  device.initPool();
  return device.getDIPUStreamfromPool();
}

DIPUStream getDefaultDIPUStream(c10::DeviceIndex device_index) {
  device_index = setupDevice(device_index);
  return StreamDeviceList()[device_index]->getDefaultDIPUStream();
}

DIPUStream getCurrentDIPUStream(c10::DeviceIndex device_index) {
  device_index = setupDevice(device_index);
  return DIPUStream(device_index, LocalStreams()[device_index]);
}

// copy from pytorch, not verify
DIPUStream getStreamFromExternal(deviceStream_t ext_stream,
                                 c10::DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return DIPUStream(device_index, reinterpret_cast<int64_t>(ext_stream));
}

void setCurrentDIPUStream(DIPUStream stream) {
  auto device_index = stream.device_index();
  // TODO(assert): assert(setupDevice(device_index) == device_index)
  setupDevice(device_index);
  LocalStreams()[device_index] = static_cast<c10::Stream>(stream).id();
}

}  // namespace dipu
