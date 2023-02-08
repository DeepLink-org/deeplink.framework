#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#include <c10/util/Exception.h>

#include "DIPUGuard.h"
#include "DIPUStream.h"

namespace torch_dipu {
namespace {

struct DIPUStreamInternal {
  DIPUStreamInternal() = default;
  ~DIPUStreamInternal() {}
  c10::DeviceIndex device_index = -1;
  int32_t stream_id = -1;
  deviceStream_t stream = nullptr;

  void initStatus(int index, int id = -1) {
    if (stream) {
      return;
    }
    stream_id = id;
    device_index = index;
    auto cur_device = devapis::current_device();
    devapis::setDevice(index);
    devapis::createStream(&stream);
    devapis::setDevice(cur_device);
  }
};

static constexpr int kStreamsPerPoolBits = 3;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;

// Global stream state and constants
static c10::DeviceIndex num_dipus = -1;

// Default streams
static std::once_flag init_flag;
static DIPUStreamInternal default_streams[C10_COMPILE_TIME_MAX_DIPUS];

static std::once_flag device_flags[C10_COMPILE_TIME_MAX_DIPUS];
static std::atomic<uint32_t> dipu_counters[C10_COMPILE_TIME_MAX_DIPUS];

static std::array<DIPUStreamInternal, kStreamsPerPool>
    pool_streams[C10_COMPILE_TIME_MAX_DIPUS];

enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  POOL = 0x1,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamIdType::POOL:
      stream << "POOL";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

static inline StreamIdType getStreamIdType(c10::StreamId s) {
  return static_cast<StreamIdType>((uint32_t)s >> kStreamsPerPoolBits);
}

static inline size_t getStreamIdIndex(c10::StreamId s) {
  return static_cast<size_t>((uint32_t)s & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType sType, size_t id) {
  return ((uint32_t)static_cast<c10::StreamId>(sType) << kStreamsPerPoolBits) |
      static_cast<c10::StreamId>(id);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) &&
      std::less<const T*>()(ptr, arr.data() + arr.size());
}

static thread_local DIPUStreamInternal** thread_local_streams = nullptr;

static void initGlobalStreamState() {
  num_dipus = devapis::getDeviceCount();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  AT_ASSERTM(
      num_dipus <= C10_COMPILE_TIME_MAX_DIPUS,
      "Number of DIPU devices on the machine is larger than the compiled "
      "max number of dipus expected (",
      C10_COMPILE_TIME_MAX_DIPUS,
      "). Increase that and recompile.");

  int device_id = devapis::current_device();
  if (device_id == -1) {
    DIPU_LOGE("Device has not been set");
  }
  // Initializes default streams
  default_streams[device_id].device_index = device_id;
  dipu_counters[device_id] = 0;
  auto& deviceStream = default_streams[device_id];
  deviceStream.initStatus(device_id);
}

static void initPoolStreamState(c10::DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  DIPUGuard device_guard{device_index};
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& streamInternal = pool_streams[device_index][i];
    streamInternal.device_index = device_index;
    devapis::createStream(&streamInternal.stream);
  }
}

static void initDIPUStreamsOnce() {
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (thread_local_streams) {
    return;
  }
  // Inits thread local streams to default streams
  thread_local_streams =
      (DIPUStreamInternal**)malloc(num_dipus * sizeof(DIPUStreamInternal*));
  if (thread_local_streams == NULL){
    DIPU_LOGE("thread_local_streams malloc failed.");
    return;
  }
  for (auto i = 0; i < num_dipus; ++i) {
    thread_local_streams[i] = &default_streams[i];
  }
}

static inline void check_dipu(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = devapis::current_device();
  }
  AT_ASSERT(device_index >= 0 && device_index < num_dipus);
}

static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

static c10::StreamId StreamInternal2StreamId(const DIPUStreamInternal* ptr) {
  c10::DeviceIndex device_index = ptr->device_index;
  if (ptr == &default_streams[device_index]) {
    return makeStreamId(StreamIdType::DEFAULT, 0);
  }
  if (pointer_within<DIPUStreamInternal>(ptr, pool_streams[device_index])) {
    return makeStreamId(
        StreamIdType::POOL, ptr - pool_streams[device_index].data());
  }
  AT_ASSERTM(
      0,
      "Could not compute stream ID for ",
      ptr,
      " on device ",
      device_index,
      " (something has gone horribly wrong!)");
}

static DIPUStreamInternal* Stream2SteamInternal(DIPUStream s) {
  c10::DeviceIndex device_index = s.device_index();
  StreamIdType st = getStreamIdType(s.unwrap().id());
  size_t sidx = getStreamIdIndex(s.unwrap().id());
  switch (st) {
    case StreamIdType::DEFAULT:
      AT_ASSERTM(
          sidx == 0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I think this should be the default stream, but I got a non-zero index ",
          sidx,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like c10::cuda::getStreamFromPool() to get a new stream.");
      return &default_streams[device_index];
    case StreamIdType::POOL:
      return &pool_streams[device_index][sidx];
    default:
      AT_ASSERTM(
          0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

static DIPUStream StreamInternal2Stream(const DIPUStreamInternal* ptr) {
  return DIPUStream(
      DIPUStream::UNCHECKED,
      c10::Stream(
          c10::Stream::UNSAFE,
          c10::Device(torch_dipu::DIPU_DEVICE_TYPE, ptr->device_index),
          StreamInternal2StreamId(ptr)));
}
} // namespace

deviceStream_t DIPUStream::rawstream() const {
  auto cur_ptr = Stream2SteamInternal(*this);
  AT_ASSERT(cur_ptr);
  return cur_ptr->stream;
}

DIPUStream getDIPUStreamFromPool(c10::DeviceIndex device_index) {
  initDIPUStreamsOnce();
  check_dipu(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initPoolStreamState, device_index);

  const auto idx = get_idx(dipu_counters[device_index]);
  return StreamInternal2Stream(&pool_streams[device_index][idx]);
}

DIPUStream getDefaultDIPUStream(c10::DeviceIndex device_index) {
  initDIPUStreamsOnce();
  if (device_index == -1) {
    device_index = devapis::current_device();
  }
  return StreamInternal2Stream(&default_streams[device_index]);
}

static const DIPUStreamInternal* _getCurrentDIPUStream(c10::DeviceIndex device_index) {
  initDIPUStreamsOnce();
  if (device_index == -1) {
    device_index = devapis::current_device();
  }
  check_dipu(device_index);
  return thread_local_streams[device_index];
}

DIPUStream getCurrentDIPUStream(c10::DeviceIndex device_index) {
  return StreamInternal2Stream(_getCurrentDIPUStream(device_index));
}

void dipuSynchronizeDevice() {
  devapis::syncDevice();
}

void setCurrentDIPUStream(DIPUStream stream) {
  initDIPUStreamsOnce();
  auto ptr = Stream2SteamInternal(stream);
  AT_ASSERT(ptr);
  thread_local_streams[ptr->device_index] = ptr;
}

std::ostream& operator<<(std::ostream& os, const DIPUStream& stream) {
  return os << stream.unwrap();
}

} // namespace torch_dipu
