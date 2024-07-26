// Copyright (c) 2023, DeepLink.
#include "deviceproxy.h"

#include <atomic>
#include <sys/sysinfo.h>

#include <c10/util/Exception.h>

#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/runtime/core/allocator/allocator_metrics.h"
#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/device/deviceapis.h"
#include "csrc_dipu/utils/Log.h"
#include "csrc_dipu/utils/env.hpp"

namespace dipu {
namespace devproxy {

void initializeVendor() {
  if (devapis::initializeVendor) {
    devapis::initializeVendor();
  }
}

void finalizeVendor() {
  if (devapis::finalizeVendor) {
    devapis::finalizeVendor();
  }
}

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  return devapis::getDeviceProperties(device_index);
}

DIPUDeviceStatus getDeviceStatus(int32_t device_index) {
  if (devapis::getDeviceStatus) {
    return devapis::getDeviceStatus(device_index);
  }
  return {};
}

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local deviceId_t currentDevice = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
deviceId_t lastDevice = -1;
}  // namespace

deviceId_t current_device() {
  if (currentDevice < 0) {
    // Because on some software stacks, getDevice is not allowed to be called
    // before setDevice is called. So we do this
    if (lastDevice > 0) {
      setDevice(lastDevice);
      TORCH_WARN_ONCE(
          "Since device ", static_cast<int>(lastDevice),
          " has been used before, and there is no indication of which device "
          "to use in the current thread, we will continue to use device ",
          lastDevice, " instead of device 0.");
    } else {
      setDevice(0);
    }
  }
  return currentDevice;
}

void setCpuAffinity(const int device) {
  static int affinity = get_env_or_default("DIPU_CPU_AFFINITY", 0);
  if (affinity < 0) {
    return;
  }
  const int num_of_processors = get_nprocs();
  const int device_count = getDeviceCount();
  const int block_size =
      (affinity == 0) ? ((num_of_processors + device_count - 1) / device_count)
                      : affinity;
  const int start_cpu_core = device * block_size;
  const int end_cpu_core =
      std::min((device + 1) * block_size, num_of_processors);
  cpu_set_t mask;
  CPU_ZERO(&mask);
  DIPU_LOG_INFO << "DIPU_CPU_AFFINITY: Bind device " << device
                << " with cpu cores: [" << start_cpu_core << "," << end_cpu_core
                << "), the number of processors:" << num_of_processors
                << std::endl;
  for (int i = start_cpu_core; i < end_cpu_core; i++) {
    CPU_SET(i, &mask);
  }
  pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

// set current device given device according to id
void setDevice(deviceId_t devId) {
  static const int kDeviceCount = getDeviceCount();
  TORCH_CHECK(devId < kDeviceCount && devId >= 0,
              "invalid device id: ", static_cast<int>(devId),
              " , device count:", kDeviceCount)
  if (currentDevice != devId) {
    devapis::setDevice(devId);
    setCpuAffinity(devId);
    if (currentDevice < 0) {
      if (lastDevice < 0) {
        // The main purpose is that if a device has been used before, but which
        // device is not specified in the subsequent new thread, then the
        // previous device will continue to be used. The lastDevice variable is
        // an artificial assumption.In fact, it is almost impossible for
        // two threads to setDevice to different devices at the same time, and
        // even if it does, which one should be choosed?
        lastDevice = devId;
      }
    }
    currentDevice = devId;
  }
}

void resetDevice(deviceId_t devId) {
  currentDevice = devId;
  if (lastDevice < 0) {
    lastDevice = devId;
  }
  return devapis::resetDevice(devId);
}

void syncDevice() { return devapis::syncDevice(); }

// check last launch succ or not, throw if fail
void checkLastError() { return devapis::checkLastError(); }

int getDeviceCount() {
  static int device_count = devapis::getDeviceCount();
  return device_count;
}

void getDriverVersion(int* version) {
  return devapis::getDriverVersion(version);
}

void getRuntimeVersion(int* version) {
  return devapis::getRuntimeVersion(version);
}

void createStream(deviceStream_t* stream, bool prior) {
  return devapis::createStream(stream, prior);
}

void destroyStream(deviceStream_t stream) {
  return devapis::destroyStream(stream);
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  return devapis::destroyStream(stream, devId);
}

void releaseStream() { return devapis::releaseStream(); }

void syncStream(deviceStream_t stream) { return devapis::syncStream(stream); }

bool streamNotNull(deviceStream_t stream) {
  return devapis::streamNotNull(stream);
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  return devapis::streamWaitEvent(stream, event);
}

// same as query last event status in stream.(every op has a event)
bool isStreamEmpty(deviceStream_t stream) {
  return devapis::isStreamEmpty(stream);
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) {
  auto index = current_device();
  return event_pool_acquire(index, *event);
}

void destroyEvent(deviceEvent_t event) {
  auto index = current_device();
  return event_pool_release(index, event);
}

void waitEvent(deviceEvent_t event) { return devapis::waitEvent(event); }

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  return devapis::recordEvent(event, stream);
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end) {
  return devapis::eventElapsedTime(time, start, end);
}

EventStatus getEventStatus(deviceEvent_t event) {
  return devapis::getEventStatus(event);
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
  devapis::mallocHost(p, nbytes);
  GlobalAllocatorGroupMetrics::host_allocator_metrics()[current_device()]
      .allocate(p ? *p : nullptr, nbytes);
}

void freeHost(void* p) {
  GlobalAllocatorGroupMetrics::host_allocator_metrics()[current_device()]
      .deallocate(p);
  return devapis::freeHost(p);
}

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  auto code = devapis::mallocDevice(p, nbytes, throwExcepion);
  GlobalAllocatorGroupMetrics::device_allocator_metrics()[current_device()]
      .allocate(p ? *p : nullptr, nbytes);
  return code;
}

void freeDevice(void* p) {
  GlobalAllocatorGroupMetrics::device_allocator_metrics()[current_device()]
      .deallocate(p);
  return devapis::freeDevice(p);
}

bool isPinnedPtr(const void* p) { return devapis::isPinnedPtr(p); }

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  return devapis::memSetAsync(stream, ptr, val, size);
}

// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  if ((dstDevId == srcDevId && dst == src) || nbytes == 0) {
    return;
  }
  return devapis::memCopyD2D(nbytes, dstDevId, dst, srcDevId, src);
}

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, /*deviceId_t dstDevId,*/ void* dst,
                /*Host srcDev,*/ const void* src) {
  if (nbytes <= 0) {
    return;
  }
  return devapis::memCopyH2D(nbytes, dst, src);
}

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, /*Host dstDev,*/ void* dst,
                /*deviceId_t srcDevId,*/ const void* src) {
  if (nbytes <= 0) {
    return;
  }
  return devapis::memCopyD2H(nbytes, dst, src);
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  if ((dstDevId == srcDevId && dst == src) || nbytes == 0) {
    return;
  }
  return devapis::memCopyD2DAsync(stream, nbytes, dstDevId, dst, srcDevId, src);
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes,
                     /*deviceId_t dstDevId,*/ void* dst,
                     /*Host srcDev,*/ const void* src) {
  return devapis::memCopyH2DAsync(stream, nbytes, dst, src);
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes,
                     /*Host dstDev,*/ void* dst,
                     /*deviceId_t srcDevId,*/ const void* src) {
  return devapis::memCopyD2HAsync(stream, nbytes, dst, src);
}

}  // end namespace devproxy
}  // end namespace dipu
