// Copyright (c) 2023, DeepLink.
#include <tops_runtime.h>
#include <tops_runtime_api.h>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

namespace devapis {

using tops_deviceId = int;
// =====================
//  Device class related
// =====================

void initializeVendor() {}

void finalizeVendor() {}

deviceId_t current_device() {
  tops_deviceId devId_;
  DIPU_CALLTOPSRT(::topsGetDevice(&devId_))
  return static_cast<deviceId_t>(devId_);
}

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  ::topsDeviceProp_t device_prop;
  DIPU_CALLTOPSRT(topsGetDeviceProperties(&device_prop, device_index))

  DIPUDeviceProperties prop;
  prop.name = device_prop.name;
  prop.totalGlobalMem = device_prop.totalGlobalMem;
  prop.major = device_prop.major;
  prop.minor = device_prop.minor;
  prop.multiProcessorCount = device_prop.multiProcessorCount;
  return prop;
}

// in tops_runtime_api.h
// set current device given device according to id
void setDevice(deviceId_t devId) {
  tops_deviceId devId_ = static_cast<deviceId_t>(devId);
  DIPU_CALLTOPSRT(::topsSetDevice(devId_))
}

void resetDevice(deviceId_t devId) { DIPU_CALLTOPSRT(::topsDeviceReset()) }

void syncDevice() { DIPU_CALLTOPSRT(::topsDeviceSynchronize()) }

// check last launch succ or not, throw if fail
void checkLastError() { DIPU_CALLTOPSRT(::topsGetLastError()) }

int getDeviceCount() {
  int num = -1;
  DIPU_CALLTOPSRT(::topsGetDeviceCount(reinterpret_cast<int*>(&num)))
  return num;
}

void getDriverVersion(int* version) {
  DIPU_CALLTOPSRT(::topsDriverGetVersion(version))
}

void getRuntimeVersion(int* version) {
  DIPU_CALLTOPSRT(::topsRuntimeGetVersion(version))
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
  if (prior) {
    DIPU_LOGW(
        "topsStreamCreateWithPriority is not ready, replace with "
        "topsStreamCreate");
    DIPU_CALLTOPSRT(::topsStreamCreate(stream))
    // DIPU_CALLTOPSRT(::topsStreamCreateWithPriority(stream, topsStreamDefault,
    // -1))
  } else {
    DIPU_CALLTOPSRT(::topsStreamCreate(stream))
  }
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLTOPSRT(::topsStreamDestroy(stream))
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  setDevice(devId);
  destroyStream(stream);
}

void releaseStream() { return; }

bool streamNotNull(deviceStream_t stream) {
  return (stream != nullptr && stream != topsStreamPerThread);
}

void syncStream(deviceStream_t stream) {
  DIPU_CALLTOPSRT(::topsStreamSynchronize(stream));
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  DIPU_CALLTOPSRT(::topsStreamWaitEvent(stream, event, 0))
}

bool isStreamEmpty(deviceStream_t stream) {
  auto err = topsStreamQuery(stream);
  if (err == ::topsSuccess) {
    return true;
  }
  return false;
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) {
  DIPU_CALLTOPSRT(::topsEventCreateWithFlags(event, topsEventDisableTiming))
}

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLTOPSRT(::topsEventDestroy(event))
}

void waitEvent(deviceEvent_t event) {
  DIPU_CALLTOPSRT(::topsEventSynchronize(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  DIPU_CALLTOPSRT(::topsEventRecord(event, stream))
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end){
    DIPU_CALLTOPSRT(topsEventElapsedTime(time, start, end))}

EventStatus getEventStatus(deviceEvent_t event) {
  ::topsError_t ret = ::topsEventQuery(event);
  if (ret == ::topsSuccess) {
    return devapis::EventStatus::READY;
  } else if (ret == ::topsErrorNotReady) {
    ::topsGetLastError(); /* reset internal error state*/
    return devapis::EventStatus::PENDING;
  } else {
    throw std::runtime_error("dipu device error, ret code:" +
                             std::to_string(ret));
  }
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
  if (nbytes != 0) DIPU_CALLTOPSRT(::topsHostMalloc(p, nbytes))
}

void freeHost(void* p) {
  if (!p) DIPU_CALLTOPSRT(::topsHostFree(p))
}

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  ::topsError_t r = ::topsMalloc(p, nbytes);
  if (r != ::topsSuccess) {
    if (throwExcepion) {
      ::topsGetLastError(); /* reset internal error state*/
      throw std::runtime_error("alloc failed in dipu");
    } else if (r == ::topsErrorMemoryAllocation) {
      return OpStatus::ERR_NOMEM;
    } else {
      return OpStatus::ERR_UNKNOWN;
    }
  }
  return OpStatus::SUCCESS;
}

void freeDevice(void* p) { DIPU_CALLTOPSRT(::topsFree(p)) }

bool isPinnedPtr(const void* p) {
  ::topsPointerAttribute_t attr;
  DIPU_CALLTOPSRT(::topsPointerGetAttributes(&attr, p))
  return attr.memoryType == topsMemoryTypeHost;
}

void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  DIPU_CALLTOPSRT(::topsMemsetAsync(ptr, val, size, stream))
}

void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  if (dstDevId == srcDevId) {
    DIPU_CALLTOPSRT(::topsMemcpy(dst, src, nbytes, ::topsMemcpyDeviceToDevice))
  } else {
    DIPU_CALLTOPSRT(::topsMemcpyPeer(dst, dstDevId, src, srcDevId, nbytes))
  }
}

// (synchronous) copy from host to a tops device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLTOPSRT(::topsMemcpy(dst, src, nbytes, ::topsMemcpyHostToDevice))
}

// (synchronous) copy from a tops device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLTOPSRT(::topsMemcpy(dst, src, nbytes, ::topsMemcpyDeviceToHost))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  if (dstDevId == srcDevId) {
    DIPU_CALLTOPSRT(
        ::topsMemcpyAsync(dst, src, nbytes, topsMemcpyDeviceToDevice, stream))
  } else {
    DIPU_CALLTOPSRT(
        ::topsMemcpyPeerAsync(dst, dstDevId, src, srcDevId, nbytes, stream))
  }
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLTOPSRT(
      ::topsMemcpyAsync(dst, src, nbytes, topsMemcpyHostToDevice, stream))
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLTOPSRT(
      ::topsMemcpyAsync(dst, src, nbytes, topsMemcpyDeviceToHost, stream));
}

}  // end namespace devapis

}  // namespace dipu
