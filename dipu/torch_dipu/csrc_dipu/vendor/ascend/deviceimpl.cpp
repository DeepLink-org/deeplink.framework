// Copyright (c) 2023, DeepLink.
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <atomic>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

#include "basecommimpl.hpp"

namespace dipu {

namespace devapis {

// =====================
//  Device class related
// =====================
using ascend_deviceId = int32_t;

std::atomic<int> kCurrentDeviceIndex(-1);

void initializeVendor() { DIPU_CALLACLRT(aclInit(nullptr)); }

void finalizeVendor() { DIPU_CALLACLRT(aclFinalize()); }

deviceId_t current_device() {
  if (kCurrentDeviceIndex < 0) {
    setDevice(0);
    return 0;
  }
  return static_cast<deviceId_t>(kCurrentDeviceIndex);
}

// set current device given device according to id
void setDevice(deviceId_t devId) {
  if (devId != kCurrentDeviceIndex) {
    kCurrentDeviceIndex = devId;
    DIPU_CALLACLRT(::aclrtSetDevice(devId))
  }
}

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  const char* device_name;
  size_t device_free;
  size_t device_total;
  device_name = aclrtGetSocName();
  DIPUDeviceProperties prop;
  if (device_name == nullptr) {
    prop.name = " ";
    DIPU_LOGE("Get ascend device name fail.");
  } else {
    prop.name = std::string(device_name);
  }
  int patch;
  DIPU_CALLACLRT(::aclrtGetVersion(&prop.major, &prop.minor, &patch));
  DIPU_CALLACLRT(::aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
  // NOTE : unit of PhysicalMemoryTotal is MB
  prop.totalGlobalMem = device_total << 20;
  prop.multiProcessorCount = 1;
  return prop;
}

void resetDevice(deviceId_t devId) { DIPU_CALLACLRT(::aclrtResetDevice(devId)) }

void syncDevice() { DIPU_CALLACLRT(::aclrtSynchronizeDevice()) }

int getDeviceCount() {
  unsigned int num = -1;
  DIPU_CALLACLRT(::aclrtGetDeviceCount(&num))
  return num;
}

void getDriverVersion(int* version) {
  int32_t majorVersion;
  int32_t minorVersion;
  int32_t patchVersion;
  DIPU_CALLACLRT(aclrtGetVersion(&majorVersion, &minorVersion, &patchVersion));
  *version = majorVersion;
}

// =====================
//  mem related
// =====================

void mallocHost(void** p, size_t nbytes) {
  if (nbytes <= 0) {
    *p = nullptr;
    return;
  }
  DIPU_CALLACLRT(aclrtMallocHost(p, nbytes))
}

void freeHost(void* p) {
  if (p == nullptr) {
    return;
  }
  DIPU_CALLACLRT(aclrtFreeHost(p))
}

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  if (nbytes <= 0) {
    *p = nullptr;
    return OpStatus::SUCCESS;
  }
  DIPU_CALLACLRT(::aclrtMalloc(p, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
  return OpStatus::SUCCESS;
}

void freeDevice(void* p) {
  if (p == nullptr) {
    return;
  }
  DIPU_CALLACLRT(::aclrtFree(p))
}

// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  syncDevice();
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
}

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
  syncDevice();
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
  syncDevice();
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  DIPU_CALLACLRT(::aclrtMemcpyAsync(dst, nbytes, src, nbytes,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLACLRT(::aclrtMemcpyAsync(dst, nbytes, src, nbytes,
                                    ACL_MEMCPY_HOST_TO_DEVICE, stream));
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLACLRT(::aclrtMemcpyAsync(dst, nbytes, src, nbytes,
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream));
}

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  DIPU_CALLACLRT(aclrtMemsetAsync(ptr, size, val, size, stream));
}

// check last launch succ or not, throw if fail
void checkLastError() {
  const char* erroInfo = aclGetRecentErrMsg();
  if (erroInfo == nullptr) {
    return;
  }
  printf("%s\n", erroInfo);
}

void getRuntimeVersion(int* version) {
  int major, minor, patch;
  DIPU_CALLACLRT(::aclrtGetVersion(&major, &minor, &patch))
  *version = major * 10000 + minor * 100 + patch;
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
  if (prior) {
    DIPU_LOGW(
        "Ascend device doesn't support prior queue(stream)."
        " Fall back on creating queue without priority.");
  }
  DIPU_CALLACLRT(::aclrtCreateStream(stream));
  std::cout << __FUNCTION__ << ":" << *stream << std::endl;
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLACLRT(::aclrtDestroyStream(stream));
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  setDevice(devId);
  destroyStream(stream);
}

void syncStream(deviceStream_t stream) {
  DIPU_CALLACLRT(::aclrtSynchronizeStream(stream));
}

bool isStreamEmpty(deviceStream_t stream) {
  // aclrtSynchronizeStreamWithTimeout(stream);
  return false;
}

// =====================
//  device event related
// =====================

void releaseStream() { return; }

bool streamNotNull(deviceStream_t stream) {
  return stream != deviceDefaultStreamLiteral;
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  DIPU_CALLACLRT(::aclrtSynchronizeEvent(event))
}

// =====================
//  device event related
// =====================
void waitEvent(deviceEvent_t event) {
  DIPU_CALLACLRT(::aclrtSynchronizeEvent(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  DIPU_CALLACLRT(::aclrtRecordEvent(event, stream));
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end){
    DIPU_CALLACLRT(aclrtEventElapsedTime(time, start, end))}

EventStatus getEventStatus(deviceEvent_t event) {
  aclrtEventRecordedStatus status;
  DIPU_CALLACLRT(aclrtQueryEventStatus(event, &status))
  if (status == ::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
    return devapis::EventStatus::READY;
  } else if (status == ::ACL_EVENT_RECORDED_STATUS_NOT_READY) {
    return devapis::EventStatus::PENDING;
  }
  throw std::runtime_error("dipu device error");
}

void createEvent(deviceEvent_t* event) {
  DIPU_CALLACLRT(::aclrtCreateEvent(event))
}

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLACLRT(::aclrtDestroyEvent(event))
}

bool isPinnedPtr(const void* p) {
  TORCH_CHECK(false, "isPinnedPtr not implemented for ascend.\n");
  return false;
}

}  // end namespace devapis
}  // end namespace dipu
