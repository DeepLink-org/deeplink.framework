// Copyright (c) 2023, DeepLink.
#include "../basedeviceimpl.hpp"

namespace dipu {
namespace devapis {

#define DIPU_INIT_CNDEV_VERSION(info) info.version = CNDEV_VERSION_5;

// camb6.0.2
// set current device given device according to id
void setDevice(deviceId_t devId) {
  camb_deviceId devId_ = static_cast<deviceId_t>(devId);
  DIPU_CALLCNRT(::cnrtSetDevice(devId_))
}

void initializeVendor() { DIPU_CALLCNDEV(::cndevInit(0)); }

void finalizeVendor() { ::cndevRelease(); }

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  ::cnrtDeviceProp_t device_prop;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t multi_processor_cnt = 1;
  ::cndevMemoryInfo_t mem_info;
  DIPU_CALLCNRT(::cnrtGetDeviceProperties(&device_prop, device_index));
  DIPU_CALLCNRT(::cnrtDeviceGetAttribute(
      &major, ::cnrtAttrComputeCapabilityMajor, device_index));
  DIPU_CALLCNRT(::cnrtDeviceGetAttribute(
      &minor, ::cnrtAttrComputeCapabilityMinor, device_index));
  // DIPU_CALLCNRT(::cnrtDeviceGetAttribute(&multi_processor_cnt,
  // ::cnrtAttrConcurrentKernels, device_index));
  DIPU_INIT_CNDEV_VERSION(mem_info);
  DIPU_CALLCNDEV(::cndevGetMemoryUsage(&mem_info, device_index));

  DIPUDeviceProperties prop;
  prop.name = device_prop.name;
  prop.totalGlobalMem = mem_info.physicalMemoryTotal << 20;
  prop.major = major;
  prop.minor = minor;
  prop.multiProcessorCount = multi_processor_cnt;
  return prop;
}

/*
  both cndevMemoryInfo_t.physicalMemoryUsed from cndevGetMemoryUsage and
cndevProcessInfo_t from cndevGetProcessInfo seems not correct, value always
zero, need further investigation. DIPUDeviceStatus getDeviceStatus(int32_t
device_index) {
}
*/

// check last launch succ or not, throw if fail
void checkLastError() { DIPU_CALLCNRT(::cnrtGetLastError()) }

void getRuntimeVersion(int *version) {
  int major, minor, patch;
  DIPU_CALLCNRT(::cnrtGetLibVersion(&major, &minor, &patch))
  *version = major * 10000 + minor * 100 + patch;
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t *stream, bool prior) {
  if (prior) {
    DIPU_LOGW(
        "Camb device doesn't support prior queue(stream)."
        " Fall back on creating queue without priority.");
  }
  DIPU_CALLCNRT(::cnrtQueueCreate(stream));
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtQueueDestroy(stream));
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  setDevice(devId);
  destroyStream(stream);
}

void syncStream(deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtQueueSync(stream));
}

bool isStreamEmpty(deviceStream_t stream) {
  auto err = cnrtQueueQuery(stream);
  if (err == CNRT_RET_SUCCESS) {
    return true;
  }
  return false;
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t *event) {
  DIPU_CALLCNRT(::cnrtNotifierCreate(event))
}

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLCNRT(::cnrtNotifierDestroy(event))
}

// =====================
//  mem related
// =====================
void mallocHost(void **p, size_t nbytes) {
  DIPU_CALLCNRT(cnrtHostMalloc(p, nbytes))
}

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void *ptr, int val, size_t size) {
  DIPU_CALLCNRT(cnrtMemsetAsync(ptr, val, size, stream))
}

}  // end namespace devapis
}  // end namespace dipu
