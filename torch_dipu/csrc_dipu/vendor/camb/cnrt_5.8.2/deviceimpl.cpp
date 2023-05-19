// Copyright (c) 2023, DeepLink.
#include "../basedeviceimpl.hpp"

namespace dipu
{
namespace devapis {
// camb5.8.2
// set current device given device according to id
void setDevice(deviceId_t devId) {
  camb_deviceId devId_ = static_cast<deviceId_t>(devId);
  cnrtDev_t dev;
  DIPU_CALLCNRT(::cnrtGetDeviceHandle(&dev, devId_))
  DIPU_CALLCNRT(::cnrtSetCurrentDevice(dev))
}

// check last launch succ or not, throw if fail
void checkLastError() {
  DIPU_CALLCNRT(::cnrtGetLastErr())
}

void getRuntimeVersion(int *version) {
  DIPU_CALLCNRT(::cnrtGetVersion(reinterpret_cast<unsigned int *>(version)))
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t *stream, bool prior) {
  if (prior)
  {
    DIPU_LOGW(
        "Camb device doesn't support prior queue(stream)."
        " Fall back on creating queue without priority.");
  }
  DIPU_CALLCNRT(::cnrtCreateQueue(stream));
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtDestroyQueue(stream));
}

void syncStream(deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtSyncQueue(stream));
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
  DIPU_CALLCNRT(::cnrtCreateNotifier(event))
}

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLCNRT(::cnrtDestroyNotifier(&event))
}

// =====================
//  mem related
// =====================
void mallocHost(void **p, size_t nbytes) {
  DIPU_CALLCNRT(cnrtMallocHost(p, nbytes, CNRT_MEMTYPE_LOCKED))
}

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void *ptr, int val, size_t size) {
  DIPU_CALLCNRT(cnrtMemsetD8Async(ptr, val, size, stream))
}

} // end namespace devapis
} // end namespace dipu
