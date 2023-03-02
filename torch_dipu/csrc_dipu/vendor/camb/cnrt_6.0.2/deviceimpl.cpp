#include "../basedeviceimpl.hpp"

namespace dipu
{
namespace devapis {
// camb6.0.2
// set current device given device according to id
void setDevice(deviceId_t devId) {
  camb_deviceId devId_ = static_cast<deviceId_t>(devId);
  DIPU_CALLCNRT(::cnrtSetDevice(devId_))
}

// check last launch succ or not, throw if fail
void checkLastError() {
     DIPU_CALLCNRT(::cnrtGetLastError())
}

void getRuntimeVersion(int *version) {
    int major, minor, patch;
    DIPU_CALLCNRT(::cnrtGetLibVersion(&major, &minor, &patch))
    *version = major * 10000 + minor * 100 + patch;
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
  DIPU_CALLCNRT(::cnrtQueueCreate(stream));
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtQueueDestroy(stream));
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

void createEvent(deviceEvent_t* event) {
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
void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
    DIPU_CALLCNRT(cnrtMemsetAsync(ptr, val, size, stream))
}

} // end namespace devapis
} // end namespace dipu
