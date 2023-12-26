

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {
DIPU_API devapis::VendorDeviceType VENDOR_TYPE = devapis::VendorDeviceType::KLX;

namespace devapis {

using klx_deviceId = int;

// =====================
//  Device class related
// =====================

void initializeVendor() {}

void finalizeVendor() {}

deviceId_t current_device() {
  klx_deviceId devId_;
  DIPU_CALLKLX(xpu_current_device(&devId_))
  return static_cast<deviceId_t>(devId_);
}

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  DIPUDeviceProperties prop;
  return prop;
}

// set current device given device according to id
void setDevice(deviceId_t devId) {
  klx_deviceId devId_ = static_cast<deviceId_t>(devId);
  DIPU_CALLKLX(xpu_set_device(devId_))
}

void resetDevice(deviceId_t devId) {
  DIPU_CALLKLX_ERROR("[kunlunxin]resetDevice is not implemented")
}

void syncDevice() { DIPU_CALLKLX(xpu_wait()); }

// check last launch succ or not, throw if fail
void checkLastError() {
  DIPU_CALLKLX_ERROR("[kunlunxin]checkLastError is not implemented")
}

int getDeviceCount() {
  int num = -1;
  DIPU_CALLKLX(xpu_device_count(reinterpret_cast<int*>(&num)))
  return num;
}

void getDriverVersion(int* version) {
  uint32_t major;
  uint32_t minor;
  DIPU_CALLKLX(xpu_get_driver_version(&major, &minor));
  *version = static_cast<int32_t>(major);
}

void getRuntimeVersion(int* version) {
  uint32_t major;
  uint32_t minor;
  DIPU_CALLKLX(xpu_get_runtime_version(&major, &minor));
  *version = static_cast<int32_t>(major);
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
  if (prior) {
    DIPU_LOGW(
        "kunlunxin device doesn't support prior queue(stream)."
        " Fall back on creating queue without priority.");
  }
  DIPU_CALLKLX(xpu_stream_create(stream));
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLKLX(xpu_stream_destroy(stream))
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  setDevice(devId);
  destroyStream(stream);
}

void releaseStream() {}

bool streamNotNull(deviceStream_t stream) { return stream != nullptr; }

void syncStream(deviceStream_t stream) { DIPU_CALLKLX(xpu_wait(stream)); }

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  DIPU_CALLKLX(xpu_stream_wait_event(stream, event))
}

bool isStreamEmpty(deviceStream_t stream) {
  DIPU_CALLKLX_ERROR("[kunlunxin]isStreamEmpty is not implemented")
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) { DIPU_CALLKLX(xpu_event_create(event)) }

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLKLX(xpu_event_destroy(event))
}

void waitEvent(deviceEvent_t event) { DIPU_CALLKLX(xpu_event_wait(event)) }

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  DIPU_CALLKLX(xpu_event_record(event, stream))
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end) {}

EventStatus getEventStatus(deviceEvent_t event) {
  return devapis::EventStatus::READY;
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
  DIPU_CALLKLX(xpu_host_alloc(p, nbytes, 0))
}

void freeHost(void* p){DIPU_CALLKLX(xpu_host_free(p))}

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  if (nbytes == 0) {
    return OpStatus::SUCCESS;
  }
  int r = xpu_malloc(p, nbytes);
  if (r != 0) {
    if (throwExcepion) {
      DIPU_LOGE("call xpu_malloc function failed.");
      throw std::runtime_error("alloc failed in dipu");
    } else if (r == XPUERR_NOMEM) {
      return OpStatus::ERR_NOMEM;
    } else {
      return OpStatus::ERR_UNKNOWN;
    }
  }
  return OpStatus::SUCCESS;
}

void freeDevice(void* p) { DIPU_CALLKLX(xpu_free(p)) }

bool isPinnedPtr(const void* p) { return false; }

static int _xpuMemset(void* ptr, int value, size_t count,
                      deviceStream_t stream) {
  if (count == 0) {
    // skip if nothing to write.
    return 0;
  }
  if (ptr == nullptr) {
    return -1;
  }

  void* ptr_host = nullptr;
  ptr_host = malloc(count);
  if (ptr_host == nullptr) {
    return -1;
  }
  int ret = xpu_memcpy(ptr, ptr_host, static_cast<uint64_t>(count),
                       XPU_HOST_TO_DEVICE);
  free(ptr_host);
  return ret;
}

void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  DIPU_CALLKLX(_xpuMemset(ptr, val, size, stream))
}

void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  if (dstDevId == srcDevId) {
    DIPU_CALLKLX(xpu_memcpy(dst, src, nbytes, XPU_DEVICE_TO_DEVICE))
  } else {
    DIPU_CALLKLX(xpu_memcpy_peer(dstDevId, dst, srcDevId, src,
                                 static_cast<uint64_t>(nbytes)))
  }
}

// (synchronous) copy from host to a DROPLET device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLKLX(xpu_memcpy(dst, src, nbytes, XPU_HOST_TO_DEVICE))
}

// (synchronous) copy from a DROPLET device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLKLX(xpu_memcpy(dst, src, nbytes, XPU_DEVICE_TO_HOST))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  memCopyD2D(nbytes, dstDevId, dst, srcDevId, src);
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  memCopyH2D(nbytes, dst, src);
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  memCopyD2H(nbytes, dst, src);
}

}  // end namespace devapis

}  // namespace dipu
