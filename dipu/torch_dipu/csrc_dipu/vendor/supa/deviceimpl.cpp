#include <supa.h>

#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

namespace devapis {

#define SUPA_CALL(Expr)                                                 \
  {                                                                     \
    suError_t __ret = Expr;                                             \
    if (__ret != suSuccess) {                                           \
      printf("call a supa function (%s) failed. return code=%d", #Expr, \
             __ret);                                                    \
    }                                                                   \
  }

void initializeVendor() {}

void finalizeVendor() {}

class DeviceGuard {
 public:
  DeviceGuard(int device) : device_bak(-1) {
    SUPA_CALL(suGetDevice(&device_bak));
    if (device_bak != device) {
      SUPA_CALL(suSetDevice(device));
    } else {
      device_bak = -1;
    }
  }
  ~DeviceGuard() {
    if (device_bak >= 0) {
      SUPA_CALL(suSetDevice(device_bak));
    }
  }

 private:
  int device_bak;
};

DIPU_API deviceId_t current_device() {
  int device = 0;
  SUPA_CALL(suGetDevice(&device));
  return static_cast<deviceId_t>(device);
};

DIPU_API DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  suDeviceProp prop;
  SUPA_CALL(suGetDeviceProperties(&prop, device_index));
  return {prop.name, prop.totalGlobalMem, prop.major, prop.minor,
          prop.multiProcessorCount};
}

// set current device given device according to id
DIPU_API void setDevice(deviceId_t devId) {
  int device = static_cast<int>(devId);
  SUPA_CALL(suSetDevice(device));
};

DIPU_API void resetDevice(deviceId_t devId) {
  int device = static_cast<int>(devId);
  DeviceGuard guard(device);
  SUPA_CALL(suDeviceReset())
}

DIPU_API void syncDevice() { SUPA_CALL(suDeviceSynchronize()); }

// check last launch succ or not, throw if fail
DIPU_API void checkLastError() {
  suError_t last_err = suGetLastError();
  if (last_err != suSuccess) {
    throw std::runtime_error(
        "dipu device error, ret code:" + std::to_string(last_err) + ":" +
        suGetErrorString(last_err));
  }
}

DIPU_API int getDeviceCount() {
  int count = 0;
  SUPA_CALL(suGetDeviceCount(&count));
  return count;
}

DIPU_API void getDriverVersion(int* version) {
  SUPA_CALL(suDriverGetVersion(version));
}

DIPU_API void getRuntimeVersion(int* version) {
  SUPA_CALL(suRuntimeGetVersion(version));
}

DIPU_API void createStream(deviceStream_t* stream, bool prior) {
  int flags = suStreamDefault;
  // Note: lower numbers are higher priorities, zero is default priority
  SUPA_CALL(suStreamCreateWithPriority(stream, flags, prior ? -1 : 0));
}

DIPU_API void destroyStream(deviceStream_t stream) {
  SUPA_CALL(suStreamDestroy(stream));
}

DIPU_API void destroyStream(deviceStream_t stream, deviceId_t devId) {
  int device = static_cast<int>(devId);
  DeviceGuard guard(device);
  SUPA_CALL(suStreamDestroy(stream));
}

DIPU_API void releaseStream() {
  // throw std::runtime_error("release stream is not support.");
}

DIPU_API void syncStream(deviceStream_t stream) {
  SUPA_CALL(suStreamSynchronize(stream));
}

DIPU_API bool streamNotNull(deviceStream_t stream) { return true; }

DIPU_API void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  SUPA_CALL(suStreamWaitEvent(stream, event));
}

// same as query last event status in stream.(every op has a event)
DIPU_API bool isStreamEmpty(deviceStream_t stream) {
  auto ret = suStreamQuery(stream);
  switch (ret) {
    case suSuccess:
      return true;
    case suErrorNotReady:
      (void)suGetLastError();  // clear error of not ready.
      __attribute__((fallthrough));
    default:
      return false;
  }
}

// =====================
//  device event related
// =====================

DIPU_API void createEvent(deviceEvent_t* event) {
  SUPA_CALL(suEventCreate(event));
}

DIPU_API void destroyEvent(deviceEvent_t event) {
  SUPA_CALL(suEventDestroy(event));
}

DIPU_API void waitEvent(deviceEvent_t event) {
  SUPA_CALL(suEventSynchronize(event));
}

DIPU_API void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  SUPA_CALL(suEventRecord(event, stream));
}

DIPU_API void eventElapsedTime(float* time, deviceEvent_t start,
                               deviceEvent_t end) {
  // unit of time is ms.
  SUPA_CALL(suEventElapsedTime(time, start, end));
}

DIPU_API EventStatus getEventStatus(deviceEvent_t event) {
  auto ret = suEventQuery(event);
  switch (ret) {
    case suSuccess:
      return EventStatus::READY;
    case suErrorNotReady:
      return EventStatus::RUNNING;
    default:
      SUPA_CALL(ret);
      return EventStatus::PENDING;
  }
}

// =====================
//  mem related
// =====================
DIPU_API void mallocHost(void** p, size_t nbytes) { *p = malloc(nbytes); }

DIPU_API void freeHost(void* p) { free(p); }

extern "C" {
void* br_device_malloc(uint64_t bytes);
void br_device_free(void* ptr);
// get physical address from ptr(virtual)
void* get_phy_ptr(const void* ptr);
}

DIPU_API OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  void* ptr = nullptr;
  ptr = br_device_malloc(nbytes);
  if (ptr != nullptr) {
    *p = ptr;
    return OpStatus::SUCCESS;
  }
  if (throwExcepion) {
    throw std::runtime_error("Failed to allocate memory");
  }
  return OpStatus::ERR_NOMEM;
}

DIPU_API void freeDevice(void* p) { br_device_free(p); }

DIPU_API bool isPinnedPtr(const void* p) { return false; }

// (asynchronous) set val
DIPU_API void memSetAsync(const deviceStream_t stream, void* ptr, int val,
                          size_t size) {
  auto phy_gpu_addr = get_phy_ptr(ptr);
  SUPA_CALL(suMemsetAsync(phy_gpu_addr, val, size, stream));
}

// (synchronous) copy from device to a device
DIPU_API void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                         deviceId_t srcDevId, const void* src) {
  // SUPA uses Unified Virtual Address
  auto phy_src_gpu_addr = get_phy_ptr(src);
  auto phy_dst_gpu_addr = get_phy_ptr(dst);
  SUPA_CALL(suMemcpy(phy_dst_gpu_addr, phy_src_gpu_addr, nbytes,
                     suMemcpyDeviceToDevice));
}

// (synchronous) copy from host to a device
DIPU_API void memCopyH2D(size_t nbytes, /*deviceId_t dstDevId,*/ void* dst,
                         /*Host srcDev,*/ const void* src) {
  auto phy_dst_gpu_addr = get_phy_ptr(dst);
  SUPA_CALL(suMemcpy(phy_dst_gpu_addr, src, nbytes, suMemcpyHostToDevice));
}

// (synchronous) copy from a device to host
DIPU_API void memCopyD2H(size_t nbytes, /*Host dstDev,*/ void* dst,
                         /*deviceId_t srcDevId,*/ const void* src) {
  auto phy_src_gpu_addr = get_phy_ptr(src);
  SUPA_CALL(suMemcpy(dst, phy_src_gpu_addr, nbytes, suMemcpyDeviceToHost));
}

// (asynchronous) copy from device to a device
DIPU_API void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                              deviceId_t dstDevId, void* dst,
                              deviceId_t srcDevId, const void* src) {
  auto phy_src_gpu_addr = get_phy_ptr(src);
  auto phy_dst_gpu_addr = get_phy_ptr(dst);
  SUPA_CALL(suMemcpyAsync(phy_dst_gpu_addr, phy_src_gpu_addr, nbytes, stream,
                          suMemcpyDeviceToDevice));
}

// (asynchronous) copy from host to a device
DIPU_API void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes,
                              /*deviceId_t dstDevId,*/ void* dst,
                              /*Host srcDev,*/ const void* src) {
  auto phy_dst_gpu_addr = get_phy_ptr(dst);
  SUPA_CALL(suMemcpyAsync(phy_dst_gpu_addr, src, nbytes, stream,
                          suMemcpyHostToDevice));
}

// (asynchronous) copy from a device to host
DIPU_API void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes,
                              /*Host dstDev,*/ void* dst,
                              /*deviceId_t srcDevId,*/ const void* src) {
  auto phy_src_gpu_addr = get_phy_ptr(src);
  SUPA_CALL(suMemcpyAsync(dst, phy_src_gpu_addr, nbytes, stream,
                          suMemcpyDeviceToHost));
}
}  // end namespace devapis
}  // end namespace dipu
