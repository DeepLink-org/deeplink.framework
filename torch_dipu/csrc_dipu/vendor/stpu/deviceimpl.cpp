

#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/common.h>

namespace dipu {
DIPU_API devapis::VendorDeviceType VENDOR_TYPE = devapis::VendorDeviceType::STPU;

namespace devapis {

using stpu_deviceId = int;
// =====================
//  Device class related
// =====================
deviceId_t current_device() {
  stpu_deviceId devId_;
  DIPU_CALLSTPU(::tangGetDevice(&devId_))
  return static_cast<deviceId_t>(devId_);
}   

// in tang_runtime_api.h
// set current device given device according to id
void setDevice(deviceId_t devId) {
    stpu_deviceId devId_ = static_cast<deviceId_t>(devId);
    DIPU_CALLSTPU(::tangSetDevice(devId_))
}

void resetDevice(deviceId_t devId) {
    // DIPU_CALLSTPU(::tangDeviceReset())
}

void syncDevice() {
    DIPU_CALLSTPU(::tangDeviceSynchronize())
}

// check last launch succ or not, throw if fail
void checkLastError() {
    DIPU_CALLSTPU(::tangGetLastError())
}

int getDeviceCount() {
  int num = -1;
  DIPU_CALLSTPU(::tangGetDeviceCount(reinterpret_cast<int*>(&num)))
  return num;
}

void getDriverVersion(int* version) {
    // DIPU_CALLSTPU(::tangDriverGetVersion(version))
}

void getRuntimeVersion(int* version) {
    // DIPU_CALLSTPU(::tangRuntimeGetVersion(version))
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
    if (prior) {
        DIPU_CALLSTPU(::tangStreamCreateWithPriority(stream, tangStreamDefault, -1))
    } else {
        DIPU_CALLSTPU(::tangStreamCreate(stream))
    }
}

void destroyStream(deviceStream_t stream) {
    DIPU_CALLSTPU(::tangStreamDestroy(stream))
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
    setDevice(devId);
    destroyStream(stream);
}

void releaseStream() {
    return;
}

bool streamNotNull(deviceStream_t stream) {
    return stream != nullptr;
    // return (stream != nullptr && stream != tangStreamLegacy && stream != tangStreamPerThread);
}

void syncStream(deviceStream_t stream) {
    DIPU_CALLSTPU(::tangStreamSynchronize(stream));
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
    DIPU_CALLSTPU(::tangStreamWaitEvent(stream, event, 0))
}

bool isStreamEmpty(deviceStream_t stream) {
  auto err = tangStreamQuery(stream);
  if (err == ::tangSuccess) {
    return true;
  }
  return false;
}


// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) {
    DIPU_CALLSTPU(::tangEventCreateWithFlags(event, tangEventDisableTiming))
}

void destroyEvent(deviceEvent_t event) {
    DIPU_CALLSTPU(::tangEventDestroy(event))
}

void waitEvent(deviceEvent_t event) {
    DIPU_CALLSTPU(::tangEventSynchronize(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
    DIPU_CALLSTPU(::tangEventRecord(event, stream))
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end) {
    DIPU_CALLSTPU(tangEventElapsedTime(time, start, end))
}

EventStatus getEventStatus(deviceEvent_t event) {
    ::tangError_t ret = ::tangEventQuery(event);
    if (ret == ::tangSuccess) {
        return devapis::EventStatus::READY;
    } else if (ret == ::tangErrorNotReady) {
        ::tangGetLastError(); /* reset internal error state*/
        return devapis::EventStatus::PENDING;
    } else {
        throw std::runtime_error("dipu device error");
    }
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
    DIPU_CALLSTPU(::tangMallocHost(p, nbytes))
}

void freeHost(void* p) {
    DIPU_CALLSTPU(::tangFreeHost(p))
}

OpStatus mallocDevice(void **p, size_t nbytes, bool throwExcepion) {
    if (nbytes == 0) return OpStatus::SUCCESS;
    ::tangError_t r = ::tangMalloc(p, nbytes);
    if (r != ::tangSuccess) {
        if(throwExcepion) {
    printf("call a tangrt function failed. return code=%d %d", r, nbytes);
            ::tangGetLastError(); /* reset internal error state*/
            throw std::runtime_error("alloc failed in dipu");
        }
        else if(r == ::tangErrorMemoryAllocation) {
            return OpStatus::ERR_NOMEM;
        }
        else {
            return OpStatus::ERR_UNKNOWN;
        }
    }
    return OpStatus::SUCCESS;
}

void freeDevice(void* p) {
    DIPU_CALLSTPU(::tangFree(p))
}

void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
    DIPU_CALLSTPU(::tangMemsetAsync(ptr, val, size, stream))
}

void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src) {
    if (dstDevId == srcDevId) {
        DIPU_CALLSTPU(::tangMemcpy(dst, src, nbytes, ::tangMemcpyDeviceToDevice))
    } else {
        // DIPU_CALLSTPU(::tangMemcpyPeer(dst, dstDevId, src, srcDevId, nbytes))
        throw std::runtime_error("dipu device error with tangMemcpyPeer not supported");
    }
}

// (synchronous) copy from host to a STPU device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
    DIPU_CALLSTPU(::tangMemcpy(dst, src, nbytes, ::tangMemcpyHostToDevice))
}

// (synchronous) copy from a STPU device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
    DIPU_CALLSTPU(::tangMemcpy(dst, src, nbytes, ::tangMemcpyDeviceToHost))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
        deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src) {
  if (dstDevId == srcDevId) {
      DIPU_CALLSTPU(::tangMemcpyAsync(
          dst, src, nbytes, tangMemcpyDeviceToDevice, stream))
  } else {
    throw std::runtime_error("dipu device error with tangMemcpyPeerAsync not supported");
    //   DIPU_CALLSTPU(::tangMemcpyPeerAsync(
    //       dst, dstDevId, src, srcDevId, nbytes, stream))
  }
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst, const void* src) {
    DIPU_CALLSTPU(::tangMemcpyAsync(
            dst, src, nbytes, tangMemcpyHostToDevice, stream))
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,  const void* src) {
  DIPU_CALLSTPU(::tangMemcpyAsync(
    dst, src, nbytes, tangMemcpyDeviceToHost, stream));
}

}   // end namespace devapis

}  // namespace parrots









