#pragma once
#include <cnrt.h>
#include <cndev.h>
#include <cnnl.h>

#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/common.h>

// use header file as common is weird. change to use base class in future.
// only vendor deviceimpl.cpp can include this header
namespace dipu {
DIPU_API devapis::VendorDeviceType VENDOR_TYPE = devapis::VendorDeviceType::MLU;

namespace devapis {

using camb_deviceId = int;
// =====================
//  Device class related
// =====================
deviceId_t current_device() {
  camb_deviceId devId_;
  DIPU_CALLCNRT(::cnrtGetDevice(&devId_))
  return static_cast<deviceId_t>(devId_);
}

void resetDevice(deviceId_t devId) {
  DIPU_CALLCNRT(::cnrtDeviceReset())
}

void syncDevice() {
  DIPU_CALLCNRT(::cnrtSyncDevice())
}

int getDeviceCount() {
  int num = -1;
  DIPU_CALLCNRT(::cnrtGetDeviceCount(reinterpret_cast<unsigned int *>(&num)))
  return num;
}

void getDriverVersion(int *version) {
  cndevVersionInfo_t verInfo;
  DIPU_CALLCNDEV(::cndevGetVersionInfo(&verInfo, 0))
  *version = verInfo.version;
}

void releaseStream() {
  return;
}

bool streamNotNull(deviceStream_t stream) {
  return stream != deviceDefaultStreamLiteral;
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  DIPU_CALLCNRT(::cnrtQueueWaitNotifier(event, stream, 0))
}



// =====================
//  device event related
// =====================

void waitEvent(deviceEvent_t event) {
  DIPU_CALLCNRT(::cnrtWaitNotifier(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  DIPU_CALLCNRT(::cnrtPlaceNotifier(event, stream));
}

void eventElapsedTime(float *time, deviceEvent_t start, deviceEvent_t end) {
  DIPU_CALLCNRT(cnrtNotifierElapsedTime(start, end, time))
}

EventStatus getEventStatus(deviceEvent_t event) {
  ::cnrtRet_t ret = ::cnrtQueryNotifier(event);
  if (ret == ::cnrtSuccess)
  {
    return devapis::EventStatus::READY;
  }
  else if (ret == ::cnrtErrorBusy || ret == ::cnrtErrorNotReady)
  {
    checkLastError(); /* reset internal error state*/
    return devapis::EventStatus::PENDING;
  }
  // throw CnrtRuntimeError(ret, DIPU_CODELOC);
  throw std::runtime_error("dipu device error");
}

// =====================
//  mem related
// =====================

void freeHost(void *p) {
    DIPU_CALLCNRT(cnrtFreeHost(p))
}

OpStatus mallocDevice(void **p, size_t nbytes, bool throwExcepion) {
  ::cnrtRet_t r = ::cnrtMalloc(p, nbytes);
  if (r != ::cnrtSuccess)
  {
    if (throwExcepion)
    {
      checkLastError(); /* reset internal error state*/
      throw std::runtime_error("alloc failed in dipu");
    }
    else if ((r == ::cnrtErrorNoMem))
    {
      return OpStatus::ERR_NOMEM;
    }
    else
    {
      return OpStatus::ERR_UNKNOWN;
    }
  }
  return OpStatus::SUCCESS;
}

void freeDevice(void *p) {
  DIPU_CALLCNRT(::cnrtFree(p))
}



// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void *dst, deviceId_t srcDevId, const void *src) {
  // TODO(zhaoxiujia) : check src const
  syncDevice();
  if (srcDevId != dstDevId)
  {
    DIPU_CALLCNRT(::cnrtMemcpyPeer(
        dst, dstDevId, const_cast<void *>(src), srcDevId, nbytes))
  }
  else
  {
    if (dst != src)
    {
      DIPU_CALLCNRT(::cnrtMemcpy(
          dst, const_cast<void *>(src), nbytes, CNRT_MEM_TRANS_DIR_DEV2DEV))
    }
  }
}

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, void *dst, const void *src) {
  syncDevice();
  DIPU_CALLCNRT(::cnrtMemcpy(
      dst, const_cast<void *>(src), nbytes, CNRT_MEM_TRANS_DIR_HOST2DEV))
}

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, void *dst, const void *src) {
  syncDevice();
  DIPU_CALLCNRT(::cnrtMemcpy(
      dst, const_cast<void *>(src), nbytes, CNRT_MEM_TRANS_DIR_DEV2HOST))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                      deviceId_t dstDevId, void *dst, deviceId_t srcDevId, const void *src) {
  if (dstDevId == srcDevId)
  {
    if (dst != src)
    {
      DIPU_CALLCNRT(::cnrtMemcpyAsync(
          dst, const_cast<void *>(src), nbytes, stream, CNRT_MEM_TRANS_DIR_DEV2DEV))
    }
  }
  else
  {
    DIPU_CALLCNRT(cnrtMemcpyPeerAsync(
        dst, dstDevId, const_cast<void *>(src), srcDevId, nbytes, stream))
  }
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void *dst, const void *src){
  DIPU_CALLCNRT(::cnrtMemcpyAsync(
      dst, const_cast<void *>(src), nbytes, stream, CNRT_MEM_TRANS_DIR_HOST2DEV))
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void *dst, const void *src) {
  DIPU_CALLCNRT(::cnrtMemcpyAsync(
      dst, const_cast<void *>(src), nbytes, stream, CNRT_MEM_TRANS_DIR_DEV2HOST))
}

} // end namespace devapis
} // end namespace dipu
