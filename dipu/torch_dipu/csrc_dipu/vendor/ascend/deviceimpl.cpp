// Copyright (c) 2023, DeepLink.
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/utils/env.hpp"
#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/utils/env.hpp>

#include "basecommimpl.hpp"

namespace dipu {

namespace devapis {

// =====================
//  Device class related
// =====================

using AscendDeviceId = int32_t;

namespace {

const bool forceFallbackP2PCopy =
    get_env_or_default("DIPU_FORCE_FALLBACK_ASCEND_P2P_COPY", false);

class NpuP2PInfo {
  enum class P2pStatus : int8_t {
    UNKNOWN = -1,
    COPY_NOT_ALLOWED = 0,
    COPY_ALLOWED = 1
  };

 public:
  static NpuP2PInfo& getInstance() {
    static NpuP2PInfo instance;
    return instance;
  }

  bool enableP2P(int32_t srcDevId, int32_t dstDevId) {
    const int32_t index = srcDevId * num_devices_ + dstDevId;
    auto& status = p2p_access_enabled_cache_.at(index);
    if (status != P2pStatus::UNKNOWN) {
      return static_cast<bool>(status);
    }

    int32_t canAccessPeer = 0;
    DIPU_CALLACLRT(
        ::aclrtDeviceCanAccessPeer(&canAccessPeer, srcDevId, dstDevId));
    if (canAccessPeer != 1) {
      TORCH_WARN("can not copy memory form device ", srcDevId, " to device ",
                 dstDevId);
      status = P2pStatus::COPY_NOT_ALLOWED;
      return false;
    }
    int32_t currentDevice = -1;
    DIPU_CALLACLRT(aclrtGetDevice(&currentDevice));

    DIPU_CALLACLRT(aclrtSetDevice(dstDevId));
    DIPU_CALLACLRT(aclrtDeviceEnablePeerAccess(srcDevId, 0 /*reserved*/));
    DIPU_CALLACLRT(aclrtSetDevice(srcDevId));
    DIPU_CALLACLRT(aclrtDeviceEnablePeerAccess(dstDevId, 0 /*reserved*/));

    DIPU_CALLACLRT(aclrtSetDevice(currentDevice));
    if (canAccessPeer == 1) {
      status = P2pStatus::COPY_ALLOWED;
      return true;
    }
    return false;
  }

 private:
  // Use a 1-dimensional vector to store 2-dimensional data
  std::vector<P2pStatus> p2p_access_enabled_cache_;

  const int64_t num_devices_;
  NpuP2PInfo() : num_devices_(getDeviceCount()) {
    p2p_access_enabled_cache_ =
        std::vector<P2pStatus>(num_devices_ * num_devices_, P2pStatus::UNKNOWN);
  }
};

}  // namespace

void initializeVendor() {
  DIPU_CALLACLRT(aclInit(nullptr));
  DIPU_CALLACLRT(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
}

void finalizeVendor() { DIPU_CALLACLRT(aclFinalize()); }

deviceId_t current_device() {
  int32_t deviceId = 0;
  DIPU_CALLACLRT(aclrtGetDevice(&deviceId));
  return static_cast<deviceId_t>(deviceId);
}

// set current device given device according to id
void setDevice(deviceId_t device_id) {
  DIPU_CALLACLRT(aclrtSetDevice(device_id));
}

DIPUDeviceProperties getDeviceProperties(AscendDeviceId device_index) {
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
  int patch = 0;
  DIPU_CALLACLRT(::aclrtGetVersion(&prop.major, &prop.minor, &patch));
  DIPU_CALLACLRT(::aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
  prop.totalGlobalMem = device_total;
  prop.multiProcessorCount = 1;
  return prop;
}

DIPUDeviceStatus getDeviceStatus(int32_t device_index) {
  DIPUDeviceStatus status;
  DIPU_CALLACLRT(aclrtGetMemInfo(ACL_HBM_MEM, &status.freeGlobalMem,
                                 &status.totalGlobalMem));
  return status;
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
  // 若用户需申请大块内存并自行划分、管理内存时，建议使用aclrtMallocAlign32接
  // 口，该接口相比aclrtMalloc接口，只会对用户申请的size向上对齐成32字节整数
  // 倍，不会再多加32字节。 大块内存用作缓存时，无需多加32字节
  aclError err = ::aclrtMallocAlign32(p, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (err == ACL_ERROR_NONE) {
    return OpStatus::SUCCESS;
  }
  if (err == ACL_ERROR_RT_MEMORY_ALLOCATION && !throwExcepion) {
    return OpStatus::ERR_NOMEM;
  }
  TORCH_CHECK(false, "aclrtMallocAlign32 error, err = ", err,
              ", size = ", nbytes, ", error msg = ", aclGetRecentErrMsg());
}

void freeDevice(void* p) {
  if (p == nullptr) {
    return;
  }
  DIPU_CALLACLRT(::aclrtFree(p))
}

void memCopyD2DFallback(size_t nbytes, deviceId_t dstDevId, void* dst,
                        deviceId_t srcDevId, const void* src) {
  int32_t currentDevice = -1;
  DIPU_CALLACLRT(aclrtGetDevice(&currentDevice));
  void* hostBuffer = nullptr;
  mallocHost(&hostBuffer, nbytes);
  std::unique_ptr<void, void (*)(void*)> hostBufferUnique(
      hostBuffer,
      freeHost);  // for exception safety
  DIPU_CALLACLRT(aclrtSetDevice(srcDevId));
  memCopyD2H(nbytes, hostBuffer, src);
  DIPU_CALLACLRT(aclrtSetDevice(dstDevId));
  memCopyH2D(nbytes, dst, hostBuffer);
  DIPU_CALLACLRT(aclrtSetDevice(currentDevice));
}

// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  if (dstDevId != srcDevId) {
    if (forceFallbackP2PCopy ||
        NpuP2PInfo::getInstance().enableP2P(srcDevId, dstDevId) == false) {
      memCopyD2DFallback(nbytes, dstDevId, dst, srcDevId, src);
      return;
    }
  }
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
}

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLACLRT(
      ::aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  if (dstDevId != srcDevId) {
    if (forceFallbackP2PCopy ||
        NpuP2PInfo::getInstance().enableP2P(srcDevId, dstDevId) == false) {
      memCopyD2DFallback(nbytes, dstDevId, dst, srcDevId, src);
      return;
    }
  }
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
  DIPU_CALLACLRT(::aclrtStreamWaitEvent(stream, event))
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
