#include <cuda_runtime_api.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/common.h>

namespace dipu {
DIPU_API devapis::VendorDeviceType VENDOR_TYPE = devapis::VendorDeviceType::CUDA;

namespace devapis {

using cuda_deviceId = int;
// =====================
//  Device class related
// =====================
deviceId_t current_device() {
  cuda_deviceId devId_;
  DIPU_CALLCUDA(::cudaGetDevice(&devId_))
  return static_cast<deviceId_t>(devId_);
}   

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  ::cudaDeviceProp device_prop;
  DIPU_CALLCUDA(cudaGetDeviceProperties(&device_prop, device_index))

  DIPUDeviceProperties prop;
  prop.name = device_prop.name;
  prop.totalGlobalMem = device_prop.totalGlobalMem;
  prop.major = device_prop.major;
  prop.minor = device_prop.minor;
  prop.multiProcessorCount = device_prop.multiProcessorCount;
  return prop;
}

// in cuda_runtime_api.h
// set current device given device according to id
void setDevice(deviceId_t devId) {
    cuda_deviceId devId_ = static_cast<deviceId_t>(devId);
    DIPU_CALLCUDA(::cudaSetDevice(devId_))
}

void resetDevice(deviceId_t devId) {
    DIPU_CALLCUDA(::cudaDeviceReset())
}

void syncDevice() {
    DIPU_CALLCUDA(::cudaDeviceSynchronize())
}

// check last launch succ or not, throw if fail
void checkLastError() {
    DIPU_CALLCUDA(::cudaGetLastError())
}

int getDeviceCount() {
  int num = -1;
  DIPU_CALLCUDA(::cudaGetDeviceCount(reinterpret_cast<int*>(&num)))
  return num;
}

void getDriverVersion(int* version) {
    DIPU_CALLCUDA(::cudaDriverGetVersion(version))
}

void getRuntimeVersion(int* version) {
    DIPU_CALLCUDA(::cudaRuntimeGetVersion(version))
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
    if (prior) {
        DIPU_CALLCUDA(::cudaStreamCreateWithPriority(stream, cudaStreamDefault, -1))
    } else {
        DIPU_CALLCUDA(::cudaStreamCreate(stream))
    }
}

void destroyStream(deviceStream_t stream) {
    DIPU_CALLCUDA(::cudaStreamDestroy(stream))
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
    setDevice(devId);
    destroyStream(stream);
}

void releaseStream() {
    return;
}

bool streamNotNull(deviceStream_t stream) {
    return (stream != nullptr && stream != cudaStreamLegacy && stream != cudaStreamPerThread);
}

void syncStream(deviceStream_t stream) {
    DIPU_CALLCUDA(::cudaStreamSynchronize(stream));
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
    DIPU_CALLCUDA(::cudaStreamWaitEvent(stream, event, 0))
}

bool isStreamEmpty(deviceStream_t stream) {
  auto err = cudaStreamQuery(stream);
  if (err == ::cudaSuccess) {
    return true;
  }
  return false;
}


// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) {
    DIPU_CALLCUDA(::cudaEventCreateWithFlags(event, cudaEventDisableTiming))
}

void destroyEvent(deviceEvent_t event) {
    DIPU_CALLCUDA(::cudaEventDestroy(event))
}

void waitEvent(deviceEvent_t event) {
    DIPU_CALLCUDA(::cudaEventSynchronize(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
    DIPU_CALLCUDA(::cudaEventRecord(event, stream))
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end) {
    DIPU_CALLCUDA(cudaEventElapsedTime(time, start, end))
}

EventStatus getEventStatus(deviceEvent_t event) {
    ::cudaError_t ret = ::cudaEventQuery(event);
    if (ret == ::cudaSuccess) {
        return devapis::EventStatus::READY;
    } else if (ret == ::cudaErrorNotReady) {
        ::cudaGetLastError(); /* reset internal error state*/
        return devapis::EventStatus::PENDING;
    } else {
        throw std::runtime_error("dipu device error");
    }
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
    DIPU_CALLCUDA(::cudaMallocHost(p, nbytes))
}

void freeHost(void* p) {
    DIPU_CALLCUDA(::cudaFreeHost(p))
}

OpStatus mallocDevice(void **p, size_t nbytes, bool throwExcepion) {
    ::cudaError_t r = ::cudaMalloc(p, nbytes);
    if (r != ::cudaSuccess) {
        if(throwExcepion) {
            ::cudaGetLastError(); /* reset internal error state*/
            throw std::runtime_error("alloc failed in dipu");
        }
        else if(r == ::cudaErrorMemoryAllocation) {
            return OpStatus::ERR_NOMEM;
        }
        else {
            return OpStatus::ERR_UNKNOWN;
        }
    }
    return OpStatus::SUCCESS;
}

void freeDevice(void* p) {
    DIPU_CALLCUDA(::cudaFree(p))
}

void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
    DIPU_CALLCUDA(::cudaMemsetAsync(ptr, val, size, stream))
}

void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src) {
    if (dstDevId == srcDevId) {
        DIPU_CALLCUDA(::cudaMemcpy(dst, src, nbytes, ::cudaMemcpyDeviceToDevice))
    } else {
        DIPU_CALLCUDA(::cudaMemcpyPeer(dst, dstDevId, src, srcDevId, nbytes))
    }
}

// (synchronous) copy from host to a CUDA device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
    DIPU_CALLCUDA(::cudaMemcpy(dst, src, nbytes, ::cudaMemcpyHostToDevice))
}

// (synchronous) copy from a CUDA device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
    DIPU_CALLCUDA(::cudaMemcpy(dst, src, nbytes, ::cudaMemcpyDeviceToHost))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
        deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src) {
  if (dstDevId == srcDevId) {
      DIPU_CALLCUDA(::cudaMemcpyAsync(
          dst, src, nbytes, cudaMemcpyDeviceToDevice, stream))
  } else {
      DIPU_CALLCUDA(::cudaMemcpyPeerAsync(
          dst, dstDevId, src, srcDevId, nbytes, stream))
  }
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst, const void* src) {
    DIPU_CALLCUDA(::cudaMemcpyAsync(
            dst, src, nbytes, cudaMemcpyHostToDevice, stream))
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,  const void* src) {
  DIPU_CALLCUDA(::cudaMemcpyAsync(
    dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
}

}   // end namespace devapis

}  // namespace parrots









