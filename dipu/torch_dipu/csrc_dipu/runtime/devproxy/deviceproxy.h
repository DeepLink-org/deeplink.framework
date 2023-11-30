// Copyright (c) 2023, DeepLink.
#pragma once

#include "../device/deviceapis.h"

namespace dipu {

namespace devproxy {

using dipu::devapis::deviceId_t;
using dipu::devapis::DIPUDeviceProperties;
using dipu::devapis::DIPUDeviceStatus;
using dipu::devapis::EventStatus;
using dipu::devapis::OpStatus;

DIPU_API void initializeVendor();

DIPU_API void finalizeVendor();

DIPU_API deviceId_t current_device();

DIPU_API DIPUDeviceProperties getDeviceProperties(int32_t device_index);
DIPU_API DIPUDeviceStatus getDeviceStatus(int32_t device_index);

// set current device given device according to id
DIPU_API void setDevice(deviceId_t devId);

DIPU_API void resetDevice(deviceId_t devId = 0);

DIPU_API void syncDevice();

// check last launch succ or not, throw if fail
DIPU_API void checkLastError();

DIPU_API int getDeviceCount();

DIPU_API void getDriverVersion(int* version);

DIPU_API void getRuntimeVersion(int* version);

DIPU_API void createStream(deviceStream_t* stream, bool prior = false);

DIPU_API void destroyStream(deviceStream_t stream);
DIPU_API void destroyStream(deviceStream_t stream, deviceId_t devId);

DIPU_API void releaseStream();

DIPU_API void syncStream(deviceStream_t stream);

DIPU_API bool streamNotNull(deviceStream_t stream);

DIPU_API void streamWaitEvent(deviceStream_t stream, deviceEvent_t event);

// same as query last event status in stream.(every op has a event)
DIPU_API bool isStreamEmpty(deviceStream_t stream);

// =====================
//  device event related
// =====================

DIPU_API void createEvent(deviceEvent_t* event);

DIPU_API void destroyEvent(deviceEvent_t event);

DIPU_API void waitEvent(deviceEvent_t event);

DIPU_API void recordEvent(deviceEvent_t event, deviceStream_t stream);

DIPU_API void eventElapsedTime(float* time, deviceEvent_t start,
                               deviceEvent_t end);

DIPU_API EventStatus getEventStatus(deviceEvent_t event);

// =====================
//  mem related
// =====================
DIPU_API void mallocHost(void** p, size_t nbytes);

DIPU_API void freeHost(void* p);

DIPU_API OpStatus mallocDevice(void** p, size_t nbytes,
                               bool throwExcepion = true);

DIPU_API void freeDevice(void* p);

DIPU_API bool isPinnedPtr(const void* p);

// (asynchronous) set val
DIPU_API void memSetAsync(const deviceStream_t stream, void* ptr, int val,
                          size_t size);

// (synchronous) copy from device to a device
DIPU_API void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                         deviceId_t srcDevId, const void* src);

// (synchronous) copy from host to a device
DIPU_API void memCopyH2D(size_t nbytes, /*deviceId_t dstDevId,*/ void* dst,
                         /*Host srcDev,*/ const void* src);

// (synchronous) copy from a device to host
DIPU_API void memCopyD2H(size_t nbytes, /*Host dstDev,*/ void* dst,
                         /*deviceId_t srcDevId,*/ const void* src);

// (asynchronous) copy from device to a device
DIPU_API void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                              deviceId_t dstDevId, void* dst,
                              deviceId_t srcDevId, const void* src);

// (asynchronous) copy from host to a device
DIPU_API void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes,
                              /*deviceId_t dstDevId,*/ void* dst,
                              /*Host srcDev,*/ const void* src);

// (asynchronous) copy from a device to host
DIPU_API void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes,
                              /*Host dstDev,*/ void* dst,
                              /*deviceId_t srcDevId,*/ const void* src);

}  // end namespace devproxy
}  // end namespace dipu
