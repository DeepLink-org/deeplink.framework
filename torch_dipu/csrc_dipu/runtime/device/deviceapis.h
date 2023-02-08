#pragma once

#include <c10/core/Device.h>
#include <cstring>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/vendor/vendorapi.h>


namespace torch_dipu {

namespace devapis {

using deviceId_t = c10::DeviceIndex;

// need cache.
// need to discuss, some device has hidden use, others may has more than one handler type.
deviceHandle_t getDeviceHandler(c10::DeviceIndex device_index);

deviceId_t current_device();

// set current device given device according to id
void setDevice(deviceId_t devId);

void resetDevice(deviceId_t devId = 0);

void syncDevice();

// check last launch succ or not, throw if fail
void checkLastError();

int getDeviceCount();

void getDriverVersion(int* version);

void getRuntimeVersion(int* version);

void createStream(deviceStream_t* stream, bool prior=false);

void destroyStream(deviceStream_t stream);
void destroyStream(deviceStream_t stream, deviceId_t devId);

void releaseStream();

void syncStream(deviceStream_t stream);

bool streamNotNull(deviceStream_t stream);

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event);

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event);

// same as query last event status in stream.(every op has a event)
bool isStreamEmpty(deviceStream_t stream);

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event);

void destroyEvent(deviceEvent_t event);

void waitEvent(deviceEvent_t event);

void recordEvent(deviceEvent_t event, deviceStream_t stream);

void eventElapsedTime(float *time, deviceEvent_t start, deviceEvent_t end);

EventStatus getEventStatus(deviceEvent_t event);

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes);

void freeHost(void* p);

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion= true);

void freeDevice(void* p);

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size);

// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src);

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, /*deviceId_t dstDevId,*/ void* dst, /*Host srcDev,*/ const void* src);

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, /*Host dstDev,*/ void* dst, /*deviceId_t srcDevId,*/ const void* src);

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
        deviceId_t dstDevId, void* dst, deviceId_t srcDevId, const void* src);

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes,
        /*deviceId_t dstDevId,*/ void* dst, /*Host srcDev,*/ const void* src);

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes,
        /*Host dstDev,*/ void* dst, /*deviceId_t srcDevId,*/ const void* src);
}  // end namespace devapis
}  // end namespace torch_dipu