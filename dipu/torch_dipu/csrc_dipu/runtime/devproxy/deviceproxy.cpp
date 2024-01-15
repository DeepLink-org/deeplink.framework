// Copyright (c) 2023, DeepLink.
#include "./deviceproxy.h"

#include "../core/DIPUEventPool.h"
namespace dipu {

namespace devproxy {

void initializeVendor() {
  if (devapis::initializeVendor) {
    devapis::initializeVendor();
  }
}

void finalizeVendor() {
  if (devapis::finalizeVendor) {
    devapis::finalizeVendor();
  }
}

deviceId_t current_device() { return devapis::current_device(); }

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  return devapis::getDeviceProperties(device_index);
}

DIPUDeviceStatus getDeviceStatus(int32_t device_index) {
  if (devapis::getDeviceStatus) {
    return devapis::getDeviceStatus(device_index);
  }
  return {};
}

// set current device given device according to id
void setDevice(deviceId_t devId) { return devapis::setDevice(devId); }

void resetDevice(deviceId_t devId) { return devapis::resetDevice(devId); }

void syncDevice() { return devapis::syncDevice(); }

// check last launch succ or not, throw if fail
void checkLastError() { return devapis::checkLastError(); }

int getDeviceCount() {
  static int device_count = devapis::getDeviceCount();
  return device_count;
}

void getDriverVersion(int* version) {
  return devapis::getDriverVersion(version);
}

void getRuntimeVersion(int* version) {
  return devapis::getRuntimeVersion(version);
}

void createStream(deviceStream_t* stream, bool prior) {
  return devapis::createStream(stream, prior);
}

void destroyStream(deviceStream_t stream) {
  return devapis::destroyStream(stream);
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  return devapis::destroyStream(stream, devId);
}

void releaseStream() { return devapis::releaseStream(); }

void syncStream(deviceStream_t stream) { return devapis::syncStream(stream); }

bool streamNotNull(deviceStream_t stream) {
  return devapis::streamNotNull(stream);
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  return devapis::streamWaitEvent(stream, event);
}

// same as query last event status in stream.(every op has a event)
bool isStreamEmpty(deviceStream_t stream) {
  return devapis::isStreamEmpty(stream);
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) { return getEventFromPool(*event); }

void destroyEvent(deviceEvent_t event) { return restoreEventToPool(event); }

void waitEvent(deviceEvent_t event) { return devapis::waitEvent(event); }

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  return devapis::recordEvent(event, stream);
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end) {
  return devapis::eventElapsedTime(time, start, end);
}

EventStatus getEventStatus(deviceEvent_t event) {
  return devapis::getEventStatus(event);
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
  return devapis::mallocHost(p, nbytes);
}

void freeHost(void* p) { return devapis::freeHost(p); }

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  return devapis::mallocDevice(p, nbytes, throwExcepion);
}

void freeDevice(void* p) { return devapis::freeDevice(p); }

bool isPinnedPtr(const void* p) { return devapis::isPinnedPtr(p); }

// (asynchronous) set val
void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  return devapis::memSetAsync(stream, ptr, val, size);
}

// (synchronous) copy from device to a device
void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  return devapis::memCopyD2D(nbytes, dstDevId, dst, srcDevId, src);
}

// (synchronous) copy from host to a device
void memCopyH2D(size_t nbytes, /*deviceId_t dstDevId,*/ void* dst,
                /*Host srcDev,*/ const void* src) {
  return devapis::memCopyH2D(nbytes, dst, src);
}

// (synchronous) copy from a device to host
void memCopyD2H(size_t nbytes, /*Host dstDev,*/ void* dst,
                /*deviceId_t srcDevId,*/ const void* src) {
  return devapis::memCopyD2H(nbytes, dst, src);
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  return devapis::memCopyD2DAsync(stream, nbytes, dstDevId, dst, srcDevId, src);
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes,
                     /*deviceId_t dstDevId,*/ void* dst,
                     /*Host srcDev,*/ const void* src) {
  return devapis::memCopyH2DAsync(stream, nbytes, dst, src);
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes,
                     /*Host dstDev,*/ void* dst,
                     /*deviceId_t srcDevId,*/ const void* src) {
  return devapis::memCopyD2HAsync(stream, nbytes, dst, src);
}

}  // end namespace devproxy
}  // end namespace dipu
