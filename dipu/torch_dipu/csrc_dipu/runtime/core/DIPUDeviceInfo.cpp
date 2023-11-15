// Copyright (c) 2023, DeepLink.
#include "./DIPUDeviceInfo.h"

#include <deque>
#include <vector>

#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>
namespace dipu {

// anonymous ns
namespace {

using c10::DeviceIndex;
using dipu::devapis::DIPUDeviceProperties;
using std::shared_ptr;

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<shared_ptr<DIPUDeviceProperties>> device_properties;

static void initDIPUContextVectors() {
  num_gpus = dipu::devproxy::getDeviceCount();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

static void initDeviceProperty(DeviceIndex device_index) {
  DIPUDeviceProperties device_prop =
      dipu::devproxy::getDeviceProperties(device_index);
  device_properties[device_index] =
      std::make_shared<DIPUDeviceProperties>(device_prop);
}

static inline void checkDevice(int32_t device_index) {
  c10::call_once(init_flag, initDIPUContextVectors);
  if (device_index == -1) {
    device_index = dipu::devproxy::current_device();
  }
  AT_ASSERT(device_index >= 0 && device_index < num_gpus);
}

}  // namespace

shared_ptr<DIPUDeviceProperties> getDevicePropertiesFromCache(
    int32_t device_index) {
  checkDevice(device_index);
  c10::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return device_properties[device_index];
}

shared_ptr<DIPUDeviceStatus> getDeviceStatus(int32_t device_index) {
  checkDevice(device_index);

  // never cache status
  DIPUDeviceStatus device_prop = dipu::devproxy::getDeviceStatus(device_index);
  return std::make_shared<DIPUDeviceStatus>(device_prop);
}

}  // namespace dipu