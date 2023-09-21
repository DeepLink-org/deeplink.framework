// Copyright (c) 2023, DeepLink.
#include <deque>
#include <vector>

#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>

#include "./device.h"

namespace dipu {

namespace device {

using dipu::devapis::DIPUDeviceProperties;
using c10::DeviceIndex;

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<std::mutex> device_mutexs;
std::vector<std::shared_ptr<DIPUDeviceProperties>> device_properties;

static void initDIPUContextVectors() {
  num_gpus = dipu::devproxy::getDeviceCount();
  device_mutexs.resize(num_gpus);
  device_properties.resize(num_gpus);
}

std::shared_ptr<DIPUDeviceProperties> getDevicePropertiesFromCache(int32_t device_index, bool force_update) {
  c10::call_once(init_flag, initDIPUContextVectors);
  if (device_index == -1) {
    device_index = dipu::devproxy::current_device();
  }
  AT_ASSERT(device_index >= 0 && device_index < num_gpus);

  std::lock_guard<std::mutex> lk(device_mutexs[device_index]);
  if (force_update || device_properties[device_index] == nullptr) {
    DIPUDeviceProperties device_prop = dipu::devproxy::getDeviceProperties(device_index);
    device_properties[device_index] = std::make_shared<DIPUDeviceProperties>(device_prop);
  }
  return device_properties[device_index];
}

}  // namespace device
}  // namespace dipu