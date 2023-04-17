#include <deque>
#include <vector>

#include <csrc_dipu/runtime/core/device.h>

#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>

namespace dipu {

namespace device {

using dipu::devapis::DIPUDeviceProperties;
using c10::DeviceIndex;

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<DIPUDeviceProperties> device_properties;

static void initDIPUContextVectors() {
  num_gpus = dipu::devapis::getDeviceCount();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

static void initDeviceProperty(DeviceIndex device_index) {
  DIPUDeviceProperties device_prop = dipu::devapis::getDeviceProperties(device_index);
  device_properties[device_index] = device_prop;
}

DIPUDeviceProperties* getDevicePropertiesFromCache(int32_t device_index) {
  c10::call_once(init_flag, initDIPUContextVectors);
  if (device_index == -1) {
    device_index = dipu::devapis::current_device();
  }
  AT_ASSERT(device_index >= 0 && device_index < num_gpus);

  c10::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

}  // namespace device
}  // namespace dipu