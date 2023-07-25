// Copyright (c) 2023, DeepLink.
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {

namespace device {

DIPU_API dipu::devapis::DIPUDeviceProperties* getDevicePropertiesFromCache(int32_t device_index);

}  // namespace device
}  // namespace dipu