// Copyright (c) 2023, DeepLink.
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {
namespace device {
using dipu::devapis::DIPUDeviceProperties;

DIPU_API std::shared_ptr<DIPUDeviceProperties> getDevicePropertiesFromCache(int32_t device_index, bool force_update);

}  // namespace device
}  // namespace dipu