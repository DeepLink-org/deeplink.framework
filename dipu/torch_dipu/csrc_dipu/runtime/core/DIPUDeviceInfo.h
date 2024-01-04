// Copyright (c) 2023, DeepLink.

#include <cstdint>
#include <memory>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu {
using dipu::devapis::DIPUDeviceProperties;
using dipu::devapis::DIPUDeviceStatus;

DIPU_API std::shared_ptr<DIPUDeviceProperties> getDevicePropertiesFromCache(
    int32_t device_index);
DIPU_API std::shared_ptr<DIPUDeviceStatus> getDeviceStatus(
    int32_t device_index);

}  // namespace dipu
