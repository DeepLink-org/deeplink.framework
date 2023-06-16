// Copyright (c) 2023, DeepLink.
#include "DIPUGuardImpl.h"

namespace dipu {

// use c10::DeviceGuard/OptionalDeviceGuard device_guard(device_of(tensor))
// or c10::StreamGuard/OptionalStreamGuard(c10::Stream) will use DIPUGuardImpl automatically.
constexpr at::DeviceType DIPUGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(DIPU_DEVICE_TYPE_MACRO, DIPUGuardImpl);


}  // namespace torch_dipu
