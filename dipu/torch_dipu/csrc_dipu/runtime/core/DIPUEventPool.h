// Copyright (c) 2023, DeepLink.
#pragma once

// TODO(vendor) - should deviceEvent_t be provided by vendor?
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"  // IWYU pragma: keep

namespace dipu {

void getEventFromPool(deviceEvent_t& event);

void restoreEventToPool(deviceEvent_t& event);

void releaseAllEvent();

}  // namespace dipu
