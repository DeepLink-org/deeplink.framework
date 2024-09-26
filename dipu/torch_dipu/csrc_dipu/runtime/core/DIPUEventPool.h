// Copyright (c) 2023, DeepLink.
#pragma once

// TODO(vendor) - should deviceEvent_t be provided by vendor?
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"  // IWYU pragma: keep

namespace dipu {

void event_pool_acquire(int index, deviceEvent_t& event);
void event_pool_release(int index, deviceEvent_t& event);
void event_pool_clear();

}  // namespace dipu
