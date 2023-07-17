// Copyright (c) 2023, DeepLink.
#pragma once

#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

namespace dipu {

void getEventFromPool(deviceEvent_t& event);
void restoreEventToPool(deviceEvent_t& event);
void releaseGlobalEventPool();

}  // namespace dipu

