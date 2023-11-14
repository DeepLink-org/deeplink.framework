// Copyright (c) 2023, DeepLink.
#pragma once
#include <succl.h>
#include <supa.h>

#include <csrc_dipu/common.h>

namespace dipu {
#define deviceDefaultStreamLiteral nullptr;

using deviceStream_t = suStream_t;
using deviceEvent_t = suEvent_t;
using deviceHandle_t = suContext *;
using deviceId_t = int;

using diclComm_t = succlComm_t;
using commUniqueId = succlUniqueId;
// using ReduceOp = succlRedOp_t;
}  // namespace dipu
