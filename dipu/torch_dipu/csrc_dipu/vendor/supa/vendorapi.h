// Copyright (c) 2023, DeepLink.
#pragma once
#include <supa.h>
#include <succl.h>
#include <csrc_dipu/common.h>

namespace dipu
{
    using deviceStream_t = suStream_t;
    #define deviceDefaultStreamLiteral nullptr;
    using deviceEvent_t = suEvent_t;
    using deviceHandle_t = suContext*;
    using deviceId_t = int;

    using diclComm_t = succlComm_t;
    using commUniqueId = succlUniqueId;
    using ReduceOp = succlRedOp_t;
}
