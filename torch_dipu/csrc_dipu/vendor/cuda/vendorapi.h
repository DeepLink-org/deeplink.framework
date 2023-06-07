// Copyright (c) 2023, DeepLink.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define DIPU_CALLCUDA(Expr)                                                     \
{                                                                               \
    cudaError_t ret = Expr;                                                     \
    if (ret != ::cudaSuccess) {                                                 \
        TORCH_CHECK(false, "call cuda error, expr = ", #Expr, ", ret = ", ret); \
    }                                                                           \
}

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

using diclComm_t = ncclComm_t;
using commUniqueId = ncclUniqueId;

}





