// Copyright (c) 2023, DeepLink.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define DIPU_CALLCUDA(Expr) \
{ \
    cudaError_t ret = Expr; \
    if (ret != ::cudaSuccess) { \
        throw std::runtime_error("dipu device error");  \
    } \
}

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

}





