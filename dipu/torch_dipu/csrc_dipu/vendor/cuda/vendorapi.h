// Copyright (c) 2023, DeepLink.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

// ----------------------------------------------------------------------------
// Code from pytorch2.1.1 c10/cuda/driver_api.h begin
// ----------------------------------------------------------------------------

#define DIPU_DRIVER_CHECK(EXPR)                       \
  do {                                                \
    CUresult __err = EXPR;                            \
    if (__err != ::CUDA_SUCCESS) {                    \
      const char* err_str;                            \
      CUresult get_error_str_err C10_UNUSED =         \
          cuGetErrorString(__err, &err_str);          \
      if (get_error_str_err != ::CUDA_SUCCESS) {      \
        AT_ERROR("CUDA driver error: unknown error"); \
      } else {                                        \
        AT_ERROR("CUDA driver error: ", err_str);     \
      }                                               \
    }                                                 \
  } while (0)

// ----------------------------------------------------------------------------
// Code from pytorch2.1.1 c10/cuda/driver_api.h end
// ----------------------------------------------------------------------------

#define DIPU_CALLCUDA(Expr)                                              \
  {                                                                      \
    cudaError_t ret = Expr;                                              \
    TORCH_CHECK(ret == ::cudaSuccess, "call cuda error, expr = ", #Expr, \
                ", ret = ", ret);                                        \
  }

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

using diclComm_t = ncclComm_t;
using commUniqueId = ncclUniqueId;

}  // namespace dipu
