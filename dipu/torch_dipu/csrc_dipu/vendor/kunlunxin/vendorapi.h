#pragma once
#include <bkcl.h>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace xdnn = baidu::xpu::api;
namespace dipu {

#define DIPU_CALLKLX_ERROR(Expr) \
  { throw std::runtime_error(#Expr); }

#define DIPU_CALLKLX(Expr)                                           \
  {                                                                  \
    int ret = (Expr);                                                \
    TORCH_CHECK(ret == XPU_SUCCESS, "call ku error, expr = ", #Expr, \
                ", ret = ", ret);                                    \
  }

using deviceId_t = int;
using deviceStream_t = XPUStream;
#define deviceDefaultStreamLiteral nullptr
using deviceEvent_t = XPUEvent;
using deviceHandle_t = xdnn::Context*;
using diclComm_t = BKCLContext_t;
using commUniqueId = BKCLUniqueId;

}  // namespace dipu
