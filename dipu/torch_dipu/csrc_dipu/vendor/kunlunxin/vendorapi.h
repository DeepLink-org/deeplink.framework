#pragma once
#include <c10/util/Exception.h>
#include <csrc_dipu/common.h>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;
namespace dipu {

#define DIPU_CALLXPU(Expr)                                             \
  {                                                                    \
    int ret = (Expr);                                                  \
    if (ret != 0) {                                                    \
      printf("call a xpu function (%s) failed. return code=%d", #Expr, \
              ret);                                                    \
      throw std::runtime_error("dipu device error");                   \
    }                                                                  \
  }

using deviceId_t = int;
using deviceStream_t = XPUStream;
#define deviceDefaultStreamLiteral nullptr
using deviceEvent_t = XPUEvent;
using deviceHandle_t = xdnn::Context*;

class pcclComm_t {};
using diclComm_t = pcclComm_t*;
class pcclUniqueId {};
using commUniqueId = pcclUniqueId;

}  // namespace dipu
