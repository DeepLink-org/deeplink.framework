// Copyright (c) 2023, DeepLink.
#pragma once
#include <csrc_dipu/common.h>
#include <eccl.h>
#include <tops_runtime.h>

namespace dipu {

#define DIPU_CALLTOPSRT(Expr)                        \
  {                                                  \
    ::topsError_t ret = Expr;                        \
    if (ret != ::topsSuccess) {                      \
      throw std::runtime_error("dipu device error, ret code:" + std::to_string(ret)); \
    }                                                \
  }

using deviceStream_t = topsStream_t;
#define deviceDefaultStreamLiteral nullptr
using deviceEvent_t = topsEvent_t;

using diclComm_t = ecclComm_t;
using commUniqueId = ecclUniqueId;
}
