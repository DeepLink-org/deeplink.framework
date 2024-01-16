// Copyright (c) 2023, DeepLink.
#pragma once

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <unistd.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define TRACK_ACL(x)                                               \
  {                                                                \
    static bool enable = std::getenv("DIPU_TRACK_ACL") != nullptr; \
    if (enable) {                                                  \
      printf("[%d %s: %d]:%s\n", getpid(), __FILE__, __LINE__, x);              \
    }                                                              \
  }

#define DIPU_CALLACLRT(Expr)                                               \
  {                                                                        \
    TRACK_ACL(#Expr);                                                      \
    ::aclError ret = Expr;                                                 \
    TORCH_CHECK(ret == ACL_SUCCESS, "ascend device error, expr = ", #Expr, \
                ", ret = ", ret, ", error msg = ", aclGetRecentErrMsg());  \
  }

using deviceStream_t = aclrtStream;
#define deviceDefaultStreamLiteral nullptr;
using deviceEvent_t = aclrtEvent;
using deviceHandle_t = aclrtContext*;

using diclComm_t = HcclComm;
using commUniqueId = HcclRootInfo;
}  // namespace dipu
