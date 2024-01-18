// Copyright (c) 2023, DeepLink.
#pragma once

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

// #include <csrc_dipu/vendor/native_memory_format.hpp>  // IWYU pragma: export

namespace dipu {

#define TRACK_ACL(x)                                               \
  {                                                                \
    static bool enable = std::getenv("DIPU_TRACK_ACL") != nullptr; \
    if (enable) {                                                  \
      printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);              \
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

enum class NativeMemoryFormat_t : int64_t {
  UNDEFINED = -1,
  NCHW = 0,
  NHWC = 1,
  ND = 2,
  NC1HWC0 = 3,
  FRACTAL_Z = 4,
  NC1HWC0_C04 = 12,
  HWCN = 16,
  NDHWC = 27,
  FRACTAL_NZ = 29,
  NCDHW = 30,
  NDC1HWC0 = 32,
  FRACTAL_Z_3D = 33
};

}  // namespace dipu
