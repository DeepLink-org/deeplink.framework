// Copyright (c) 2023, DeepLink.
#pragma once

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

using deviceStream_t = aclrtStream;
#define deviceDefaultStreamLiteral nullptr;
using deviceEvent_t = aclrtEvent;
using deviceHandle_t = aclrtContext*;

using diclComm_t = HcclComm;
using commUniqueId = HcclRootInfo;
}  // namespace dipu
