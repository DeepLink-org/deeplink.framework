// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Utils.h>
#include <c10/core/DispatchKey.h>
#include <csrc_dipu/runtime/device/basedef.h>


#define C10_COMPILE_TIME_MAX_DIPUS 16

#define DIPU_DEVICE_TYPE_MACRO XPU
#define DIPU_AUTOGRAD_DEVICE_TYPE_MACRO AutogradXPU

#define DeviceTypeDIPU  at::DeviceType::XPU
#define AutocastDIPU  AutocastXPU

#define ADD_NS(RAW_OP) at::RAW_OP

// to do: abstract a layer which not depend on pytorch
namespace dipu {

// XPU is originally intel output-of-tree code https://github.com/intel/intel-extension-for-pytorch ( branch xpu-master )
// we use this type but PrivateUse1 not to impersonate our DIPU device. because compared with PrivateUse1,
// XPU has richer support in pytorch trunk and not too much feature in torch to interfere our logic (as XLA).
const auto DIPU_DEVICE_TYPE = at::DeviceType::XPU;

const auto DIPU_DISPATCH_KEY = c10::DispatchKey::XPU;
const auto DIPU_DISPATCH_AUTOGRAD_KEY = c10::DispatchKey::AutogradXPU;

const auto DIPU_Backend_TYPE = c10::Backend::XPU;

const auto DICL_BACKEND_NAME = "dicl";

} // end ns dipu
