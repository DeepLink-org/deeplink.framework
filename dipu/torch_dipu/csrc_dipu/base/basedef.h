// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Utils.h>
#include <c10/core/DispatchKey.h>

#include "csrc_dipu/runtime/device/basedef.h"

#define DIPU_DEVICE_TYPE_MACRO XPU
#define DIPU_AUTOGRAD_DEVICE_TYPE_MACRO \
  C10_CONCATENATE(Autograd, DIPU_DEVICE_TYPE_MACRO)
#define DIPU_AUTOCAST_DEVICE_TYPE_MACRO \
  C10_CONCATENATE(Autocast, DIPU_DEVICE_TYPE_MACRO)
#define DIPU_SPARSE_DEVICE_TYPE_MACRO \
  C10_CONCATENATE(Sparse, DIPU_DEVICE_TYPE_MACRO)

// to do: abstract a layer which not depend on pytorch
namespace dipu {

// XPU is originally intel output-of-tree code
// https://github.com/intel/intel-extension-for-pytorch ( branch xpu-master )
// we use this type but PrivateUse1 not to impersonate our DIPU device.
// because compared with PrivateUse1, XPU has richer support in pytorch trunk
// and not too much feature in torch to interfere our logic (as XLA).
const auto DIPU_DEVICE_TYPE = at::DeviceType::DIPU_DEVICE_TYPE_MACRO;

const auto DIPU_DISPATCH_KEY = c10::DispatchKey::DIPU_DEVICE_TYPE_MACRO;
const auto DIPU_DISPATCH_AUTOGRAD_KEY =
    c10::DispatchKey::DIPU_AUTOGRAD_DEVICE_TYPE_MACRO;

const auto DIPU_BACKEND_TYPE = c10::Backend::DIPU_DEVICE_TYPE_MACRO;
const auto DIPU_BACKEND_SPARSE_TYPE =
    c10::Backend::DIPU_SPARSE_DEVICE_TYPE_MACRO;

const auto DICL_BACKEND_NAME = "dicl";

}  // namespace dipu
