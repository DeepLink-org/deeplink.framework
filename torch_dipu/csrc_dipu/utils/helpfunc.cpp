// Copyright (c) 2023, DeepLink.
#include "./helpfunc.hpp"

namespace dipu {
bool isDeviceTensor(const at::Tensor &tensor) { return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE; }
}  // namespace dipu