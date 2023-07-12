// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace dipu {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

}  // namespace dipu
