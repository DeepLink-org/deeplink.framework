// Copyright (c) 2023, DeepLink.
#include <cstdint>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_native.h>
#include <c10/core/CompileTimeFunctionPointer.h>

#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/base/basedef.h"

namespace at {

namespace {
at::Tensor wrapper_DIPU__cudnn_rnn_flatten_weight(
    at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
    int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers,
    bool batch_first, bool bidirectional) {
  return at::native::_cudnn_rnn_flatten_weight(
      weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size,
      num_layers, batch_first, bidirectional);
}

}  // namespace

DIPU_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  m.impl("_cudnn_rnn_flatten_weight",
         TORCH_FN(wrapper_DIPU__cudnn_rnn_flatten_weight));
}

}  // namespace at
