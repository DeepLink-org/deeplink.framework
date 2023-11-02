// Copyright (c) 2023, DeepLink.
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/profiler/profiler.h>
#include <csrc_dipu/runtime/core/DIPUCopyInplace.h>
#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/aten/RegisterDIPU.hpp>

namespace at {

namespace {
  at::Tensor wrapper_DIPU__cudnn_rnn_flatten_weight(at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
    return at::native::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  }

}  // end of inner anonymous namespace

DIPU_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  m.impl("_cudnn_rnn_flatten_weight", TORCH_FN(wrapper_DIPU__cudnn_rnn_flatten_weight));
}

}  // namespace at
