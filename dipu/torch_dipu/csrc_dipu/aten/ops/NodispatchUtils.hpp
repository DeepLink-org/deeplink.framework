// Copyright (c) 2023, DeepLink.
//
// This file contains useful wrappers for DIPU ATen functions.
// You should use `nodispatch::foo` instead of calling `at::foo` whenever
// possible to avoid dispatch overhead.

#pragma once

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"

namespace dipu {
namespace native {
namespace nodispatch {
// add any other `at::foo` functions you need here

// an equivalent to `at::empty` but without dispatch
inline at::Tensor empty(
    at::IntArrayRef size, at::TensorOptions options = {},
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  return dipu_aten::empty(
      size, c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(), options.pinned_memory_opt(),
      c10::impl::check_tensor_options_and_extract_memory_format(options,
                                                                memory_format));
}

// The code that calls this overloaded function is all for allocating CPU memory
inline at::Tensor empty_cpu(
    at::IntArrayRef size, at::ScalarType dtype,
    c10::optional<at::Device> device_opt = c10::nullopt,
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  return dipu_aten::empty_cpu(size, dtype, at::Layout::Strided,
                              device_or_default(device_opt), false,
                              c10::get_contiguous_memory_format());
}

inline at::Tensor empty_like(
    const at::Tensor& self, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> optional_memory_format) {
  at::TensorOptions options_ = at::TensorOptions()
                                   .dtype(dtype)
                                   .layout(layout)
                                   .device(device)
                                   .pinned_memory(pin_memory);
  at::TensorOptions options =
      self.options().merge_in(options_).merge_memory_format(
          optional_memory_format);

  TORCH_CHECK(!(options.layout() != c10::kStrided &&
                optional_memory_format.has_value()),
              "memory format option is only supported by strided tensors");

  auto memory_format =
      options.memory_format_opt().value_or(at::MemoryFormat::Preserve);

  at::Tensor result;

  if (memory_format == at::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      result = at::empty_strided_symint(self.sym_sizes(), self.sym_strides(), options.memory_format(c10::nullopt));
    } else if (self.unsafeGetTensorImpl()->support_as_strided() && self.layout() == c10::kStrided) {
      // If input tensor is not dense and non-overlapping but strided, we will infer an output strides
      // which keeps the layout permutation of the input tensor.
      std::vector<int64_t> strides = at::infer_dense_strides(self.sizes(), self.strides());
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty_strided(self.sizes(), strides, options.memory_format(c10::nullopt));
    } else {
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty_symint(self.sym_sizes(), options.memory_format(self.suggest_memory_format()), c10::nullopt);
    }
  } else {
    // See Note [Explicit nullopt MemoryFormat argument]
    result = at::empty_symint(self.sym_sizes(), options.memory_format(memory_format), c10::nullopt);
  }

  if (self.opt_names()) {
    at::namedinference::propagate_names(result, self.names());
  }

  // never propagate Conjugate, Negative, and ZeroTensor dispatch key
  result._set_conj(false);
  result._set_neg(false);
  result._set_zero(false);
  return result;
}

// an simplified version of `at::empty_like` but without dispatch
inline at::Tensor empty_like(
    const at::Tensor& self, at::TensorOptions options = {},
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  return nodispatch::empty_like(
      self, c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(), options.pinned_memory_opt(),
      c10::impl::check_tensor_options_and_extract_memory_format(options,
                                                                memory_format));
}

}  // namespace nodispatch
}  // namespace native
}  // namespace dipu
