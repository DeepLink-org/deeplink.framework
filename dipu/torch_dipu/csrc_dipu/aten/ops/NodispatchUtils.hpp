// Copyright (c) 2023, DeepLink.
//
// This file contains useful wrappers for DIPU ATen functions.
// You should use `nodispatch::foo` instead of calling `at::foo` whenever
// possible to avoid dispatch overhead.

#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
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
  if (c10::device_or_default(options.device_opt()).type() ==
      c10::DeviceType::CPU) {
    return dipu_aten::empty_cpu(
        size, c10::optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(), options.device_opt(), options.pinned_memory_opt(),
        c10::impl::check_tensor_options_and_extract_memory_format(
            options, memory_format));
  } else {
    return dipu_aten::empty(
        size, c10::optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(), options.device_opt(), options.pinned_memory_opt(),
        c10::impl::check_tensor_options_and_extract_memory_format(
            options, memory_format));
  }
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

// an simplified version of `at::empty_like` but without dispatch
inline at::Tensor empty_like(
    const at::Tensor& self, at::TensorOptions options = {},
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  if (!options.has_device()) {
    return empty(self.sizes(), self.options(), memory_format);
  } else {
    return empty(self.sizes(), options, memory_format);
  }
}

// In the current code, this interface has only been called once, including
// dtype, layout, device, and pin_memoriy are all c10::nullopt, so just ignore
// these parameters
inline at::Tensor empty_like(const at::Tensor& self,
                             c10::optional<at::ScalarType> dtype,
                             c10::optional<at::Layout> layout,
                             c10::optional<at::Device> device,
                             c10::optional<bool> pin_memory,
                             c10::optional<at::MemoryFormat> memory_format) {
  return empty(self.sizes(), self.options(), memory_format);
}

}  // namespace nodispatch
}  // namespace native
}  // namespace dipu
