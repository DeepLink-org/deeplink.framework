// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstddef>
#include <cstdint>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Optional.h>

namespace dipu {
namespace native {
namespace dipu_aten {
// dipu native func
at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
                 c10::optional<at::Layout> layout_opt,
                 c10::optional<at::Device> device_opt,
                 c10::optional<bool> pin_memory_opt,
                 c10::optional<at::MemoryFormat> memory_format_opt);
at::Tensor empty_cpu(at::IntArrayRef size,
                     c10::optional<at::ScalarType> dtype_opt,
                     c10::optional<at::Layout> layout_opt,
                     c10::optional<at::Device> device_opt,
                     c10::optional<bool> pin_memory_opt,
                     c10::optional<at::MemoryFormat> memory_format_opt);

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt,
                         c10::optional<at::Device> device_opt,
                         c10::optional<bool> pin_memory_opt);
at::Tensor empty_strided_cpu(at::IntArrayRef size, at::IntArrayRef stride,
                             c10::optional<at::ScalarType> dtype_opt,
                             c10::optional<at::Layout> layout_opt,
                             c10::optional<at::Device> device_opt,
                             c10::optional<bool> pin_memory_opt);

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size,
                          c10::optional<at::MemoryFormat> memory_format);

at::Scalar _local_scalar_dense_dipu(const at::Tensor& self);

at::Tensor& set_storage_dipu_(at::Tensor& result, c10::Storage storage,
                              int64_t storage_offset, at::IntArrayRef size,
                              at::IntArrayRef stride);
at::Tensor& set_dipu_(at::Tensor& self);

void resize_bytes_dipu(c10::StorageImpl* storage, size_t newsize_bytes);

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device);
at::Tensor _pin_memory(const at::Tensor& self,
                       c10::optional<at::Device> device);

// todo:: use same format as autogen
// diopi function defined in AutoGenedKernels.cpp,
};  // namespace dipu_aten

}  // namespace native
}  // namespace dipu
