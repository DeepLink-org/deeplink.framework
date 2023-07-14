// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace dipu::native {

struct DIPUATenFunctions {

    // dipu native func
    static at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt);

    static at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt);

    static at::Tensor& copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking);

    static const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);

    static at::Scalar _local_scalar_dense_dipu(const at::Tensor& self);

    static at::Tensor& set_storage_dipu_(at::Tensor& result, c10::Storage storage, int64_t storage_offset,
                                         at::IntArrayRef size, at::IntArrayRef stride);
    static at::Tensor& set_dipu_(at::Tensor& self);

    static void resize_bytes_dipu(c10::StorageImpl* storage, size_t newsize_bytes);

    static bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device);
    static at::Tensor _pin_memory(const at::Tensor& self, c10::optional<at::Device> device);
    
    // todo:: use same format as autogen
    // diopi function defined in AutoGenedKernels.cpp, 
};

}  // namespace dipu::native
