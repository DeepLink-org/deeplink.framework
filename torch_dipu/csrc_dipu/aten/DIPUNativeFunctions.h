#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>

namespace torch_dipu {
namespace native {
  at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt);

  at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
      c10::optional<bool> pin_memory_opt);

  at::Tensor& copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking);
  

} //end ns native
} //end ns torch_dipu
