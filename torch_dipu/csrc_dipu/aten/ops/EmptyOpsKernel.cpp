// Copyright (c) 2023, DeepLink.
#include <ATen/EmptyTensor.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/accumulate.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/runtime/rthelper.h>

using c10::device_or_default;
using c10::layout_or_default;
using c10::StorageImpl;
using c10::TensorImpl;
using at::Layout;

namespace dipu::native {

  static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
    if (pin_memory) {
      return dipu::getAllocator(at::DeviceType::CPU);
    }
    return c10::GetCPUAllocator();
  }

  // use old logic, test
  at::Tensor DIPUATenFunctions::empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt) {

    AT_ASSERT(c10::device_or_default(device_opt).type() == dipu::DIPU_DEVICE_TYPE);
    TORCH_CHECK(!pinned_memory_or_default(pin_memory_opt), "Only dense tensors can be pinned");

    c10::Allocator *allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
    // ?? do gurad in allocator or wrapper?
    const int64_t nelements = c10::multiply_integers(size);

    auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    int64_t size_bytes = nelements * dtype.itemsize();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        true);

    constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
    auto tensor = at::detail::make_tensor<TensorImpl>( std::move(storage_impl), dipu_ks, dtype);

    // Default at::TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    if (memory_format_opt.has_value()) {
      // Restriding a just-created empty contiguous tensor does nothing.
      if (*memory_format_opt != at::MemoryFormat::Contiguous) {
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
      }
    }
    return tensor;
  }

  at::Tensor DIPUATenFunctions::empty_cpu(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
      c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() == c10::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) == c10::Layout::Strided);

    auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
    auto dtype = c10::dtype_or_default(dtype_opt);
    auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
    constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
    return at::detail::empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
  }

  // use empty_generic, test
  at::Tensor DIPUATenFunctions::empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
      c10::optional<bool> pin_memory_opt) {

    auto device = c10::device_or_default(device_opt);
    AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
    AT_ASSERT(layout_or_default(layout_opt) == Layout::Strided);
    auto dtype = dtype_or_default(dtype_opt);

    c10::Allocator *allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
    constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
    return at::detail::empty_strided_generic(size, stride, allocator, dipu_ks, dtype);
  }

  at::Tensor DIPUATenFunctions::empty_strided_cpu(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
      c10::optional<bool> pin_memory_opt) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() == c10::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) == c10::Layout::Strided);

    auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
    auto dtype = c10::dtype_or_default(dtype_opt);
    auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
    constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
    return at::detail::empty_strided_generic(size, stride, allocator, cpu_ks, dtype);
  }

} //end ns dipu::native