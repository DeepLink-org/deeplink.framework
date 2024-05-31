// Copyright (c) 2023, DeepLink.
#include <ATen/EmptyTensor.h>
#include <ATen/core/ATen_fwd.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/accumulate.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/rthelper.h"

namespace dipu {
namespace native {

static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return dipu::getAllocator(at::DeviceType::CPU);
  }
  return c10::GetCPUAllocator();
}

template <typename T>
at::TensorBase _empty_generic(
    at::ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    at::ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  at::detail::check_size_nonnegative(size);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = at::detail::computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = at::detail::make_tensor_base<dipu::native::dipu_aten::dnativeTensorImpl>(
      std::move(storage_impl), ks, dtype);
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (ks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }
  
  if (memory_format_opt.has_value()) {
      // Restriding a just-created empty contiguous tensor does nothing.
      if (*memory_format_opt != c10::MemoryFormat::Contiguous) {
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
      }
  }
  return tensor;
}

at::TensorBase empty_generic(
    at::IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    at::ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}


template <typename T>
at::TensorBase _empty_strided_generic(
    T size,
    T stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    at::ScalarType scalar_type) {
  at::detail::check_size_nonnegative(size);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      true);

  auto tensor = at::detail::make_tensor_base<dipu::native::dipu_aten::dnativeTensorImpl>(
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

at::TensorBase empty_strided_generic(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    at::ScalarType scalar_type) {
  return _empty_strided_generic<at::IntArrayRef>(size, stride, allocator, ks, scalar_type);
}

at::Tensor dipu_aten::empty(at::IntArrayRef size,
                            c10::optional<at::ScalarType> dtype_opt,
                            c10::optional<at::Layout> layout_opt,
                            c10::optional<at::Device> device_opt,
                            c10::optional<bool> pin_memory_opt,
                            c10::optional<at::MemoryFormat> memory_format_opt) {
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() ==
                                   dipu::DIPU_DEVICE_TYPE);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) ==
                                   c10::Layout::Strided);

  c10::Allocator* allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
  constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
  return empty_generic(size, allocator, dipu_ks,
                                   c10::dtype_or_default(dtype_opt),
                                   memory_format_opt);
}

at::Tensor dipu_aten::empty_cpu(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() ==
                                   c10::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) ==
                                   c10::Layout::Strided);

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = c10::dtype_or_default(dtype_opt);
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype,
                                   memory_format_opt);
}

// use empty_generic, test
at::Tensor dipu_aten::empty_strided(at::IntArrayRef size,
                                    at::IntArrayRef stride,
                                    c10::optional<at::ScalarType> dtype_opt,
                                    c10::optional<at::Layout> layout_opt,
                                    c10::optional<at::Device> device_opt,
                                    c10::optional<bool> pin_memory_opt) {
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  auto device = c10::device_or_default(device_opt);
  AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
  AT_ASSERT(c10::layout_or_default(layout_opt) == at::Layout::Strided);
  auto dtype = dtype_or_default(dtype_opt);

  c10::Allocator* allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
  constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
  return empty_strided_generic(size, stride, allocator, dipu_ks,
                                           dtype);
}

at::Tensor dipu_aten::empty_strided_cpu(at::IntArrayRef size,
                                        at::IntArrayRef stride,
                                        c10::optional<at::ScalarType> dtype_opt,
                                        c10::optional<at::Layout> layout_opt,
                                        c10::optional<at::Device> device_opt,
                                        c10::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() ==
                                   c10::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) ==
                                   c10::Layout::Strided);

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = c10::dtype_or_default(dtype_opt);
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(size, stride, allocator, cpu_ks,
                                           dtype);
}

}  // namespace native
}  // namespace dipu
