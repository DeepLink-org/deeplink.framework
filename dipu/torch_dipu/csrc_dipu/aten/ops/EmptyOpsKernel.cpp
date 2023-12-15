// Copyright (c) 2023, DeepLink.
#include <ATen/EmptyTensor.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/accumulate.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/profiler/profiler.h>
#include <csrc_dipu/runtime/core/DIPUStorageImpl.h>
#include <csrc_dipu/runtime/rthelper.h>

namespace dipu {
namespace native {

static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return dipu::getAllocator(at::DeviceType::CPU);
  }
  return c10::GetCPUAllocator();
}

at::Tensor DIPUATenFunctions::empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::device_or_default(device_opt).type() ==
                                   dipu::DIPU_DEVICE_TYPE);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(c10::layout_or_default(layout_opt) ==
                                   c10::Layout::Strided);
  at::detail::check_size_nonnegative(size);
  c10::Allocator* allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
  auto dtype = c10::scalarTypeToTypeMeta(c10::dtype_or_default(dtype_opt));
  auto size_bytes =
      at::detail::computeStorageNbytesContiguous(size, dtype.itemsize());
  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<dipu::DIPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, true);
  constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage_impl), dipu_ks, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }
  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != c10::MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }
  DIPUStorageImpl::GetImplPtr(tensor)->init_desc(size, memory_format_opt);
  return tensor;
}

at::Tensor DIPUATenFunctions::empty_cpu(
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
  return at::detail::empty_generic(size, allocator, cpu_ks, dtype,
                                   memory_format_opt);
}

at::Tensor DIPUATenFunctions::empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  auto device = c10::device_or_default(device_opt);
  AT_ASSERT(device.type() == dipu::DIPU_DEVICE_TYPE);
  AT_ASSERT(c10::layout_or_default(layout_opt) == at::Layout::Strided);
  at::detail::check_size_nonnegative(size);
  auto scalar_type = dtype_or_default(dtype_opt);
  c10::Allocator* allocator = dipu::getAllocator(dipu::DIPU_DEVICE_TYPE);
  constexpr c10::DispatchKeySet dipu_ks({dipu::DIPU_DISPATCH_KEY});
  caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(scalar_type);
  auto size_bytes =
      at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<dipu::DIPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, true);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage_impl), dipu_ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  DIPUStorageImpl::GetImplPtr(tensor)->init_desc(size, c10::nullopt);
  return tensor;
}

at::Tensor DIPUATenFunctions::empty_strided_cpu(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
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
