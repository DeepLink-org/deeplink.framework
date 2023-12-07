// Copyright (c) 2023, DeepLink.
#include <ATen/core/NamedTensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/runtime/core/MemChecker.h>
#include <csrc_dipu/runtime/rthelper.h>

using at::IntArrayRef;
using c10::MemoryFormat;
using c10::StorageImpl;
using c10::TensorImpl;
using dipu::devproxy::current_device;

namespace dipu {
namespace native {
void DIPUATenFunctions::resize_bytes_dipu(StorageImpl* storage,
                                          size_t newsize_bytes) {
  TORCH_CHECK(storage->resizable(),
              "Trying to resize dipu storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr,
              "Trying to resize dipu storage without an allocator");

  auto device = current_device();
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  if (newsize_bytes == 0) {
    storage->set_data_ptr_noswap(
        at::DataPtr(nullptr, at::Device(dipu::DIPU_DEVICE_TYPE, device)));
    storage->set_nbytes(0);
    return;
  }
  size_t nbytes = std::min(storage->nbytes(), newsize_bytes);
  at::DataPtr data = allocator->allocate(newsize_bytes);  // alloc new
  if (storage->data_ptr()) {                              // copy old to new
    MemChecker::instance().check(data.get());
    MemChecker::instance().check(storage->data());
    if (storage->data() != nullptr) {
      dipu::devproxy::memCopyD2DAsync(stream.rawstream(), nbytes, device,
                                      data.get(), device, storage->data());
    }
  }
  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(newsize_bytes);
}

static inline TensorImpl* _resize_impl_dipu_(TensorImpl* self, IntArrayRef size,
                                             at::OptionalIntArrayRef stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }
  const DIPUGuard device_guard(self->device());
  // need add guard to support device change.
  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t new_storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    new_storage_size = at::detail::computeStorageNbytes(size, *stride, itemsize,
                                                        storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    new_storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  const c10::Storage& storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (self->numel() > 0 && new_storage_size > storage.nbytes()) {
    DIPUATenFunctions::resize_bytes_dipu(storage.unsafeGetStorageImpl(),
                                         new_storage_size);
  }
  return self;
}

const at::Tensor& DIPUATenFunctions::resize_(
    const at::Tensor& self, at::IntArrayRef size,
    c10::optional<at::MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return at::native::resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  // not support stride now
  _resize_impl_dipu_(self_, size, /*stride=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(memory_format != MemoryFormat::Preserve,
                "Unsupported memory format", memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

at::Tensor& DIPUATenFunctions::set_storage_dipu_(at::Tensor& result,
                                                 c10::Storage storage,
                                                 int64_t storage_offset,
                                                 at::IntArrayRef size,
                                                 at::IntArrayRef stride) {
  at::native::checkSetStorage(result, std::move(storage), storage_offset, size,
                              stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  _resize_impl_dipu_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

at::Tensor& DIPUATenFunctions::set_dipu_(at::Tensor& self) {
  caffe2::TypeMeta dtype = self.dtype();
  c10::Storage storage(c10::Storage::use_byte_size_t(), 0,
                       dipu::getAllocator(dipu::DIPU_DEVICE_TYPE), true);
  DIPUATenFunctions::set_storage_dipu_(self, storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == self.dtype());
  return self;
}
}  // namespace native
}  // namespace dipu
