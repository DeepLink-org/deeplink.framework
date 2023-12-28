// Copyright (c) 2023, DeepLink.
#include "DIPUStorageImpl.h"

#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <diopi/diopirt.h>

namespace dipu {
DIPUStorageImpl::DIPUStorageImpl(use_byte_size_t use_byte_size,
                                 size_t size_bytes, at::Allocator* allocator,
                                 bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, allocator, resizable) {}

void DIPUStorageImpl::release_resources() { StorageImpl::release_resources(); }

void DIPUStorageImpl::init_desc(
    c10::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format_opt) {
  storage_sizes_.set_sizes(size);
  if (!memory_format_opt.has_value() || *memory_format_opt == c10::MemoryFormat::Contiguous) {
    format_ = diopiMemoryFormat_t::Contiguous;
  } else if (*memory_format_opt == c10::MemoryFormat::ChannelsLast) {
    format_ = diopiMemoryFormat_t::ChannelsLast;
  } else if (*memory_format_opt == c10::MemoryFormat::ChannelsLast3d) {
    format_ = diopiMemoryFormat_t::ChannelsLast3d;
  } else {
    format_ = diopiMemoryFormat_t::Undefined;
  }
}

void DIPUStorageImpl::get_desc(diopiStorageDesc_t* desc) const {
  desc->sizes.data = storage_sizes_.sizes_data();
  desc->sizes.len = storage_sizes_.size();
  desc->format = format_;
}

void DIPUStorageImpl::set_desc(const diopiStorageDesc_t& desc) {
  storage_sizes_.set_sizes(
      c10::IntArrayRef{desc.sizes.data, static_cast<size_t>(desc.sizes.len)});
  format_ = desc.format;
}

DIPUStorageImpl* DIPUStorageImpl::GetImplPtr(const at::Tensor& tensor) {
  auto* ptr =
      dynamic_cast<DIPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl());
  TORCH_CHECK(ptr, "tensor must use DIPUStorageImpl");
  return ptr;
}

DIPUStorageImpl* DIPUStorageImpl::GetImplPtr(const at::Tensor* tensor) {
  auto* ptr =
      dynamic_cast<DIPUStorageImpl*>(tensor->storage().unsafeGetStorageImpl());
  TORCH_CHECK(ptr, "tensor must use DIPUStorageImpl");
  return ptr;
}

}  // namespace dipu
