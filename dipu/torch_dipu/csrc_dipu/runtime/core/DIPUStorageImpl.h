// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>

#include <diopi/diopirt.h>

namespace dipu {
class DIPUStorageImpl : public c10::StorageImpl {
 public:
  explicit DIPUStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes,
                           at::Allocator* allocator, bool resizable);
  ~DIPUStorageImpl() override = default;

  void release_resources() override;

  void init_desc(c10::IntArrayRef size,
                 c10::optional<at::MemoryFormat> memory_format_opt);

  void get_desc(diopiStorageDesc_t* desc) const;

  void set_desc(const diopiStorageDesc_t& desc);

  static DIPUStorageImpl* GetImplPtr(const at::Tensor& tensor);
  static DIPUStorageImpl* GetImplPtr(const at::Tensor* tensor);

 private:
  c10::impl::SizesAndStrides storage_sizes_;
  diopiMemoryFormat_t format_ = diopiMemoryFormat_t::Undefined;
};
}  // namespace dipu
