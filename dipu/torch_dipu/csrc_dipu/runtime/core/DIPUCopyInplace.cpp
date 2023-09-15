// Copyright (c) 2023, DeepLink.
#include "DIPUCopyInplace.h"

#include <algorithm>
#include <c10/util/Exception.h>

#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/helpfunc.hpp>
#include <csrc_dipu/aten/DIPUATenFunctions.h>

namespace dipu {

at::Tensor& DIPUCopyInplace::run(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  c10::optional<at::DimnameList> names = src.opt_names();
  if (names.has_value()) {
    internal_set_names_inplace(self, names);
  }

  if (self.numel() == 0 || self.is_same(src)) {
    return self;
  }

  // Exit early if self and src are views of the same data
  const bool is_same_data = (
      self.is_alias_of(src) &&
      self.storage_offset() == src.storage_offset() &&
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type()
    );
  if (is_same_data) {
    return self;
  }

  auto iter = at::TensorIteratorConfig()
    .add_output(self)
    .add_input(src)
    .resize_outputs(false)
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();
  if (iter.numel() == 0) {
    return self;
  }

  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  // 1. copy between devices
  if (dst_device.type() == DIPU_DEVICE_TYPE && src_device.type() == DIPU_DEVICE_TYPE) {
    return copy_between_devices(iter, self, src, non_blocking);
  }

  // 2. copy between cpu and device, same dtype and shape, contiguous
  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    return copy_contiguous(iter, self, src, non_blocking);
  }

  // 3. copy between cpu and device, different dtype or view
  return copy_uncontiguous(iter, self, src, non_blocking);
}

at::Tensor& DIPUCopyInplace::copy_between_devices(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  int64_t numel = iter.numel();
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);

  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool memcpy_eligible = same_type && iter.is_contiguous();
  if (!memcpy_eligible) {
    return native::DIPUATenFunctions::copy_(self, src, non_blocking);
  }

  void *dst_ptr = iter.data_ptr(0);
  void *src_ptr = iter.data_ptr(1);
  if (src_ptr == dst_ptr && src_device == dst_device) {
    return self;
  }

  size_t size = numel * iter.element_size(0);
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  dipu::devproxy::memCopyD2DAsync(stream.rawstream(), size, dst_device.index(), dst_ptr, src_device.index(), src_ptr);

  if (!non_blocking) {
    dipu::devproxy::syncStream(stream.rawstream());
  }
  return self;
}

at::Tensor& DIPUCopyInplace::copy_contiguous(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  dipu::OptionalDIPUGuard device_guard(dst_device.is_cuda() ? dst_device : src_device);

  int64_t nbytes = iter.numel() * iter.element_size(0);
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  if (dst_device.type() == DIPU_DEVICE_TYPE && src_device.is_cpu()) {
    dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, iter.data_ptr(0), iter.data_ptr(1));
  } else if (dst_device.is_cpu() && src_device.type() == DIPU_DEVICE_TYPE) {
    dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, iter.data_ptr(0), iter.data_ptr(1));
  } else {
    TORCH_CHECK(false, "unsupported devices in copy_");
  }

  if (!non_blocking) {
    dipu::devproxy::syncStream(stream.rawstream());
  }
  return self;
}

at::Tensor& DIPUCopyInplace::copy_uncontiguous(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  auto& dst = iter.tensor(0);
  at::Tensor dst_contig;
  at::Tensor src_contig;
  if (iter.device_type(0) == DIPU_DEVICE_TYPE || non_blocking) {
    dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
  } else {
    bool same_type = iter.dtype(0) == iter.dtype(1);
    dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    src_contig = iter.tensor(1).expand_as(dst).contiguous();
  }
  // perform a same-dtype copy on contiguous tensors
  TORCH_CHECK(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_CHECK(dst_contig.scalar_type() == src_contig.scalar_type());
  dst_contig.copy_(src_contig, non_blocking);

  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_CHECK(dst_contig.device() == dst.device());
    dst.copy_(dst_contig, non_blocking);
  }
  return self;
}

static DIPUCopyInplace default_copy_inplace_op;
static DIPUCopyInplace *dipu_copy_inplace_op = nullptr;

DIPUCopyInplace* getDipuCopyInplace() {
  TORCH_CHECK(dipu_copy_inplace_op, "dipu copy inplace not registered");
  return dipu_copy_inplace_op;
}

void setDipuCopyInplace(DIPUCopyInplace *op) {
  if (dipu_copy_inplace_op == nullptr) {
    dipu_copy_inplace_op = op;
  } else if (dipu_copy_inplace_op == &default_copy_inplace_op) {
    dipu_copy_inplace_op = op;
  }
}

static int32_t default_init = []() {
  setDipuCopyInplace(&default_copy_inplace_op);
  return 1;
}();

}  // namespace dipu
