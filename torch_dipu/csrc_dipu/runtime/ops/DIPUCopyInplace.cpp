// Copyright (c) 2023, DeepLink.

#include "DIPUCopyInplace.h"

#include <algorithm>
#include <c10/util/Exception.h>

#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/helpfunc.hpp>

namespace dipu {

inline int printCopyLog() {
    const char* ptr = std::getenv("DIPU_PRINT_COPY_LOG");
    int val = ptr ? std::atoi(ptr) : 0;
    return val;
}

static int print_copy_log = printCopyLog();

#define DEBUG_LOG if (print_copy_log > 0) std::cout

int64_t getCopyBytes(const at::Tensor& dst, const at::Tensor& src) {
  // outer bytes must same. different type is unsuported
  TORCH_CHECK(dst.nbytes() == src.nbytes(), "dipu copy with different size is not allowed");

  int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
  int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
  return std::min(dstBytes, srcBytes);
}

void copy_from_device_to_host(at::Tensor &dst, const at::Tensor &src, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
  int64_t nbytes = getCopyBytes(dst, src);
  DIPUGuard guard(src.device());
  DIPUStream stream = dipu::getCurrentDIPUStream();

  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();
  MemChecker::instance().check(src);
  dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);

  if (non_blocking) {
    // TODO(caikun): remove syncStream when cache allocator is ready
    dipu::devproxy::syncStream(stream.rawstream());
  } else {
    dipu::devproxy::syncStream(stream.rawstream());
  }
}

void copy_from_host_to_device(at::Tensor &dst, const at::Tensor &src, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
  int64_t nbytes = getCopyBytes(dst, src);
  DIPUGuard guard(dst.device());
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();
  MemChecker::instance().check(dst);
  dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);

  if (non_blocking) {
    // TODO(caikun): remove syncStream when cache allocator is ready
    dipu::devproxy::syncStream(stream.rawstream());
  } else {
    dipu::devproxy::syncStream(stream.rawstream());
  }
}

void dumpTensor(const at::Tensor &tensor) {
  DEBUG_LOG << "numel: " << tensor.numel() << ", sizes: " << tensor.sizes() << ", stride: " << tensor.strides() << ", is_view: " << tensor.is_view() << ", dtype: " << tensor.dtype()
        << ", device:" << tensor.device() << ", layout:" << tensor.layout() << ", requires_grad: " << (tensor.requires_grad() ? "true" : "false") << ", pinned_memory: " << (tensor.is_pinned() ? "true" : "false") 
        << ", memory_format: "  << tensor.suggest_memory_format() << ", data_ptr: " << tensor.data_ptr() << std::endl;
}

at::Tensor& DIPUCopyInplace::run(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  DEBUG_LOG << "enter into DIPUCopyInplace" << std::endl;

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

  dumpTensor(self);
  dumpTensor(src);

  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  // 1. copy between devices
  if (dst_device.type() == DIPU_DEVICE_TYPE && src_device.type() == DIPU_DEVICE_TYPE) {
    copy_between_devices(self, src, iter, non_blocking);
    return self;
  }

  // 2. copy between cpu and device, same dtype and shape, contiguous
  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // TODO(caikun): optimize it, better function name
    copy_same_dtype(self, src, iter, non_blocking);
    return self;
  }

  // 3. copy between cpu and device, different dtype or view
  copy_between_host_device(self, src, iter, non_blocking);
  return self;
}

void DIPUCopyInplace::copy_between_devices(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
  int64_t numel = iter.numel();
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);

  bool same_type = iter.dtype(0) == iter.dtype(1);
  // TODO(caikun): same dtype and shape? check what does is_contiguous actual mean!!!! 
  bool memcpy_eligible = same_type && iter.is_contiguous();
  if (!memcpy_eligible) {
    slow_copy(self, src, non_blocking);
    return;
  }

  void *dst_ptr = iter.data_ptr(0);
  void *src_ptr = iter.data_ptr(1);
  if (src_ptr == dst_ptr && src_device == dst_device) {
    return;
  }

  size_t size = numel * iter.element_size(0);
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  dipu::devapis::memCopyD2DAsync(stream.rawstream(), size, dst_device.index(), dst_ptr, src_device.index(), src_ptr);

  if (non_blocking) {
    // TODO(caikun): remove syncStream when cache allocator is ready
    dipu::devproxy::syncStream(stream.rawstream());
  } else {
    dipu::devproxy::syncStream(stream.rawstream());
  }
}

void DIPUCopyInplace::copy_same_dtype(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  dipu::OptionalDIPUGuard device_guard(dst_device.is_cuda() ? dst_device : src_device);

  int64_t nbytes = iter.numel() * iter.element_size(0); 
  dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
  if (dst_device.type() == DIPU_DEVICE_TYPE && src_device.is_cpu()) {
    DEBUG_LOG << "memcopy host to device" << std::endl;
    dipu::devapis::memCopyH2DAsync(stream.rawstream(), nbytes, iter.data_ptr(0), iter.data_ptr(1));
  } else if (dst_device.is_cpu() && src_device.type() == DIPU_DEVICE_TYPE) {
    DEBUG_LOG << "memcopy device to host" << std::endl;
    dipu::devapis::memCopyD2HAsync(stream.rawstream(), nbytes, iter.data_ptr(0), iter.data_ptr(1));
  } else {
    TORCH_CHECK(false, "unsupported devices in copy_");
  }

  if (non_blocking) {
    // TODO(caikun): remove syncStream when cache allocator is ready
    dipu::devproxy::syncStream(stream.rawstream());
  } else {
    dipu::devproxy::syncStream(stream.rawstream());
  }
}

void DIPUCopyInplace::copy_between_host_device(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
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
}

at::Tensor& DIPUCopyInplace::slow_copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  DEBUG_LOG << "enter into " << __FUNCTION__ << std::endl;
  at::Tensor src_cpu = src;
  if (isDeviceTensor(src)) {
    src_cpu = at::empty_strided(src.sizes(), src.strides(),
        src.options().device(c10::DeviceType::CPU));
    copy_from_device_to_host(src_cpu, src, non_blocking);
  }

  if (!isDeviceTensor(dst)) {
    return dst.copy_(src_cpu);
  }

  at::Tensor dst_cpu = at::empty_strided(dst.sizes(), dst.strides(),
      dst.options().device(c10::DeviceType::CPU));
  copy_from_device_to_host(dst_cpu, dst, non_blocking);

  dst_cpu.copy_(src_cpu);
  copy_from_host_to_device(dst, dst_cpu, non_blocking);
  return dst;
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

static int32_t default_init = [&]() {
    setDipuCopyInplace(&default_copy_inplace_op);
    return 1;
}();

}  // namespace dipu