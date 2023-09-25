// Copyright (c) 2023, DeepLink.
#include <ATen/core/NamedTensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/accumulate.h>
#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/Layout.h>
#include <ATen/Dispatch.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/runtime/core/MemChecker.h>
#include <csrc_dipu/utils/helpfunc.hpp>

using c10::device_or_default;
using c10::layout_or_default;
using c10::StorageImpl;
using c10::TensorImpl;
using at::Layout;
using dipu::devapis::deviceId_t;
using c10::IntArrayRef;

namespace dipu::native {

  // need abstract cast strategy before copy, some device(eg camb) not support all types,
  inline at::Tensor cast2CompatibleDeviceTensor(const at::Tensor& hostTensor) {
    return hostTensor;
  }
  inline int64_t getCopyBytes(const at::Tensor& dst, const at::Tensor& src) {
    if (dst.nbytes() != src.nbytes()) {  // outer bytes must same. different type is unsuported
      TORCH_CHECK(false, "dipu copy with different size is not allowed");
    }
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    // a view one +  a real stor one  is supported
    return srcBytes < dstBytes ? srcBytes : dstBytes;
  }

  static void copy_H2D(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = getCopyBytes(dst, src);
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    auto src_cast = cast2CompatibleDeviceTensor(src);
    void* src_ptr = src_cast.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(dst);
    dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
    if (!non_blocking) {
      dipu::devproxy::syncStream(stream.rawstream());
    }
  }

  static void copy_D2H(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = getCopyBytes(dst, src);
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(src);
    dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
    if (!non_blocking) {
      dipu::devproxy::syncStream(stream.rawstream());
    }
  }

  inline bool isDiffStrides(const IntArrayRef stride1, const IntArrayRef stride2) {
    if (stride1.size() != stride2.size()) {
      return true;
    }
    for (auto i = 0; i < stride1.size() ; i++ ) {
      if (stride1[i] != stride2[i]) {
        return true;
      }
    }
    return false;
  }

  //  1. expand, 2. patial view. 3. type cast.
  inline bool canDirectCopy(const at::Tensor& dst, const at::Tensor& src) {
    // assume layout always = not suppport Sparse layout
    TORCH_CHECK(dst.options().layout() == c10::Layout::Strided, "only Strided layout is supported");

    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    if (srcBytes != dstBytes || dst.numel() != src.numel() || dst.options().dtype() != src.options().dtype()) {
      return false;
    }
    if (isDiffStrides(dst.strides(), src.strides())) {
      return false;
    }
    // view(with no-zero offset) direct copy may cause err(not sure how long real stor data should be copyed) not supported
     if (dst.storage_offset() != 0 || src.storage_offset() != 0) {
      return false;
    }
    // even tensors have zero offset and same stride/type cannot do simple safe direct copy
    // because we cannot simply decide how much data will be copyed from raw stor (unless check stride).
    // so we always return false now.
    // need enhance in future, because always copy with the help of cpu is toooo0 slow.
    // **** check if copy safely using tensor.nbytes() when is_contiguous() = true.
    return false;
  }

  static void copy_D2D(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = getCopyBytes(dst, src);
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(src);
    MemChecker::instance().check(dst);
    dipu::devproxy::memCopyD2DAsync(stream.rawstream(), nbytes, dst.device().index(), dst_ptr,
                                   src.device().index(), src_ptr);
    if (!non_blocking) {
      dipu::devproxy::syncStream(stream.rawstream());
    }
  }

  inline void doRealCp(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (dipu::isDeviceTensor(self) && !dipu::isDeviceTensor(src)) {
        // src is cpu.
        copy_H2D(self, src, non_blocking);
    }
    else if (!dipu::isDeviceTensor(self) && dipu::isDeviceTensor(src)) {
      // self is cpu.
      copy_D2H(self, src, non_blocking);
    }
    else {   // device to device
      copy_D2D(self, src, non_blocking);
    }
  }

  // self is dest
  // not handle storage offset, need?
  at::Tensor& DIPUATenFunctions::copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (self.numel() == 0) {
      return self;
    }
    // save tensor dim name
    c10::optional<at::DimnameList> names = src.opt_names();
    if (names.has_value()) {
      internal_set_names_inplace(self, names);
    }
    if (!canDirectCopy(self, src)) {
      at::Tensor src_cpu = src;
      // src to cpu
      if (dipu::isDeviceTensor(src)) {
        src_cpu = at::empty_strided(src.sizes(), src.strides(),
             src.options().device(c10::DeviceType::CPU));
        // src storage size may bigger than src_cpu's  if src is a partial view.
        // but not smaller. because src_cpu use same stride as src.
        // src -> src_cpu
        doRealCp(src_cpu, src, non_blocking);
      }

      if(dipu::isDeviceTensor(self)) {
        at::Tensor dst_cpu = at::empty_strided(self.sizes(), self.strides(),
              self.options().device(c10::DeviceType::CPU));
        doRealCp(dst_cpu, self, non_blocking);
        // proxy to cpu to handle different type/view problem
        dst_cpu.copy_(src_cpu);

        doRealCp(self, dst_cpu, non_blocking);
      } else {  // self is cpu
        self.copy_(src_cpu);
      }
    } else {
      doRealCp(self, src, non_blocking);
    }
    return self;
  }

  at::Scalar DIPUATenFunctions::_local_scalar_dense_dipu(const at::Tensor& self) {
    at::Scalar r;
    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBool, self.scalar_type(), "_local_scalar_dense_dipu", [&] {
          scalar_t value;
          dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
          MemChecker::instance().check(self);
          dipu::devproxy::memCopyD2HAsync(stream.rawstream(), sizeof(scalar_t), &value, self.data_ptr<scalar_t>());
          dipu::devproxy::syncStream(stream.rawstream());
          r =  at::Scalar(value);
        });
    return r;
  }
}