#include <ATen/core/NamedTensor.h>
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
using dipu::devapis::MemCPKind;

namespace dipu::native {

  // need abstract cast strategy before copy, some device(eg camb) not support all types,
  inline at::Tensor cast2CompatibleDeviceTensor(const at::Tensor& hostTensor) {
    return hostTensor;
  }

  static void copy_H2D(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = dst.numel() * dst.element_size();
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    auto src_cast = cast2CompatibleDeviceTensor(src);
    void* src_ptr = src_cast.data_ptr();
    void* dst_ptr = dst.data_ptr();

    dipu::devapis::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
    if (non_blocking) {
      /// need add host cache allocator
      dipu::devapis::syncStream(stream.rawstream());
    } else {
      dipu::devapis::syncStream(stream.rawstream());
    }
  }

  static void copy_D2H(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = dst.numel() * dst.element_size();
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    dipu::devapis::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
    if (non_blocking) {
        DIPU_LOGE("Copy data back to CPU device with " \
            "non_blocking is not supported now ");
      dipu::devapis::syncStream(stream.rawstream());
    } else {
      dipu::devapis::syncStream(stream.rawstream());
    }
  }

  // self is dest
  at::Tensor& DIPUATenFunctions::copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (self.numel() == 0) {
      return self;
    }
    // save tensor dim name
    c10::optional<at::DimnameList> names = src.opt_names();
    if (names.has_value()) {
      internal_set_names_inplace(self, names);
    }
    if (dipu::isDeviceTensor(self)) {
      if (!dipu::isDeviceTensor(src)) {
        copy_H2D(self, src, non_blocking);
      } else {
        throw std::runtime_error("not support d2d copy");  
      }
    } else {
      if (dipu::isDeviceTensor(src)) {
        copy_D2H(self, src, non_blocking);
      }
    }
    return self;
  }
}