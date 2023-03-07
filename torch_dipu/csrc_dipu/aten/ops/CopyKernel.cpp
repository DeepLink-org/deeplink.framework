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
using dipu::devapis::current_device;
using dipu::devapis::deviceId_t;
namespace dipu::native {

  // need abstract cast strategy before copy, some device(eg camb) not support all types,
  inline at::Tensor cast2CompatibleDeviceTensor(const at::Tensor& hostTensor) {
    return hostTensor;
  }
  inline int64_t getCopyBytes(const at::Tensor& dst, const at::Tensor& src) {
    if (dst.nbytes() != src.nbytes()) {  // outer bytes must same. different type is unsuported
      throw std::runtime_error("dipu copy with different size is not allowed"); 
    }
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    // view or expand is supported
    return srcBytes < dstBytes ? srcBytes : dstBytes;
  }

  static void copy_H2D(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = getCopyBytes(dst, src);
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
    int64_t nbytes = getCopyBytes(dst, src);

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

  //  1. expand, 2. patial view. 3. type cast.
  inline bool isStorageSizeDiff(const at::Tensor& dst, const at::Tensor& src) {
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    return srcBytes != dstBytes;
  }

  static void copy_D2D(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    int64_t nbytes = getCopyBytes(dst, src);

    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();

    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();
    // not support between device copy now, need enhance!
    deviceId_t devid = current_device();

    dipu::devapis::memCopyD2DAsync(stream.rawstream(), nbytes, devid, dst_ptr, devid, src_ptr);
    if (non_blocking) {
        DIPU_LOGE("Copy between devices with " \
            "non_blocking is not supported now ");
      dipu::devapis::syncStream(stream.rawstream());
    } else {
      dipu::devapis::syncStream(stream.rawstream());
    }
  }

  inline void doRealCp(at::Tensor& self, const at::Tensor& src,  bool non_blocking) {
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
    if (isStorageSizeDiff(self, src)) {
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
}