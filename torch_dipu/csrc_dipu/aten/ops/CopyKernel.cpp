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
  // temp solution handle view
  inline int64_t getCopyBytes(const at::Tensor& dst, const at::Tensor& src) {
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
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
    if (dipu::isDeviceTensor(self) && !dipu::isDeviceTensor(src)) {
      if(self.dtype() != src.dtype()) {  // src is cpu.
        // use cpu cast, need enhance
        auto cpu_casted_tensor = src.to(c10::dtype(self.scalar_type()));
        copy_H2D(self, cpu_casted_tensor, non_blocking);
      } else {
        copy_H2D(self, src, non_blocking);
      }
    } 
    else if (!dipu::isDeviceTensor(self) && dipu::isDeviceTensor(src)) {   
      if(self.dtype() != src.dtype()) {  // self is cpu.
        std::vector<at::Tensor> tensor_args;
        tensor_args.push_back(src);
        // use cpu cast, need enhance
        auto cpu_src_tensor = at::_to_cpu(tensor_args)[0].to(c10::dtype(self.scalar_type()));
        // use cpu copy_ as alternative, need enhance.
         self.copy_(cpu_src_tensor);
      } else {
        copy_D2H(self, src, non_blocking);
      }
    }
    else {   // device to device 
      if(self.dtype() != src.dtype()) {  
        std::vector<at::Tensor> tensor_args;
        tensor_args.push_back(src);
        auto cpu_src_tensor = at::_to_cpu(tensor_args)[0].to(c10::dtype(self.scalar_type()));
        copy_(self, cpu_src_tensor, non_blocking);
      } else {
        copy_D2D(self, src, non_blocking);
      }
    }
    return self;
  }
}