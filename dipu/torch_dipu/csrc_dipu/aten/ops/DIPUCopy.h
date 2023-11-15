// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/helpfunc.hpp>
#include "csrc_dipu/profiler/profiler.h"

namespace dipu::native {
// these 2 func defined in AutoGenedKernels.cpp
// if dipu autogen support header file gen, remove this 
extern at::Tensor dipu_wrap_diopi_cast_dtype(const at::Tensor& src, at::ScalarType dtype);

// if dipu autogen support proxy one torch op to multiple diopi op, remove this. 
extern at::Tensor& dipu_wrap_diopi_copy_inp(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

} // end namespace dipu::native

namespace dipu {

using dipu::native::dipu_wrap_diopi_cast_dtype;
using dipu::native::dipu_wrap_diopi_copy_inp;

enum DIPUCopyType {
  // in one device
  D2Self,
  D2OtherD,
  D2H,
  H2D,
};

struct CopyParamsInfo {
  DIPUCopyType copyType;
  DIPUStream curStream;
  // basic info
  // if cast needed
  bool sameDtype = false;
  // if stride based copy need
  bool allContiguous = false;
  // if bcast needed.
  bool bcastCopy = false;

  // composite info
  // can direct mem copy 
  bool directCopy = false;
};

namespace {
  inline void try_record_stream(const at::Tensor& tensor, DIPUStream& curStream, bool is_default_stream) {
    if (tensor.is_cpu()) {
      if (tensor.options().pinned_memory()) {
        tensor.record_stream(curStream);
      }
    } else if (!is_default_stream) {
      tensor.record_stream(curStream);
    }
  }

  inline DIPUCopyType getCopyType(const at::Tensor& dst, const at::Tensor& src) {
    bool isSrcDevice = dipu::isDeviceTensor(src);
    bool isDstDevice = dipu::isDeviceTensor(dst);
    if (!isSrcDevice) {
      return DIPUCopyType::H2D;
    }
    if (!isDstDevice) {
      return DIPUCopyType::D2H;
    }
    else if (src.device().index() != dst.device().index()) {
      return DIPUCopyType::D2OtherD;
    }
    return DIPUCopyType::D2Self;
  }

  inline CopyParamsInfo getParamsInfo(const at::Tensor& dst, const at::Tensor& src) {
    // assume layout always = not suppport Sparse layout
    TORCH_CHECK(dst.options().layout() == c10::Layout::Strided, "only Strided layout is supported");
    CopyParamsInfo info;
    info.sameDtype = (dst.scalar_type() == src.scalar_type());
    info.allContiguous = (dst.is_contiguous() && src.is_contiguous());
    info.bcastCopy = (dst.numel() != src.numel());
    info.directCopy = info.sameDtype && info.allContiguous && !info.bcastCopy;
    info.copyType = getCopyType(dst, src);
    info.curStream = dipu::getCurrentDIPUStream();
    return info;
  }


  inline int64_t getDirectCopyBytes(const at::Tensor& dst, const at::Tensor& src) {
    if (dst.nbytes() != src.nbytes()) {  // outer bytes must same. different type is unsuported
      TORCH_CHECK(false, "dipu copy with different size is not allowed");
    }
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    return srcBytes < dstBytes ? srcBytes : dstBytes;
  }


  inline void copyH2D(const at::Tensor& dst, const at::Tensor& src, dipu::DIPUStream& stream) {
    int64_t nbytes = getDirectCopyBytes(dst, src);
    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(dst);
    dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
  }

  inline void copyD2H(const at::Tensor& dst, const at::Tensor& src, dipu::DIPUStream& stream) {
    int64_t nbytes = getDirectCopyBytes(dst, src);
    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(src);
    dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
  }

  inline void copyD2D(const at::Tensor& dst, const at::Tensor& src, dipu::DIPUStream& stream) {
    int64_t nbytes = getDirectCopyBytes(dst, src);
    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    MemChecker::instance().check(src);
    MemChecker::instance().check(dst);
    dipu::devproxy::memCopyD2DAsync(stream.rawstream(), nbytes, dst.device().index(), dst_ptr,
                                   src.device().index(), src_ptr);
  }
} // anonymous namespace


class DIPUCopyBase {
public:
  DIPUCopyBase() = default;
  virtual ~DIPUCopyBase() = default;

  virtual void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) = 0;
 
};

/*
if input, output tensor occupy same storage size and has same mem-format, DIPUCopyInplace
will directly call mem_copy; if not, it call  copyNodirectXX.

DiopiCast: means call separate diopiCast func, it's a forward compatible solutions because some vendor's
DiopiCopy not support cast. new DiopiCopy api require cast:

DiopiCopyBcast: if vendor diopiCopy cannot do bcast copy (false). set this to False, DIPUCopyInplace will
try to expand tensor to same size before call diopiCopy, it's not yet implemented. always true now.
*/ 
template <bool DiopiCopy, bool DiopiCast, bool DiopiCopyBcast=true>
class DIPUCopyInplace : public DIPUCopyBase {
private:
  bool useDiopiCopy_ = DiopiCopy;
  bool useDiopiCast_ = DiopiCast;
  bool diopiSupportBcast = true;

public:
  DIPUCopyInplace() = default;
  void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) override {
    TORCH_CHECK(dst.defined(), "dst is undefined");
    TORCH_CHECK(src.defined(), "src is undefined");

    auto paramsInfo = getParamsInfo(dst, src);
    // recordBeforeCopy
    if (non_blocking) {
      const bool is_default_stream = dipu::getDefaultDIPUStream() == paramsInfo.curStream;
      try_record_stream(dst, paramsInfo.curStream, is_default_stream);
      try_record_stream(src, paramsInfo.curStream, is_default_stream);
    }

    copyAll(dst, src, non_blocking, paramsInfo);

    // syncAfterCopy
    if (!non_blocking) {
      dipu::devapis::syncStream(paramsInfo.curStream.rawstream());
    }
  }

protected:
  // support view <-> real-stor tensor and real <-> real tensor copy.
  // not support all view <-> view copy (may casue extra data to be copied)
  void doDirectCopy(at::Tensor& dst, const at::Tensor& src,
                    DIPUStream& curStream, DIPUCopyType copyType, bool strictCheck = false) {
    // check is time consuming in fast path, default false
    if (strictCheck && dst.is_view() && src.is_view()) {
      TORCH_CHECK(false, "doDirectCopy cannot support all view-view copy");
    }

    switch(copyType) {
      case DIPUCopyType::H2D:
        // src is cpu.
        copyH2D(dst, src, curStream);
        break;
      case DIPUCopyType::D2H:
        // dst is cpu.
        copyD2H(dst, src, curStream);
        break;
      default:// device to device
        copyD2D(dst, src, curStream);
        break;
    }
  }

   /*
   d2h: direct src (device) -> src_cpu. src_cpu -> dst (cpu)
   h2d: direct dst (device) -> dst_cpu (if view).. src (cpu)  -> dst_cpu.
        direct dst_cpu -> dst (device)
   d2d: direct src (device) -> src_cpu. direct dst (device) -> dst_cpu (if view).
        src_cpu -> dst_cpu.  direct dst_cpu -> dst (device), very very slow.
  */
  void doCpuIntermediateCopy(at::Tensor& dst, const at::Tensor& src,
                             DIPUStream& curStream, bool non_blocking) {
    at::Tensor src_cpu = src;
    if(dipu::isDeviceTensor(src)) {
      // empty_strided can handle both hollow and overlap no-contiguous case and create correct
      // size tensor. it's the basis why we can do direct copy between 2 no-contiguous tensor.
      src_cpu = at::empty_strided(src.sizes(), src.strides(),
          src.options().device(c10::DeviceType::CPU));
      // src storage size may bigger than src_cpu's when src is a partial view.
      // but not smaller. because src_cpu use same stride as src.
      // src -> src_cpu
      doDirectCopy(src_cpu, src, curStream, DIPUCopyType::D2H);
    }

    if(dipu::isDeviceTensor(dst)) {
      at::Tensor dst_cpu = at::empty_strided(dst.sizes(), dst.strides(),
            dst.options().device(c10::DeviceType::CPU));
      if (dst.is_view()) {
        doDirectCopy(dst_cpu, dst, curStream, DIPUCopyType::D2H);
      }
      // proxy to cpu to handle different type/view problem
      dst_cpu.copy_(src_cpu);

      doDirectCopy(dst, dst_cpu, curStream, DIPUCopyType::H2D);
    } else {  // dst is cpu
      dst.copy_(src_cpu);
    }
  }

  // handle no-direct mem copy on one device, dipu has a simple configurable template strategy which 
  // use DIOPI copy/cast correctly. if vendor has no-complete implementation of DIOPI copy/cast.
  // please override copy_nodirect_device to decide the case needed to be executed by diopiCopy
  // and proxy other case back to 'doCpuIntermediateCopy' which contain a slow implementaion.

  // 1. type cast. 2. expand/bcast. 3.1 simple no-contiguous (storage contiguous but not nhwc)
  // 3.2 special no-contiguous (partial/hollow/overlap) 
  virtual void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                                    bool non_blocking, CopyParamsInfo& info) {
    if (!DiopiCast && !DiopiCopy) {
      doCpuIntermediateCopy(dst, src, info.curStream, non_blocking);
    }
    auto tmpSrc = src;
    if (DiopiCast) {
      if (!info.sameDtype) {
        tmpSrc = dipu_wrap_diopi_cast_dtype(src, dst.scalar_type());
      }
      // need further convert
      if (!info.allContiguous || info.bcastCopy) {
        dipu_wrap_diopi_copy_inp(dst, tmpSrc, non_blocking);
      } else {
        doDirectCopy(dst, tmpSrc, info.curStream, info.copyType);
      }
    } else {
      dipu_wrap_diopi_copy_inp(dst, tmpSrc, non_blocking);
    }
  }

  // handle no-direct mem copy between different devices, dipu has default strategy which use a
  // intermidiate tensor, it's slow. vendor who has more efficient p2p device copy can override it 
  // (eg: device has unified addressing and supports passing in different device addresses to one kernel
  // can direct use copyNodirectOnDevice() to do 'between-device-copy')
  virtual void copyNodirectBetweenDevices(at::Tensor& dst, const at::Tensor& src,
                                          bool non_blocking, CopyParamsInfo& info) {
    if (DiopiCopy) { 
      // need 3 steps: direct dst -> dst_in_src_device (if view).
      // then src -> dst_in_src_device.  direct dst_in_src_device -> dst.
      // but it still better than doCpuIntermediateCopy
      auto dstInDevSrc = at::empty_strided(dst.sizes(), dst.strides(), dst.options().device(src.device()));
      if (dst.is_view()) {
        doDirectCopy(dstInDevSrc, dst, info.curStream, DIPUCopyType::D2OtherD);
      }
      info.copyType = DIPUCopyType::D2Self;
      copyNodirectOnDevice(dstInDevSrc, src, non_blocking, info);
      doDirectCopy(dst, dstInDevSrc, info.curStream, DIPUCopyType::D2OtherD);
     
    } else { // if diopiCopy = false, direct do cpu copy is best.
      doCpuIntermediateCopy(dst, src, info.curStream, non_blocking);
    }
  }

  // copy no-direct mem copy between cpu and device, dipu has default strategy use
  // intermidiate tensor, it's slow. vendor who has more efficient solution can override it.
  virtual void copyNodirectDeviceHost(at::Tensor& dst, const at::Tensor& src,
                                      bool non_blocking, CopyParamsInfo& info) {
    if (DiopiCopy) {
      switch(info.copyType) {
        case DIPUCopyType::H2D: {
          // direct src_cpu -> src_device, src_device -> dst(device)
          auto srcIndev = at::empty_strided(src.sizes(), src.strides(), src.options().device(dst.device()));
          doDirectCopy(srcIndev, src, info.curStream, DIPUCopyType::H2D);
          info.copyType = DIPUCopyType::D2Self;
          copyNodirectOnDevice(dst, srcIndev, non_blocking, info);
          break;
        }
        // if dst tensor is view, leverage device copy need 3 steps: 1. direct dst_cpu -> dst_device.
        // 2. src_device -> dst_device. 3. direct dst_device -> dst_cpu. do stride based copy on device.
        // but doCpuIntermediateCopy D2H only need 2 step: direct src (device) -> src_cpu. src_cpu -> dst(cpu)
        // do stride based copy on cpu (slow). not sure which is better.
        case DIPUCopyType::D2H: {
          if (dst.is_view()) {
            doCpuIntermediateCopy(dst, src, info.curStream, non_blocking);
          } else {
            auto dstIndev = at::empty_strided(dst.sizes(), dst.strides(), dst.options().device(src.device()));
            info.copyType = DIPUCopyType::D2Self;
            copyNodirectOnDevice(dstIndev, src, non_blocking, info);
            doDirectCopy(dst, dstIndev, info.curStream, DIPUCopyType::D2H);
          }
          break;
        }
        default:
          TORCH_CHECK(false, "call copyNodirectDeviceHost with incorrect copy type");
      }
    } else { // if diopiCopy = false, direct do cpu copy is best.
        doCpuIntermediateCopy(dst, src, info.curStream, non_blocking);
    }
  }

  // overriding this func is possible but not recommended
  virtual void copyAll(at::Tensor& dst, const at::Tensor& src,
                      bool non_blocking, CopyParamsInfo& info) {
    if (info.directCopy) {
      doDirectCopy(dst, src, info.curStream, info.copyType);
    } else {
      switch(info.copyType) {
        case DIPUCopyType::D2Self:
          copyNodirectOnDevice(dst, src, non_blocking, info);
          break;
        case DIPUCopyType::D2OtherD:
          copyNodirectBetweenDevices(dst, src, non_blocking, info);
          break;
        default:
          copyNodirectDeviceHost(dst, src, non_blocking, info);
      }
    }  
  }
};

DIPUCopyBase* getDipuCopyClass();

void setDipuCopyClass(DIPUCopyBase *op);

}  // namespace dipu