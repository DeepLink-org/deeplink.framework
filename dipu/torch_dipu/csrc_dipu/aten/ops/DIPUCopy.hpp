// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>

#include "csrc_dipu/profiler/profiler.h"
#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/helpfunc.hpp>

namespace dipu::native {
// these 2 func defined in AutoGenedKernels.cpp
// if dipu autogen support header file gen, remove this
extern at::Tensor dipu_wrap_diopi_cast_dtype(const at::Tensor& src,
                                             at::ScalarType dtype);

// if dipu autogen support proxy one torch op to multiple diopi op, remove this.
extern at::Tensor& dipu_wrap_diopi_copy_inp(at::Tensor& dst,
                                            const at::Tensor& src,
                                            bool non_blocking);

}  // end namespace dipu::native

namespace dipu {

using dipu::native::dipu_wrap_diopi_cast_dtype;
using dipu::native::dipu_wrap_diopi_copy_inp;
using MemoryFormat = at::MemoryFormat;

enum DIPUCopyType {
  // in one device
  D2Self,
  D2OtherD,
  D2H,
  H2D,
};

namespace {
// Align with pytorch's behavior, see TensorIterator.cpp compute_mem_overlaps()
inline void checkOverlap(const at::Tensor& dst, const at::Tensor& src) {
  assert_no_internal_overlap(dst);
  assert_no_partial_overlap(dst, src);
}

inline void try_record_stream(const at::Tensor& tensor, DIPUStream& curStream,
                              bool is_default_stream) {
  if (tensor.is_cpu() && tensor.options().pinned_memory()) {
    tensor.record_stream(curStream);
  } else if (!is_default_stream) {
    tensor.record_stream(curStream);
  }
}

inline DIPUCopyType getCopyType(const at::Tensor& dst, const at::Tensor& src) {
  bool isSrcDevice = dipu::isDeviceTensor(src);
  bool isDstDevice = dipu::isDeviceTensor(dst);
  if (!isSrcDevice) {
    return DIPUCopyType::H2D;  // this op not handle h2h, dest always device
  } else if (!isDstDevice) {
    return DIPUCopyType::D2H;  // here src always device
  } else if (src.device().index() != dst.device().index()) {
    return DIPUCopyType::D2OtherD;
  } else {
    return DIPUCopyType::D2Self;
  }
}

inline int64_t getMemCopyBytes(const at::Tensor& dst, const at::Tensor& src,
                               bool nonOverlappingAndDense) {
  if (dst.nbytes() !=
      src.nbytes()) {  // outer bytes must same. different type is unsuported
    TORCH_CHECK(false, "mem copy with different tensor size is not allowed");
  }
  if (nonOverlappingAndDense) {
    return dst.nbytes();
  } else {
    int64_t dstBytes = dst.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    int64_t srcBytes = src.unsafeGetTensorImpl()->unsafe_storage().nbytes();
    return srcBytes < dstBytes ? srcBytes : dstBytes;
  }
}

inline void memCopyH2D(const at::Tensor& dst, const at::Tensor& src,
                       dipu::DIPUStream& stream, int64_t nbytes) {
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();

  MemChecker::instance().check(dst);
  dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
}

inline void memCopyD2H(const at::Tensor& dst, const at::Tensor& src,
                       dipu::DIPUStream& stream, int64_t nbytes) {
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();

  MemChecker::instance().check(src);
  dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr, src_ptr);
}

inline void memCopyD2D(const at::Tensor& dst, const at::Tensor& src,
                       dipu::DIPUStream& stream, int64_t nbytes) {
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();

  MemChecker::instance().check(src);
  MemChecker::instance().check(dst);
  dipu::devproxy::memCopyD2DAsync(stream.rawstream(), nbytes,
                                  dst.device().index(), dst_ptr,
                                  src.device().index(), src_ptr);
}

inline void memCopy(const at::Tensor& dst, const at::Tensor& src,
                    dipu::DIPUStream& stream, DIPUCopyType copyType,
                    bool nonOverlappingAndDense) {
  int64_t nbytes = getMemCopyBytes(dst, src, nonOverlappingAndDense);
  switch (copyType) {
    case DIPUCopyType::H2D:
      // src is cpu.
      memCopyH2D(dst, src, stream, nbytes);
      break;
    case DIPUCopyType::D2H:
      // dst is cpu.
      memCopyD2H(dst, src, stream, nbytes);
      break;
    default:  // device to device
      memCopyD2D(dst, src, stream, nbytes);
  }
}
}  // anonymous namespace

struct CopyParamsInfo {
  DIPUCopyType copyType_;
  DIPUStream curStream_;
  // basic info
  // if cast needed
  bool sameDtype_ = false;
  // determine if expand needed.
  bool sameSize_ = false;
  bool sameStride_ = false;
  bool denseAndNoOverlap_ = false;

  // composite info, can direct mem copy
  bool directMemCopy_ = false;

  void recomputeTensorsInfo(const at::Tensor& dst, const at::Tensor& src) {
    sameDtype_ = dst.scalar_type() == src.scalar_type();
    sameSize_ = dst.sizes().equals(src.sizes());
    sameStride_ = dst.strides().equals(src.strides());
    denseAndNoOverlap_ = dst.is_non_overlapping_and_dense() &&
                         src.is_non_overlapping_and_dense();
    directMemCopy_ =
        sameDtype_ && sameSize_ && sameStride_ && denseAndNoOverlap_;
  }

  CopyParamsInfo(const at::Tensor& dst, const at::Tensor& src) {
    // assume layout always = not suppport Sparse layout
    TORCH_CHECK(dst.options().layout() == c10::Layout::Strided,
                "only Strided layout is supported");
    copyType_ = getCopyType(dst, src);
    curStream_ = dipu::getCurrentDIPUStream();

    recomputeTensorsInfo(dst, src);
  }

  void updateCopyType(DIPUCopyType copyType) { copyType_ = copyType; }
};

class DIPUCopyBase {
 public:
  DIPUCopyBase() = default;
  virtual ~DIPUCopyBase() = default;
  virtual void run(at::Tensor& dst, const at::Tensor& src,
                   bool non_blocking) = 0;
};

/*
if input, output tensor occupy same storage size and has same mem-format,
DIPUCopyInplace will directly call mem_copy; if not, it call  copyNodirectXX.

DiopiCast: means call separate diopiCast func, it's a forward compatible
solutions because some vendor's DiopiCopy not support cast. new DiopiCopy api
require cast/
*/
template <bool DiopiCopy, bool DiopiCast>
class DIPUCopyInplace : public DIPUCopyBase {
 public:
  DIPUCopyInplace() = default;
  void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) override {
    TORCH_CHECK(dst.defined(), "dst is undefined");
    TORCH_CHECK(src.defined(), "src is undefined");
    if (dst.numel() == 0 || dst.is_same(src)) {
      return;
    }
    auto curStream = dipu::getCurrentDIPUStream();
    // recordBeforeCopy
    if (non_blocking) {
      const bool is_default_stream = dipu::getDefaultDIPUStream() == curStream;
      try_record_stream(dst, curStream, is_default_stream);
      try_record_stream(src, curStream, is_default_stream);
    }
    checkOverlap(dst, src);

    auto info = CopyParamsInfo(dst, src);
    // Exit early if self and src are views of the same data
    bool isSameData =
        (dst.is_alias_of(src) && dst.storage_offset() == src.storage_offset() &&
         info.sameStride_ && info.sameDtype_);
    if (isSameData) {
      return;
    }
    copyAll(dst, src, non_blocking, info);
    // syncAfterCopy
    if (!non_blocking) {
      dipu::devapis::syncStream(curStream.rawstream());
    }
  }

 protected:
  /*
  the memory area of the dst tensor (contains hollow area actually not belong to
  tensor) will be totally overwrited by the same-size src mem area. support copy
  between 2 tensor with same stride and dtype, no-dense and overlapped tensors
  are also supported. but the 2 cannot both be view
  (will casue data outside mem area be overwrited).
  */
  void doDirectMemFill(at::Tensor& dst, const at::Tensor& src,
                       DIPUStream& curStream, DIPUCopyType copyType) {
    if (dst.is_view() && src.is_view()) {
      TORCH_CHECK(false, "doDirectMemFill cannot support all view-view copy");
    }
    memCopy(dst, src, curStream, copyType, false);
  }

  // support mem copy between 2 nonOverlappingAndDense tensor with same stride
  // and dtype. both 2 can be view.
  void doDirectMemCopy(at::Tensor& dst, const at::Tensor& src,
                       DIPUStream& curStream, DIPUCopyType copyType) {
    memCopy(dst, src, curStream, copyType, true);
  }

  at::Tensor makeSameStrideTensor(const at::Tensor& src, DIPUStream& curStream,
                                  at::Device newDevice,
                                  bool willBackfillSrc = false) {
    if (src.is_contiguous(MemoryFormat::ChannelsLast) || src.is_contiguous()) {
      auto sameAsSrc = at::empty(src.sizes(), src.options().device(newDevice),
                                 src.suggest_memory_format());
      return sameAsSrc;
    } else {
      // empty_strided is much expensive than empty_memory_format().
      // see src/ATen/EmptyTensor.cpp computeStorageNbytes()
      auto sameAsSrc = at::empty_strided(src.sizes(), src.strides(),
                                         src.options().device(newDevice));
      // prefill newTensor to support backfill in future.
      if (willBackfillSrc && !src.is_non_overlapping_and_dense()) {
        doDirectMemFill(sameAsSrc, src, curStream, getCopyType(sameAsSrc, src));
      }
      return sameAsSrc;
    }
  }

  // this func maximize leverage device copy (D2Self)
  // as relay in d2h, h2d, d2d copy. cannot used in device copy(D2Self).
  void doDeviceRelayCopy(at::Tensor& dst, const at::Tensor& src,
                         bool non_blocking, CopyParamsInfo& info) {
    switch (info.copyType_) {
      // create dst_device (relay, same stride)
      // 1. direct dst_cpu/otherdevice -> dst_device (is view).
      // 2. src_device -> dst_device. 3. direct dst_device ->
      // dst_cpu/otherdevice.
      case DIPUCopyType::D2OtherD:
      case DIPUCopyType::D2H: {
        auto curCopyType = info.copyType_;
        // same stride as dst.
        auto dstInDevSrc =
            makeSameStrideTensor(dst, info.curStream_, src.device(), true);
        info.updateCopyType(DIPUCopyType::D2Self);
        copyNodirectOnDevice(dstInDevSrc, src, non_blocking, info);
        doDirectMemFill(dst, dstInDevSrc, info.curStream_, curCopyType);
      } break;
      // create src_device (relay, same stride)
      // direct src_cpu -> src_device, src_device -> dst(device)
      case DIPUCopyType::H2D: {
        auto srcInDstdev =
            makeSameStrideTensor(src, info.curStream_, dst.device());
        doDirectMemFill(srcInDstdev, src, info.curStream_, DIPUCopyType::H2D);
        info.updateCopyType(DIPUCopyType::D2Self);
        copyNodirectOnDevice(dst, srcInDstdev, non_blocking, info);
      } break;
      default:
        TORCH_CHECK(false,
                    "doDeviceRelayCopy not support one device device, it's a "
                    "proxy method");
    }
  }

  // doDeviceRelayCopy need create a relay tensor having same stride as the
  // dst/src. it's expensive if the tensor is a view with big hollow, so supply
  // this simple wrap method to help d2h, h2d, d2d copy. cannot used in device
  // copy(D2Self). logical approach:
  // 1. create dst_contig. 2. create src_contig and src -> src_contig.
  // 3. direct src_contig -> dst_contig  4. dst_contig -> dst
  //  (todo: automatic use)
  void doContigTensorRelayCopy(at::Tensor& dst, const at::Tensor& src,
                               bool non_blocking, CopyParamsInfo& info) {
    switch (info.copyType_) {
      case DIPUCopyType::D2OtherD:
      case DIPUCopyType::D2H: {
        // 1. create dst_contig. same device.
        auto dstContig =
            dst.is_contiguous()
                ? dst
                : at::empty_like(dst, c10::MemoryFormat::Contiguous);
        auto newInfo = CopyParamsInfo(dstContig, src);
        if (newInfo.directMemCopy_) {
          doDirectMemCopy(dstContig, src, newInfo.curStream_,
                          newInfo.copyType_);
        } else {
          // equivalent as logical approach:
          // 2. create dst contig in src Device and do src -> dst_contigs_2.
          // 3. direct: dst_contigs_2(D) -> dst_contig(cpu/otherD).
          doDeviceRelayCopy(dstContig, src, non_blocking, newInfo);
        }
        // 4. dst_contig -> dst (in same device/cpu), this operation need
        // recurse call kernel, direcet copy cannot handle it.
        if (!dstContig.is_same(dst)) {
          dst.copy_(dstContig);
        }
      } break;
      case DIPUCopyType::H2D: {
        // 2. create src_contig and src -> src_contig (both cpu).
        auto srcContig = src.contiguous(c10::MemoryFormat::Contiguous);
        auto newInfo = CopyParamsInfo(dst, srcContig);
        if (newInfo.directMemCopy_) {
          doDirectMemCopy(dst, srcContig, newInfo.curStream_,
                          newInfo.copyType_);
        }
        // equivalent as logical approach:
        // 1. create src_contig_2(D).  3. direct src_contig(CPU) -> src_contig_2
        // (D).
        // 4. src_contig_2 (device) -> dst (device),
        doDeviceRelayCopy(dst, srcContig, non_blocking, newInfo);
      } break;
      default:
        TORCH_CHECK(false,
                    "doDeviceRelayCopy not support one device device, it's a "
                    "proxy method");
    }
  }

  /*
  d2h: direct src (device) -> src_cpu. src_cpu -> dst (cpu)
  h2d: direct dst (device) -> dst_cpu (if view).. src (cpu)  -> dst_cpu.
       direct dst_cpu -> dst (device)
  d2d: direct src (device) -> src_cpu. direct dst (device) -> dst_cpu (if
  view). src_cpu -> dst_cpu.  direct dst_cpu -> dst (device), very very
  slow. this can handle any case, it's fallback solution.
 */
  void doCpuRelayCopy(at::Tensor& dst, const at::Tensor& src,
                      DIPUStream& curStream, bool non_blocking) {
    at::Tensor src_cpu = src;
    if (dipu::isDeviceTensor(src)) {
      auto src_cpu = makeSameStrideTensor(src, curStream,
                                          c10::Device(c10::DeviceType::CPU));
      // src storage size may bigger than src_cpu's when src is a partial view.
      // but not smaller. because src_cpu use same stride as src.
      // src -> src_cpu
      doDirectMemFill(src_cpu, src, curStream, DIPUCopyType::D2H);
    }

    if (dipu::isDeviceTensor(dst)) {
      auto dst_cpu = makeSameStrideTensor(
          dst, curStream, c10::Device(c10::DeviceType::CPU), true);
      // proxy to cpu to handle different type/view problem
      dst_cpu.copy_(src_cpu);

      doDirectMemFill(dst, dst_cpu, curStream, DIPUCopyType::H2D);
    } else {  // dst is cpu
      dst.copy_(src_cpu);
    }
  }

  // handle no-direct mem copy on one device, dipu has a simple configurable
  // template strategy which use DIOPI copy/cast correctly. if vendor has
  // no-complete implementation of DIOPI copy/cast. please override
  // copy_nodirect_device to decide the case needed to be executed by diopiCopy
  // and proxy other case back to 'doCpuRelayCopy' which contain a slow
  // implementaion.

  // 1. type cast. 2. expand/bcast. 3.1 special no-contiguous (stride
  // hollow/overlap) 3.2 mem-format no-contiguous (storage contiguous but not
  // nchw), we don't handle this
  virtual void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                                    bool non_blocking, CopyParamsInfo& info) {
    if (!DiopiCast && !DiopiCopy) {
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    }
    auto tmpSrc = src;
    if (DiopiCast) {
      if (!info.sameDtype_) {
        tmpSrc = dipu_wrap_diopi_cast_dtype(src, dst.scalar_type());
        info.recomputeTensorsInfo(dst, tmpSrc);
      }
      if (info.directMemCopy_) {
        doDirectMemCopy(dst, tmpSrc, info.curStream_, info.copyType_);
      } else {  // need further convert
        dipu_wrap_diopi_copy_inp(dst, tmpSrc, non_blocking);
      }
    } else {
      dipu_wrap_diopi_copy_inp(dst, tmpSrc, non_blocking);
    }
  }

  // handle no-direct mem copy between different devices, dipu has default
  // strategy which use a intermidiate tensor, it's slow. vendor who has more
  // efficient p2p device copy can override it (eg: device has unified
  // addressing and supports passing in different device addresses to one kernel
  // can use copyNodirectOnDevice() to do 'between-device-copy')
  virtual void copyNodirectBetweenDevices(at::Tensor& dst,
                                          const at::Tensor& src,
                                          bool non_blocking,
                                          CopyParamsInfo& info) {
    if (DiopiCopy) {
      // doContigTensorRelayCopy(dst, src, non_blocking, info);
      doDeviceRelayCopy(dst, src, non_blocking, info);
    } else {  // if diopiCopy = false, direct do cpu copy is best.
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    }
  }

  // copy no-direct mem copy between cpu and device, dipu has default strategy
  // use intermidiate tensor, it's slow. vendor who has more efficient solution
  // can override it.
  virtual void copyNodirectDeviceHost(at::Tensor& dst, const at::Tensor& src,
                                      bool non_blocking, CopyParamsInfo& info) {
    if (DiopiCopy) {  // try to maximum leverage device copy,
      // doContigTensorRelayCopy(dst, src, non_blocking, info);
      doDeviceRelayCopy(dst, src, non_blocking, info);
    } else {  // if diopiCopy = false, direct do cpu copy is best.
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    }
  }

  // overriding this func is possible but not recommended
  virtual void copyAll(at::Tensor& dst, const at::Tensor& src,
                       bool non_blocking, CopyParamsInfo& info) {
    at::Tensor tmpSrc = src;
    if (!info.sameSize_) {
      tmpSrc = src.expand_as(dst);
      info.recomputeTensorsInfo(tmpSrc, dst);
    }
    if (info.directMemCopy_) {
      doDirectMemCopy(dst, tmpSrc, info.curStream_, info.copyType_);
    } else {
      switch (info.copyType_) {
        case DIPUCopyType::D2Self:
          copyNodirectOnDevice(dst, tmpSrc, non_blocking, info);
          break;
        case DIPUCopyType::D2OtherD:
          copyNodirectBetweenDevices(dst, tmpSrc, non_blocking, info);
          break;
        default:
          copyNodirectDeviceHost(dst, tmpSrc, non_blocking, info);
      }
    }
  }
};

DIPUCopyBase* getDipuCopyClass();

void setDipuCopyClass(DIPUCopyBase* op);

}  // namespace dipu